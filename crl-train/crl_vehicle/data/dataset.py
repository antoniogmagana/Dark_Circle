"""SensorDataset and StratifiedPairDataset for CRL training."""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from crl_vehicle.config import (
    CRLConfig, LABEL_BACKGROUND, LABEL_MULTI, CATEGORY_TO_IDX, DATASET_VEHICLE_MAP
)
from crl_vehicle.data.transforms import remove_dc

# Stratum identifiers for partner sampling
STRATUM_CONSEC    = 0
STRATUM_SAME_TYPE = 1
STRATUM_DIFF_TYPE = 2
STRATUM_CROSS_DS  = 3

# Known dataset names
_KNOWN_DATASETS = {"iobt", "focal", "m3nvc"}

# Per-dataset source sample rates. Mirrors server-load/sample_parse.py:22-23.
# Hardcoded because parquets carry no rate metadata; recordings were captured at
# different rates per dataset and resampled to canonical rates at load time.
_SOURCE_RATES = {
    "focal": {"audio": 16000, "seismic": 100},
    "iobt":  {"audio": 16000, "seismic": 100},
    "m3nvc": {"audio": 1600,  "seismic": 200},
}


def _source_sample_rate(stem: str, sensor: str) -> int:
    """Return the source (on-disk) sample rate for a parquet, by dataset prefix."""
    dataset = stem.split("_", 1)[0]
    rates = _SOURCE_RATES.get(dataset)
    if rates is None:
        raise ValueError(f"Unknown dataset prefix in stem {stem!r}; "
                         f"expected one of {sorted(_SOURCE_RATES)}")
    return rates[sensor]


def _resample_to_target(arr: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample a 1-D float32 signal from source_rate to target_rate.

    No-op if rates match. Uses torchaudio.functional.resample with the default
    Kaiser window. Returns float32 numpy array.
    """
    if source_rate == target_rate:
        return arr
    import torchaudio.functional as AF
    t = torch.from_numpy(arr).float().unsqueeze(0)  # (1, N)
    out = AF.resample(t, orig_freq=source_rate, new_freq=target_rate)
    return out.squeeze(0).numpy().astype(np.float32, copy=False)


# Vehicle name → (vehicle_type_idx, is_valid)
# vehicle_type_idx follows CATEGORY_TO_IDX; background/multi use LABEL_* constants
def _vehicle_to_labels(dataset: str, vehicle: str) -> tuple[int, bool]:
    """Return (vehicle_type_idx, is_valid) for a (dataset, vehicle) pair.

    Looks up DATASET_VEHICLE_MAP[dataset][vehicle]. Multi-vehicle stems
    (containing '_' in m3nvc) are excluded. Unknown dataset/vehicle pairs
    are excluded.
    """
    if dataset == "m3nvc":
        if "_" in vehicle:  # multi-vehicle compound stem — excluded
            return LABEL_MULTI, False
        if vehicle == "background":
            return LABEL_BACKGROUND, True
        ds_map = DATASET_VEHICLE_MAP.get("m3nvc", {})
        entry = ds_map.get(vehicle)
        if entry is None:
            return -99, False
        return CATEGORY_TO_IDX[entry[0]], True

    ds_map = DATASET_VEHICLE_MAP.get(dataset, {})
    entry = ds_map.get(vehicle)
    if entry is None:
        return -99, False
    return CATEGORY_TO_IDX[entry[0]], True


def _read_parquet_numpy(
    path: Path, target_window_size: int, source_rate: int, target_rate: int,
) -> np.ndarray:
    """Read a flat time-series parquet → float32 array (N_windows, target_window_size).

    Expects an 'amplitude' signal column at source_rate. Resamples to target_rate
    if they differ, then reshapes into 1-second windows (target_window_size = target_rate).
    Trailing samples that don't fill a complete window are discarded.
    """
    col = pq.read_table(path, columns=["amplitude"], use_threads=True)
    arr = col.column("amplitude").to_numpy().astype(np.float32)
    arr = _resample_to_target(arr, source_rate, target_rate)
    n_windows = len(arr) // target_window_size
    return arr[: n_windows * target_window_size].reshape(n_windows, target_window_size)


def _read_parquet_present(
    path: Path, source_rate: int,
) -> np.ndarray:
    """Read the 'present' boolean column → per-window majority vote.

    Returns bool array of shape (N_windows,) where each entry corresponds to a
    1-second window. A window is True iff strictly more than 50% of its source
    samples have present=True. Computed at source_rate (no resampling needed for
    bools): a 1-second window contains source_rate samples on disk regardless of
    the target rate the amplitude is resampled to.
    """
    col = pq.read_table(path, columns=["present"], use_threads=True)
    arr = col.column("present").to_numpy()
    n_windows = len(arr) // source_rate
    arr = arr[: n_windows * source_rate].reshape(n_windows, source_rate)
    return arr.mean(axis=1) > 0.5


def _parse_stem(stem: str, sensor: str) -> tuple[str, str, str] | None:
    """Parse '{dataset}_{sensor}_{vehicle}_{rs}' → (dataset, vehicle, rs_node) or None."""
    parts = stem.split("_")
    if len(parts) < 4:
        return None
    ds = parts[0]
    if ds not in _KNOWN_DATASETS:
        return None
    if parts[1] != sensor:
        return None
    # rs node is the last part, vehicle is everything in between
    rs = parts[-1]
    if not rs.startswith("rs"):
        return None
    vehicle = "_".join(parts[2:-1])
    return ds, vehicle, rs


# ---------------------------------------------------------------------------
# SensorDataset
# ---------------------------------------------------------------------------

class SensorDataset(Dataset):
    """Loads pre-windowed audio and seismic parquet files.

    Parquet filename pattern: {dataset}_{sensor}_{vehicle}_{rs_node}.parquet
    Each row is one window of sensor data.

    __getitem__ returns a dict with:
      x_audio:         (1, W_audio) float32 tensor
      x_seismic:       (1, W_seismic) float32 tensor
      audio_avail:     bool
      seismic_avail:   bool
      vehicle_type:    int
      detection_label: int  (1 if vehicle present, 0 if background)
      segment_id:      int
    """

    def __init__(
        self,
        parquet_dir: str | Path,
        config: CRLConfig,
        is_train: bool = True,
        cache_dir: Path | None = None,
        *,
        use_id_split: bool = False,
        role: str = "train",
        id_root: str | Path | None = None,
        id_cache_dir: Path | None = None,
    ) -> None:
        self.parquet_dir = Path(parquet_dir)
        self.cfg = config
        self.is_train = is_train
        self.use_id_split = use_id_split
        self.role = role
        self.id_root = Path(id_root) if id_root is not None else None
        self.id_cache_dir = Path(id_cache_dir) if id_cache_dir is not None else None
        # metadata store: (stem, seg_key) → {path, n_windows}
        self._cache: dict[str, dict] = {"audio": {}, "seismic": {}}
        # Shared-memory tensor store: (stem, seg_key) → Tensor(N, W) in shared memory.
        # Populated once at init, shared across all DataLoader workers with zero duplication.
        self._data_cache: dict[str, dict] = {"audio": {}, "seismic": {}}
        self._index: list = []   # [(gkey, w_idx, vtype, det_label, audio_seg_id, seismic_seg_id)]
        self._groups: dict = {}  # gkey → group metadata
        # id_split skip counters: reason → count
        self._id_skip_counts: dict[str, int] = {}

        if self.use_id_split:
            if self.id_root is None:
                raise ValueError("use_id_split=True requires id_root")
            if self.role not in ("train", "val", "test"):
                raise ValueError(f"role must be 'train'|'val'|'test' (got {self.role!r})")
            logging.getLogger(__name__).warning(
                f"use_id_split=True; parquet_dir={self.parquet_dir!s} is ignored "
                f"(splits read from DATASET_VEHICLE_MAP under id_root={self.id_root!s})"
            )

        self._load_data(cache_dir)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data(self, cache_dir: Path | None) -> None:
        if self.use_id_split:
            from crl_vehicle.data.id_split import load_or_build_manifest
            self._id_manifest = load_or_build_manifest(
                id_root=self.id_root,
                mapping=DATASET_VEHICLE_MAP,
                window_sizes={
                    "audio":   self.cfg.modality_cfg("audio").window_size,
                    "seismic": self.cfg.modality_cfg("seismic").window_size,
                },
                cache_dir=self.id_cache_dir if self.id_cache_dir is not None
                           else self.id_root.parent / "id_cache",
            )
            # Scan all subdirs of id_root, dedupe by stem
            seen: dict[str, Path] = {}
            for p in sorted(self.id_root.glob("*/*.parquet")):
                seen.setdefault(p.stem, p)
            parquet_files = sorted(seen.values())
        else:
            self._id_manifest = None
            parquet_files = sorted(self.parquet_dir.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(
                f"No parquet files found "
                f"({'id_root=' + str(self.id_root) if self.use_id_split else 'parquet_dir=' + str(self.parquet_dir)})"
            )
        self._build_from_parquet(parquet_files)
        if self.use_id_split and self._id_skip_counts:
            summary = ", ".join(
                f"{k}={v}" for k, v in sorted(self._id_skip_counts.items())
            )
            logging.getLogger(__name__).info(
                f"id_split: role={self.role!r} skipped groups: {summary}"
            )
        self._preload_shared(cache_dir)

    def _build_from_parquet(self, files: list[Path]) -> None:
        # Group files by (dataset, vehicle, rs_node) and sensor
        audio_files:   dict[tuple, Path] = {}
        seismic_files: dict[tuple, Path] = {}

        for f in files:
            parsed = _parse_stem(f.stem, "audio")
            if parsed:
                ds, vehicle, rs = parsed
                audio_files[(ds, vehicle, rs)] = f
                continue
            parsed = _parse_stem(f.stem, "seismic")
            if parsed:
                ds, vehicle, rs = parsed
                seismic_files[(ds, vehicle, rs)] = f

        # All unique (dataset, vehicle, rs) keys
        all_keys = set(audio_files) | set(seismic_files)
        seg_id = 0

        for ds, vehicle, rs in sorted(all_keys):
            vtype, valid = _vehicle_to_labels(ds, vehicle)
            if not valid:
                continue

            det_label = 0 if vtype == LABEL_BACKGROUND else 1
            gkey = (ds, vehicle, rs, None)

            a_file = audio_files.get((ds, vehicle, rs))
            s_file = seismic_files.get((ds, vehicle, rs))

            audio_stem   = f"{ds}_audio_{vehicle}_{rs}"   if a_file else None
            seismic_stem = f"{ds}_seismic_{vehicle}_{rs}" if s_file else None

            audio_nw = seismic_nw = 0
            audio_seg_id = seismic_seg_id = seg_id
            audio_present_per_window: np.ndarray | None = None
            seismic_present_per_window: np.ndarray | None = None

            if a_file:
                # Window count comes from source rate (1 window = 1 second of source).
                src_sr_a = _source_sample_rate(audio_stem, "audio")
                audio_nw = pq.read_metadata(a_file).num_rows // src_sr_a
                self._cache["audio"][(audio_stem, None)] = {
                    "path": a_file, "n_windows": audio_nw
                }
                seg_id += 1
                audio_seg_id = seg_id
                audio_present_per_window = _read_parquet_present(a_file, src_sr_a)

            if s_file:
                src_sr_s = _source_sample_rate(seismic_stem, "seismic")
                seismic_nw = pq.read_metadata(s_file).num_rows // src_sr_s
                self._cache["seismic"][(seismic_stem, None)] = {
                    "path": s_file, "n_windows": seismic_nw
                }
                seg_id += 1
                seismic_seg_id = seg_id
                seismic_present_per_window = _read_parquet_present(s_file, src_sr_s)

            n_windows = min(audio_nw, seismic_nw) if audio_nw and seismic_nw else (audio_nw or seismic_nw)
            if n_windows == 0:
                continue

            # Build combined per-window presence using OR rule, sliced to n_windows
            if audio_present_per_window is not None and seismic_present_per_window is not None:
                combined_present = (
                    audio_present_per_window[:n_windows] | seismic_present_per_window[:n_windows]
                )
            elif audio_present_per_window is not None:
                combined_present = audio_present_per_window[:n_windows]
            elif seismic_present_per_window is not None:
                combined_present = seismic_present_per_window[:n_windows]
            else:
                combined_present = np.ones(n_windows, dtype=bool)

            self._groups[gkey] = {
                "audio_stem":    audio_stem,
                "seismic_stem":  seismic_stem,
                "seg_key":       None,
                "audio_nw":      audio_nw,
                "seismic_nw":    seismic_nw,
                "vehicle_type":  vtype,
                "audio_seg_id":  audio_seg_id,
                "seismic_seg_id": seismic_seg_id,
            }

            # ID-split routing: filter the window range based on marker / role
            if self.use_id_split:
                window_iter = self._id_split_window_indices(
                    ds=ds, vehicle=vehicle, rs=rs, n_windows=n_windows,
                )
                if window_iter is None:
                    continue  # group not present in this role
            else:
                window_iter = range(n_windows)

            for w in window_iter:
                if combined_present[w]:
                    w_vtype, w_det_label = vtype, det_label
                else:
                    w_vtype, w_det_label = LABEL_BACKGROUND, 0
                self._index.append((gkey, w, w_vtype, w_det_label, audio_seg_id, seismic_seg_id))

    def _id_split_window_indices(
        self, ds: str, vehicle: str, rs: str, n_windows: int,
    ) -> list[int] | None:
        """Return window indices this group contributes to self.role, or None to skip.

        - "train"/"val"/"test" markers: full range iff marker matches role.
        - "split"/"split_runs" markers: indices from manifest's
          split_assignments[role] for this group_key.
        - background / unknown: routed to "train".
        """
        ds_map = DATASET_VEHICLE_MAP.get(ds, {})
        entry = ds_map.get(vehicle)

        # Background or unknown → train only
        if entry is None or len(entry) < 3:
            if self.role == "train":
                return list(range(n_windows))
            self._id_skip_counts["no_marker_triple_role_mismatch"] = (
                self._id_skip_counts.get("no_marker_triple_role_mismatch", 0) + 1
            )
            return None

        marker = entry[2]
        if marker in ("train", "val", "test"):
            if marker == self.role:
                return list(range(n_windows))
            self._id_skip_counts["marker_role_mismatch"] = (
                self._id_skip_counts.get("marker_role_mismatch", 0) + 1
            )
            return None

        if marker in ("split", "split_runs"):
            gkey_str = f"{ds}__{vehicle}__{rs}"
            group = self._id_manifest.get("groups", {}).get(gkey_str)
            if group is None:
                self._id_skip_counts["manifest_missing_group"] = (
                    self._id_skip_counts.get("manifest_missing_group", 0) + 1
                )
                return None
            intervals = group.get("split_assignments", {}).get(self.role, [])
            indices: list[int] = []
            for start, end in intervals:
                indices.extend(range(int(start), min(int(end), n_windows)))
            if not indices:
                self._id_skip_counts["zero_windows_for_role"] = (
                    self._id_skip_counts.get("zero_windows_for_role", 0) + 1
                )
                return None
            return indices

        # Unknown marker — skip with a warning
        logging.getLogger(__name__).warning(
            f"id_split: unknown marker {marker!r} for {ds}/{vehicle} — skipping"
        )
        return None

    def _preload_shared(self, cache_dir: Path | None) -> None:
        """Load every file into a shared-memory tensor before workers are forked.

        Load order per file:
          1. If a valid .pt cache file exists (newer than source parquet), load it.
          2. Otherwise read the parquet, normalize, save .pt atomically, then use it.
          3. Call share_memory_() so all forked workers read the same physical pages.
        """
        if cache_dir is not None:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)

        for sensor in ("audio", "seismic"):
            mc = self.cfg.modality_cfg(sensor)
            W  = mc.window_size                # canonical (target) window size
            target_sr = mc.sample_rate         # canonical (target) sample rate
            for cache_key, entry in self._cache[sensor].items():
                stem    = cache_key[0]
                src     = entry["path"]
                source_sr = _source_sample_rate(stem, sensor)
                # Cache filename embeds target rate so old (pre-resample) caches
                # don't get mistakenly loaded after this schema change.
                pt_path = (
                    cache_dir / f"{stem}_sr{target_sr}.pt"
                ) if cache_dir else None

                data_dict: dict | None = None

                # Try loading from disk cache
                if pt_path is not None and pt_path.exists():
                    if pt_path.stat().st_mtime >= src.stat().st_mtime:
                        loaded = torch.load(pt_path, weights_only=True)
                        if isinstance(loaded, torch.Tensor):
                            loaded = None  # old bare-tensor format — force re-read
                        data_dict = loaded

                if data_dict is None:
                    arr = _read_parquet_numpy(src, W, source_sr, target_sr)
                    pres_arr = _read_parquet_present(src, source_sr)
                    if source_sr != target_sr:
                        logging.getLogger(__name__).info(
                            f"Loaded {stem}: resampled {source_sr}→{target_sr} Hz, "
                            f"{len(arr)} windows"
                        )
                    data_dict = {
                        "amplitude": torch.from_numpy(arr.copy()),
                        "present":   torch.from_numpy(pres_arr.copy()),
                    }
                    if pt_path is not None:
                        tmp = pt_path.with_suffix(".tmp")
                        torch.save(data_dict, tmp)
                        tmp.rename(pt_path)  # atomic on POSIX

                data_dict["amplitude"].share_memory_()
                data_dict["present"].share_memory_()
                self._data_cache[sensor][cache_key] = data_dict

    # ------------------------------------------------------------------
    # Window loading
    # ------------------------------------------------------------------

    def _get_window(
        self, sensor: str, stem: str, seg_key: Any, w: int,
    ) -> torch.Tensor:
        """Return a clean (1, W) window. Interventions are applied later on the
        GPU side in the training step — see apply_intervention_batch."""
        mc = self.cfg.modality_cfg(sensor)
        cache_key = (stem, seg_key)
        data_dict = self._data_cache[sensor].get(cache_key)

        if data_dict is None or w >= len(data_dict["amplitude"]):
            return torch.zeros(1, mc.window_size)

        x = data_dict["amplitude"][w].unsqueeze(0).clone()  # clone: don't write into shared pages
        x = remove_dc(x)
        return x

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        gkey, w, vtype, det_label, audio_seg_id, seismic_seg_id = self._index[idx]
        g = self._groups[gkey]

        audio_avail   = g["audio_stem"]   is not None
        seismic_avail = g["seismic_stem"] is not None

        x_audio = (self._get_window("audio", g["audio_stem"], g["seg_key"], w)
                   if audio_avail else torch.zeros(1, self.cfg.modality_cfg("audio").window_size))
        x_seismic = (self._get_window("seismic", g["seismic_stem"], g["seg_key"], w)
                     if seismic_avail else torch.zeros(1, self.cfg.modality_cfg("seismic").window_size))

        return {
            "x_audio":         x_audio,
            "x_seismic":       x_seismic,
            "audio_avail":     audio_avail,
            "seismic_avail":   seismic_avail,
            "vehicle_type":    vtype,
            "detection_label": det_label,
            "segment_id":      audio_seg_id,
        }


# ---------------------------------------------------------------------------
# StratifiedPairDataset
# ---------------------------------------------------------------------------

class StratifiedPairDataset(Dataset):
    """Returns anchor + stratified partners for CRL pre-training.

    Partners (indexed p0, p1, ...):
      p0:           consecutive window (STRATUM_CONSEC)
      p1:           same dataset, same vehicle type (STRATUM_SAME_TYPE)
      p2:           same dataset, different vehicle type (STRATUM_DIFF_TYPE)
      p3:           different dataset (STRATUM_CROSS_DS)

    __getitem__ returns a dict with keys:
      x_audio_t, x_seismic_t, detection_label_t, vehicle_type_t,
      audio_avail, seismic_avail,
      x_audio_p{p}, x_seismic_p{p}, detection_label_p{p}, vehicle_type_p{p},
      partner_stratum_p{p}   for p in 0..n_partners-1

    Memory layout: shared lookup dicts are built once and sampled at __getitem__
    time. Per-anchor storage is O(1) — only the consecutive successor index.
    """

    def __init__(self, sensor_dataset: SensorDataset) -> None:
        self.ds = sensor_dataset
        cfg = sensor_dataset.cfg

        self._n_same  = cfg.n_partners_same_type
        self._n_diff  = cfg.n_partners_diff_type
        self._n_cross = cfg.n_partners_cross_ds

        self._build_index()

    def _build_index(self) -> None:
        index = self.ds._index
        groups = self.ds._groups

        self._anchors: list[int] = []
        # Per-anchor: only the single consecutive successor (O(1) per anchor)
        self._consec_next: dict[int, int] = {}

        # Shared pools — built once, O(1) lookup at sample time
        # (ds_name, vtype) → indices with that (ds, type)
        self._ds_vtype_idx: dict[tuple, list[int]] = {}
        # ds_name → indices from that dataset
        self._ds_idx: dict[str, list[int]] = {}
        # cross-ds: derived on demand from _ds_idx[other_ds] — no separate storage

        gkey_to_sorted: dict[tuple, list[int]] = {}
        for i, (gkey, w, vtype, det, a_seg, s_seg) in enumerate(index):
            gkey_to_sorted.setdefault(gkey, []).append(i)
            ds = gkey[0]
            self._ds_vtype_idx.setdefault((ds, vtype), []).append(i)
            self._ds_idx.setdefault(ds, []).append(i)
        for gkey in gkey_to_sorted:
            gkey_to_sorted[gkey].sort(key=lambda i: index[i][1])

        for gkey, sorted_list in gkey_to_sorted.items():
            g = groups[gkey]
            if not (g["audio_stem"] is not None and g["seismic_stem"] is not None):
                continue
            for pos, global_idx in enumerate(sorted_list[:-1]):
                self._anchors.append(global_idx)
                self._consec_next[global_idx] = sorted_list[pos + 1]

    def _sample_partner(self, anchor_idx: int, stratum: int) -> int:
        """Sample one partner index for the given stratum, falling back to consec."""
        index  = self.ds._index
        gkey, _, vtype, _, _, _ = index[anchor_idx]
        ds_name = gkey[0]
        fallback = self._consec_next[anchor_idx]

        if stratum == STRATUM_CONSEC:
            return fallback

        if stratum == STRATUM_SAME_TYPE:
            # rejection-sample from shared pool to exclude same gkey
            candidates = self._ds_vtype_idx.get((ds_name, vtype), [])
            pool = [j for j in candidates if index[j][0] != gkey]
        elif stratum == STRATUM_DIFF_TYPE:
            candidates = self._ds_idx.get(ds_name, [])
            pool = [j for j in candidates
                    if index[j][2] != vtype and index[j][2] >= 0
                    and index[j][0] != gkey]
        else:  # STRATUM_CROSS_DS — pick a random other dataset, then sample from it
            other_ds_names = [d for d in self._ds_idx if d != ds_name]
            if other_ds_names:
                other_ds = random.choice(other_ds_names)
                pool = self._ds_idx[other_ds]
            else:
                pool = []

        return random.choice(pool) if pool else fallback

    def _fetch(self, idx: int) -> dict:
        """Fetch a single window item from the underlying SensorDataset."""
        gkey, w, vtype, det, a_seg, s_seg = self.ds._index[idx]
        g = self.ds._groups[gkey]
        audio_avail   = g["audio_stem"]   is not None
        seismic_avail = g["seismic_stem"] is not None
        mc_a = self.ds.cfg.modality_cfg("audio")
        mc_s = self.ds.cfg.modality_cfg("seismic")
        x_audio = (self.ds._get_window("audio", g["audio_stem"], g["seg_key"], w)
                   if audio_avail else torch.zeros(1, mc_a.window_size))
        x_seismic = (self.ds._get_window("seismic", g["seismic_stem"], g["seg_key"], w)
                     if seismic_avail else torch.zeros(1, mc_s.window_size))
        return {
            "x_audio": x_audio, "x_seismic": x_seismic,
            "vehicle_type": vtype, "detection_label": det,
            "audio_avail": audio_avail, "seismic_avail": seismic_avail,
        }

    def __len__(self) -> int:
        return len(self._anchors)

    def __getitem__(self, idx: int) -> dict:
        anchor_idx = self._anchors[idx]
        anchor = self._fetch(anchor_idx)

        item: dict = {
            "x_audio_t":         anchor["x_audio"],
            "x_seismic_t":       anchor["x_seismic"],
            "detection_label_t": anchor["detection_label"],
            "vehicle_type_t":    anchor["vehicle_type"],
            "audio_avail":       anchor["audio_avail"],
            "seismic_avail":     anchor["seismic_avail"],
        }

        strata = (
            [STRATUM_CONSEC]
            + [STRATUM_SAME_TYPE] * self._n_same
            + [STRATUM_DIFF_TYPE] * self._n_diff
            + [STRATUM_CROSS_DS]  * self._n_cross
        )
        for p, stratum in enumerate(strata):
            pidx = self._sample_partner(anchor_idx, stratum)
            pw = self._fetch(pidx)
            item[f"x_audio_p{p}"]         = pw["x_audio"]
            item[f"x_seismic_p{p}"]       = pw["x_seismic"]
            item[f"detection_label_p{p}"] = pw["detection_label"]
            item[f"vehicle_type_p{p}"]    = pw["vehicle_type"]
            item[f"partner_stratum_p{p}"] = stratum

        return item


# ---------------------------------------------------------------------------
# Collate functions
# ---------------------------------------------------------------------------

def collate_single(batch: list[dict]) -> dict:
    """Standard collate for SensorDataset."""
    out: dict = {}
    for key in batch[0]:
        vals = [b[key] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            out[key] = torch.stack(vals)
        elif isinstance(vals[0], bool):
            out[key] = torch.tensor(vals, dtype=torch.bool)
        else:
            out[key] = torch.tensor(vals)
    return out


def compute_class_weights(ds: SensorDataset) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute class weights from the dataset index without iterating the dataloader.

    Returns:
        pres_pos_weight: scalar tensor for BCEWithLogitsLoss pos_weight (n_neg / n_pos)
        type_weights:    (4,) tensor for CrossEntropyLoss weight (inverse freq, normalised)
    """
    from crl_vehicle.config import CATEGORY_TO_IDX
    n_classes = len(CATEGORY_TO_IDX)

    n_pos = n_neg = 0
    type_counts = [0] * n_classes

    for _, _, vtype, det_label, _, _ in ds._index:
        if det_label == 1:
            n_pos += 1
        else:
            n_neg += 1
        if 0 <= vtype < n_classes:
            type_counts[vtype] += 1

    pres_pos_weight = torch.tensor(n_neg / max(n_pos, 1), dtype=torch.float32)

    total_typed = sum(type_counts)
    if total_typed == 0:
        type_weights = torch.ones(n_classes, dtype=torch.float32)
    else:
        inv_freq = [total_typed / max(c, 1) for c in type_counts]
        s = sum(inv_freq)
        type_weights = torch.tensor([v / s * n_classes for v in inv_freq], dtype=torch.float32)

    return pres_pos_weight, type_weights


def collate_pairs(batch: list[dict]) -> dict:
    """Collate for StratifiedPairDataset.

    Anchor keys (x_*_t, *_t, *_avail) collate normally. Per-partner keys
    (x_*_p{i}, detection_label_p{i}, vehicle_type_p{i}, partner_stratum_p{i})
    are STACKED across the partner axis into single tensors per modality, so
    the worker→main IPC handoff is O(modalities) rather than O(modalities*P).

    This was the dominant bottleneck on 2026-04-26: ~26 fd-handoffs per sample
    × B=128 saturated PyTorch's resource_sharer. Stacking partner-side tensors
    cuts the fd count to a small constant.

    New schema (B = batch, P = partners):
      x_audio_partners        (B, P, 1, W_a)
      x_seismic_partners      (B, P, 1, W_s)
      detection_label_partners (B, P)
      vehicle_type_partners    (B, P)
      partner_stratum_partners (B, P)
    """
    out: dict = {}
    sample = batch[0]

    n_partners = sum(1 for k in sample if k.startswith("x_audio_p"))
    out["n_partners"] = n_partners

    # Anchor and non-partner scalar keys collate normally.
    partner_keys = {
        "x_audio_p", "x_seismic_p",
        "detection_label_p", "vehicle_type_p", "partner_stratum_p",
    }
    def _is_partner_key(k: str) -> bool:
        return any(k.startswith(pk) for pk in partner_keys)

    for key in sample:
        if _is_partner_key(key):
            continue
        vals = [b[key] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            out[key] = torch.stack(vals)
        elif isinstance(vals[0], bool):
            out[key] = torch.tensor(vals, dtype=torch.bool)
        else:
            out[key] = torch.tensor(vals)

    if n_partners == 0:
        return out

    # Stack partners: each output is (B, P, ...) — one fd per modality, not per partner.
    def _stack_partner(prefix: str) -> torch.Tensor:
        # Inner: stack P partners per sample → (P, ...). Outer: stack B samples → (B, P, ...).
        per_sample = [
            torch.stack([b[f"{prefix}{p}"] for p in range(n_partners)])
            for b in batch
        ]
        return torch.stack(per_sample)

    def _stack_partner_scalars(prefix: str, dtype) -> torch.Tensor:
        return torch.tensor(
            [[b[f"{prefix}{p}"] for p in range(n_partners)] for b in batch],
            dtype=dtype,
        )

    out["x_audio_partners"]   = _stack_partner("x_audio_p")
    out["x_seismic_partners"] = _stack_partner("x_seismic_p")
    out["detection_label_partners"] = _stack_partner_scalars("detection_label_p", torch.long)
    out["vehicle_type_partners"]    = _stack_partner_scalars("vehicle_type_p", torch.long)
    out["partner_stratum_partners"] = _stack_partner_scalars("partner_stratum_p", torch.long)

    return out
