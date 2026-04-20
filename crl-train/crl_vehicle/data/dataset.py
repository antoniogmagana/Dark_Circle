"""SensorDataset and StratifiedPairDataset for CRL training."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from crl_vehicle.config import (
    CRLConfig, LABEL_BACKGROUND, LABEL_MULTI, CATEGORY_TO_IDX
)
from crl_vehicle.data.transforms import remove_dc, apply_intervention, N_INTERVENTIONS

# Stratum identifiers for partner sampling
STRATUM_CONSEC    = 0
STRATUM_SAME_TYPE = 1
STRATUM_DIFF_TYPE = 2
STRATUM_CROSS_DS  = 3

# Known dataset names
_KNOWN_DATASETS = {"iobt", "focal", "m3nvc"}

# Vehicle name → (vehicle_type_idx, is_valid)
# vehicle_type_idx follows CATEGORY_TO_IDX; background/multi use LABEL_* constants
_VEHICLE_REGISTRY: dict[str, tuple[int, bool]] = {}


def _build_vehicle_registry() -> dict[str, tuple[int, bool]]:
    """Build a lookup from vehicle stem → (label_idx, is_valid).
    Vehicles not in this registry are invalid and excluded from training."""
    reg: dict[str, tuple[int, bool]] = {}
    # iobt vehicles
    for name in ["polaris0150pm", "polaris0255pm", "polaris0075pm"]:
        reg[name] = (CATEGORY_TO_IDX["light"], True)
    for name in ["silverado0255pm", "silverado0150pm", "silverado0075pm",
                 "ram0255pm", "ram0150pm", "ram0075pm",
                 "f2500255pm", "f250150pm", "f250075pm"]:
        reg[name] = (CATEGORY_TO_IDX["utility"], True)
    for name in ["camaro0255pm", "camaro0150pm", "camaro0075pm",
                 "mustang0255pm", "mustang0150pm", "mustang0075pm"]:
        reg[name] = (CATEGORY_TO_IDX["sport"], True)
    for name in ["pedestrian0255pm", "pedestrian0150pm", "pedestrian0075pm",
                 "pedestrian"]:
        reg[name] = (CATEGORY_TO_IDX["pedestrian"], True)
    # focal vehicles
    for name in ["motor", "sedan", "suv", "pickup"]:
        reg[name] = (CATEGORY_TO_IDX["light"], True)
    # m3nvc: background and multi-vehicle
    reg["background"] = (LABEL_BACKGROUND, True)
    return reg


_VEHICLE_REGISTRY = _build_vehicle_registry()


def _vehicle_to_labels(dataset: str, vehicle: str) -> tuple[int, bool]:
    """Return (vehicle_type_idx, is_valid) for a (dataset, vehicle) pair."""
    if dataset == "m3nvc":
        if "_" in vehicle:  # multi-vehicle: e.g., cx30_miata
            return LABEL_MULTI, True
        return LABEL_BACKGROUND, True
    if vehicle in _VEHICLE_REGISTRY:
        return _VEHICLE_REGISTRY[vehicle]
    return -99, False


def _read_parquet_numpy(path: Path) -> np.ndarray:
    """Read a parquet file → float32 numpy array (N_windows, W) using pyarrow threads."""
    import pyarrow as pa
    table = pq.read_table(path, use_threads=True)
    numeric_cols = [
        name for name, typ in zip(table.schema.names, table.schema.types)
        if pa.types.is_integer(typ) or pa.types.is_floating(typ)
    ]
    if numeric_cols:
        table = table.select(numeric_cols)
    return table.to_pandas().values.astype(np.float32)


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
    ) -> None:
        self.parquet_dir = Path(parquet_dir)
        self.cfg = config
        self.is_train = is_train
        # metadata store: (stem, seg_key) → {path, n_windows}
        self._cache: dict[str, dict] = {"audio": {}, "seismic": {}}
        # Shared-memory tensor store: (stem, seg_key) → Tensor(N, W) in shared memory.
        # Populated once at init, shared across all DataLoader workers with zero duplication.
        self._data_cache: dict[str, dict] = {"audio": {}, "seismic": {}}
        self._index: list = []   # [(gkey, w_idx, vtype, det_label, audio_seg_id, seismic_seg_id)]
        self._groups: dict = {}  # gkey → group metadata

        self._load_data(cache_dir)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data(self, cache_dir: Path | None) -> None:
        parquet_files = sorted(self.parquet_dir.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files in {self.parquet_dir}")
        self._build_from_parquet(parquet_files)
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

            if a_file:
                audio_nw = pq.read_metadata(a_file).num_rows
                self._cache["audio"][(audio_stem, None)] = {
                    "path": a_file, "n_windows": audio_nw
                }
                seg_id += 1
                audio_seg_id = seg_id

            if s_file:
                seismic_nw = pq.read_metadata(s_file).num_rows
                self._cache["seismic"][(seismic_stem, None)] = {
                    "path": s_file, "n_windows": seismic_nw
                }
                seg_id += 1
                seismic_seg_id = seg_id

            n_windows = min(audio_nw, seismic_nw) if audio_nw and seismic_nw else (audio_nw or seismic_nw)
            if n_windows == 0:
                continue

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

            for w in range(n_windows):
                self._index.append((gkey, w, vtype, det_label, audio_seg_id, seismic_seg_id))

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
            W  = mc.window_size
            for cache_key, entry in self._cache[sensor].items():
                stem    = cache_key[0]
                src     = entry["path"]
                pt_path = (cache_dir / f"{stem}.pt") if cache_dir else None

                data: torch.Tensor | None = None

                # Try loading from disk cache
                if pt_path is not None and pt_path.exists():
                    if pt_path.stat().st_mtime >= src.stat().st_mtime:
                        data = torch.load(pt_path, weights_only=True)

                if data is None:
                    arr = _read_parquet_numpy(src)
                    if arr.shape[1] > W:
                        arr = arr[:, :W]
                    elif arr.shape[1] < W:
                        arr = np.pad(arr, ((0, 0), (0, W - arr.shape[1])))
                    data = torch.from_numpy(arr.copy())  # owns its memory before sharing
                    if pt_path is not None:
                        tmp = pt_path.with_suffix(".tmp")
                        torch.save(data, tmp)
                        tmp.rename(pt_path)  # atomic on POSIX

                data.share_memory_()
                self._data_cache[sensor][cache_key] = data

    # ------------------------------------------------------------------
    # Window loading
    # ------------------------------------------------------------------

    def _get_window(
        self, sensor: str, stem: str, seg_key: Any, w: int, interv_idx: int
    ) -> torch.Tensor:
        mc = self.cfg.modality_cfg(sensor)
        cache_key = (stem, seg_key)
        data = self._data_cache[sensor].get(cache_key)

        if data is None or w >= len(data):
            return torch.zeros(1, mc.window_size)

        x = data[w].unsqueeze(0).clone()  # clone: don't write into shared pages
        x = remove_dc(x)
        if interv_idx > 0:
            x = apply_intervention(x, interv_idx, mc.sample_rate)
        return x

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        gkey, w, vtype, det_label, audio_seg_id, seismic_seg_id = self._index[idx]
        g = self._groups[gkey]

        interv_idx = random.randint(0, N_INTERVENTIONS) if self.is_train else 0

        audio_avail   = g["audio_stem"]   is not None
        seismic_avail = g["seismic_stem"] is not None

        x_audio = (self._get_window("audio", g["audio_stem"], g["seg_key"], w, interv_idx)
                   if audio_avail else torch.zeros(1, self.cfg.modality_cfg("audio").window_size))
        x_seismic = (self._get_window("seismic", g["seismic_stem"], g["seg_key"], w, interv_idx)
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
        interv_idx = 0  # no intervention on partners
        audio_avail   = g["audio_stem"]   is not None
        seismic_avail = g["seismic_stem"] is not None
        mc_a = self.ds.cfg.modality_cfg("audio")
        mc_s = self.ds.cfg.modality_cfg("seismic")
        x_audio = (self.ds._get_window("audio", g["audio_stem"], g["seg_key"], w, interv_idx)
                   if audio_avail else torch.zeros(1, mc_a.window_size))
        x_seismic = (self.ds._get_window("seismic", g["seismic_stem"], g["seg_key"], w, interv_idx)
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


def collate_pairs(batch: list[dict]) -> dict:
    """Collate for StratifiedPairDataset. Discovers partner slots dynamically."""
    out: dict = {}
    sample = batch[0]

    # Count partners
    n_partners = sum(1 for k in sample if k.startswith("x_audio_p"))
    out["n_partners"] = n_partners

    # Collate all keys
    for key in sample:
        vals = [b[key] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            out[key] = torch.stack(vals)
        elif isinstance(vals[0], bool):
            out[key] = torch.tensor(vals, dtype=torch.bool)
        else:
            out[key] = torch.tensor(vals)

    return out
