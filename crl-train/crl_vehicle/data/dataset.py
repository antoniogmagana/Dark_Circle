"""
SensorDataset and StratifiedPairDataset

Reads seismic AND audio parquet files, returning them independently so that
each modality can be processed through its own frontend → TemporalEncoder
chain with no early signal mixing (no entanglement before the latent space).

Parquet filename convention:
    {dataset}_{sensor}_{vehicle}_{rs}.parquet

Supported sensors: "audio", "seismic"   (accel is ignored for now)

SensorDataset.__getitem__ returns:
    {
      'x_audio':         tensor (1, W_audio)  or zeros if unavailable
      'x_seismic':       tensor (1, W_seismic) or zeros if unavailable
      'audio_avail':     bool
      'seismic_avail':   bool
      'interv_idx':      int    — 0=none, 1-7=noise type (same label for both)
      'vehicle_type':    int    — LABEL_BACKGROUND/LABEL_MULTI or 0-3
      'detection_label': int    — 0=absent, 1=present
      'segment_id':      int    — unique int per continuous recording segment
    }


Windows overlap: stride = cfg.horizon_stride_sec seconds (default 0.7 s).
Raw waveforms are cached once so there is no additional memory cost.

StratifiedPairDataset wraps SensorDataset for CITRIS CRL training.
Each anchor yields K+1 partners drawn at __init__ time into pre-built pools:
    - 1 consecutive (w+1): same recording, same vehicle, same environment
    - n_partners_same_type: same dataset, same vehicle_type, different recording
    - n_partners_diff_type: same dataset, different vehicle_type
    - n_partners_cross_ds:  any other dataset

Pool indices are stored as plain int lists (8 bytes × N_anchors × K total),
so __getitem__ is O(1): one randint pick per stratum, no filtering at runtime.

Batch keys (per partner slot p=0..K-1, plus anchor at 't'):
    'x_audio_t', 'x_seismic_t'           — anchor window
    'x_audio_p{p}', 'x_seismic_p{p}'     — partner p
    'audio_avail', 'seismic_avail'        — both True (anchor must have both)
    'detection_label_t', 'vehicle_type_t' — anchor labels
    'detection_label_p{p}', 'vehicle_type_p{p}' — partner labels
    'partner_stratum_p{p}'                — 0=consec,1=same_type,2=diff_type,3=cross
"""

import re
import hashlib
import pickle
import numpy as np
import pandas as pd
import torch
import torchaudio
from pathlib import Path
from torch.utils.data import Dataset

from crl_vehicle.config import (
    CRLConfig,
    NATIVE_SR,
    ADC_SCALE,
    DATASET_VEHICLE_MAP,
    CATEGORY_TO_IDX,
    LABEL_BACKGROUND,
    LABEL_MULTI,
    MODALITIES,
)
from crl_vehicle.data.transforms import (
    rms_normalize,
    apply_intervention,
    N_INTERVENTIONS,
)

# Target sample rates: each modality has its own.
# Audio at 16 kHz preserves engine harmonics (2–8 kHz); seismic at 200 Hz.
_TARGET_SR = {
    "audio":   16000,
    "seismic": 200,
}

_RS_RE = re.compile(r"_rs\d+$")


# ---------------------------------------------------------------------------
# Disk-cache key
# ---------------------------------------------------------------------------

def _compute_dir_hash(parquet_dir: Path) -> str:
    """
    Stable hash of parquet directory contents: sorted (filename, mtime, size) tuples.
    Invalidates when files are added, removed, or modified.
    """
    entries = sorted(
        (p.name, p.stat().st_mtime_ns, p.stat().st_size)
        for p in parquet_dir.glob("*.parquet")
    )
    return hashlib.sha256(str(entries).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

def _parse_stem(stem: str, sensor: str):
    """
    Parse a parquet stem for a specific sensor.
    Returns (dataset, vehicle, rs_node) or None.
    """
    parts = stem.split("_", 2)
    if len(parts) < 3:
        return None
    dataset, file_sensor, rest = parts
    if dataset not in NATIVE_SR:
        return None
    if file_sensor != sensor:
        return None
    rs_match = _RS_RE.search(rest)
    if not rs_match:
        return None
    rs_node = rest[rs_match.start() + 1:]
    vehicle = rest[:rs_match.start()]
    return dataset, vehicle, rs_node


def _vehicle_to_labels(dataset: str, vehicle: str):
    """Map (dataset, vehicle) → (vehicle_type_int, is_valid)."""
    ds_map = DATASET_VEHICLE_MAP.get(dataset, {})
    entry = ds_map.get(vehicle)
    if entry is None:
        for key in sorted(ds_map.keys(), key=len, reverse=True):
            if vehicle.startswith(key):
                entry = ds_map[key]
                break
    if entry is None:
        return None, False
    type_str = entry[0]
    if type_str == "multi":
        return LABEL_MULTI, True
    if type_str == "background":
        return LABEL_BACKGROUND, True
    return CATEGORY_TO_IDX[type_str], True


def _sample_interv(is_train: bool) -> int:
    """Return a random intervention index (1..N_INTERVENTIONS) with 60% probability during training."""
    if is_train and torch.rand(1).item() < 0.60:
        return torch.randint(1, N_INTERVENTIONS + 1, (1,)).item()
    return 0


# ---------------------------------------------------------------------------
# SensorDataset
# ---------------------------------------------------------------------------

class SensorDataset(Dataset):
    """
    Args:
        parquet_dir : path to a split directory (data/parsed/train, val, etc.)
        config      : CRLConfig instance
        is_train    : enables random noise interventions during __getitem__
    """

    def __init__(
        self, parquet_dir: str, config: CRLConfig, is_train: bool = True
    ):
        self.parquet_dir = Path(parquet_dir)
        self.cfg = config
        self.is_train = is_train

        # RAM cache per modality:
        # (stem, seg_key) → {"data": ndarray [1,T], "present": bool[], "native_sr": int}
        self._cache: dict[str, dict] = {m: {} for m in MODALITIES}
        self._resamplers: dict = {}

        # Segment registry: (modality, stem, seg_key) → int segment_id
        self._segment_id_map: dict = {}
        self._seg_counter = 0

        # Primary index keyed by (dataset, vehicle, rs_node, seg_key)
        # Each group entry: {"segs": {modality: (stem, n_windows)}, labels...}
        self._groups: dict = {}

        # Final flat index:
        # (gkey, w, vehicle_type, det_label, audio_seg_id, seismic_seg_id)
        self._index: list = []

        self._build_index()
        print(
            f"  SensorDataset [{self.parquet_dir.name}]: "
            f"{len(self._index)} windows, "
            f"{sum(1 for e in self._index if e[4] >= 0)} with audio, "
            f"{sum(1 for e in self._index if e[5] >= 0)} with seismic"
        )

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _build_index(self):
        files = sorted(self.parquet_dir.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files in {self.parquet_dir}")

        # Pass 1: group files by (dataset, vehicle, rs_node) per modality
        raw_groups: dict = {}
        for path in files:
            for sensor in MODALITIES:
                parsed = _parse_stem(path.stem, sensor)
                if parsed is None:
                    continue
                dataset, vehicle, rs_node = parsed
                gkey = (dataset, vehicle, rs_node)
                raw_groups.setdefault(gkey, {})[sensor] = path

        # Pass 2: load, resolve labels, build window index
        for (dataset, vehicle, rs_node), sensor_paths in raw_groups.items():
            vehicle_type, valid = _vehicle_to_labels(dataset, vehicle)
            if not valid:
                continue

            # Load each available modality
            mod_segs: dict[str, dict] = {}  # modality → {seg_key: (stem, n_windows)}
            for sensor, path in sensor_paths.items():
                segs = self._load_file(path, path.stem, sensor, dataset)
                if segs:
                    mod_segs[sensor] = segs

            if not mod_segs:
                continue

            # Collect all seg_keys across all loaded modalities
            all_seg_keys: set = set()
            for segs in mod_segs.values():
                all_seg_keys.update(segs.keys())

            for seg_key in sorted(all_seg_keys, key=lambda x: (x is None, x)):
                # Determine window count for each available modality
                audio_stem, audio_nw, audio_seg_id = None, 0, -1
                seismic_stem, seismic_nw, seismic_seg_id = None, 0, -1

                for sensor in MODALITIES:
                    segs = mod_segs.get(sensor, {})
                    if seg_key in segs:
                        stem, nw = segs[seg_key]
                        sid = self._get_segment_id(sensor, stem, seg_key)
                        if sensor == "audio":
                            audio_stem, audio_nw, audio_seg_id = stem, nw, sid
                        else:
                            seismic_stem = stem
                            seismic_nw = nw
                            seismic_seg_id = sid

                # Need at least one modality with at least 1 window
                max_windows = max(audio_nw, seismic_nw)
                if max_windows < 1:
                    continue

                gkey = (dataset, vehicle, rs_node, seg_key)
                self._groups[gkey] = {
                    "audio_stem":    audio_stem,
                    "seismic_stem":  seismic_stem,
                    "seg_key":       seg_key,
                    "audio_nw":      audio_nw,
                    "seismic_nw":    seismic_nw,
                    "vehicle_type":  vehicle_type,
                    "audio_seg_id":  audio_seg_id,
                    "seismic_seg_id": seismic_seg_id,
                }

                # Determine presence label from whichever modality is available.
                # Detection label is sampled at the native-SR stride position.
                ref_sensor = "seismic" if seismic_stem else "audio"
                ref_stem = seismic_stem if seismic_stem else audio_stem
                ref_cache = self._cache[ref_sensor]
                ref_entry = ref_cache.get((ref_stem, seg_key), {})
                native_sr = ref_entry.get("native_sr", 1)
                stride_native = int(native_sr * self.cfg.horizon_stride_sec)
                stride_native = max(stride_native, 1)
                present_arr = ref_entry.get(
                    "present", np.ones(1, dtype=bool)
                )

                for w in range(max_windows):
                    row_idx = w * stride_native
                    det_label = int(present_arr[row_idx]) if row_idx < len(present_arr) else 1
                    self._index.append((gkey, w, vehicle_type, det_label, audio_seg_id, seismic_seg_id))

    def _get_segment_id(self, sensor: str, stem: str, seg_key) -> int:
        key = (sensor, stem, seg_key)
        if key not in self._segment_id_map:
            self._segment_id_map[key] = self._seg_counter
            self._seg_counter += 1
        return self._segment_id_map[key]

    def _resample_entry(self, entry: dict, sensor: str) -> dict:
        """
        Resample entry["data"] from native_sr to target SR in-place (replaces array).
        present array stays at native_sr length — it is only indexed by w*stride_native
        which is still valid because stride_native/native_sr == stride_target/target_sr.
        """
        target_sr = _TARGET_SR[sensor]
        native_sr = entry["native_sr"]
        if native_sr == target_sr:
            entry["target_sr"] = target_sr
            return entry
        key = (native_sr, target_sr)
        if key not in self._resamplers:
            self._resamplers[key] = torchaudio.transforms.Resample(
                orig_freq=native_sr, new_freq=target_sr
            )
        tensor = torch.from_numpy(entry["data"])          # (1, T_native)
        resampled = self._resamplers[key](tensor)         # (1, T_target)
        entry["data"] = resampled.numpy()
        entry["target_sr"] = target_sr
        return entry

    def _load_file(self, path: Path, stem: str, sensor: str, dataset: str) -> dict:
        """Load parquet, split m3nvc segments. Returns {seg_key: (stem, n_windows)}."""
        try:
            df = pd.read_parquet(path)
        except Exception as e:
            print(f"  Warning: could not read {path}: {e}")
            return {}

        native_sr = NATIVE_SR.get(dataset, {}).get(sensor)
        if native_sr is None:
            return {}

        target_sr = _TARGET_SR[sensor]
        stride_target = max(int(target_sr * self.cfg.horizon_stride_sec), 1)

        result = {}
        has_segments = "scene_id" in df.columns and "run_id" in df.columns

        if has_segments:
            for (scene_id, run_id), seg_df in df.groupby(["scene_id", "run_id"], sort=True):
                seg_key = (int(scene_id), int(run_id))
                cache_key = (stem, seg_key)
                if cache_key not in self._cache[sensor]:
                    entry = self._df_to_entry(seg_df, sensor, native_sr)
                    if entry is None:
                        continue
                    entry = self._resample_entry(entry, sensor)
                    self._cache[sensor][cache_key] = entry
                nw = self._n_windows(self._cache[sensor][cache_key], stride_target)
                if nw >= 1:
                    result[seg_key] = (stem, nw)
        else:
            cache_key = (stem, None)
            if cache_key not in self._cache[sensor]:
                entry = self._df_to_entry(df, sensor, native_sr)
                if entry is None:
                    return {}
                entry = self._resample_entry(entry, sensor)
                self._cache[sensor][cache_key] = entry
            nw = self._n_windows(self._cache[sensor][cache_key], stride_target)
            if nw >= 1:
                result[None] = (stem, nw)

        return result

    def _n_windows(self, entry: dict, stride: int) -> int:
        target_sr = entry["target_sr"]
        win_len = int(target_sr * self.cfg.sample_seconds)
        total = entry["data"].shape[-1]
        if total < win_len:
            return 0
        return (total - win_len) // stride + 1

    @staticmethod
    def _df_to_entry(df: pd.DataFrame, sensor: str, native_sr: int) -> dict | None:
        if sensor == "audio":
            if "amplitude" not in df.columns:
                return None
            arr = df["amplitude"].to_numpy(dtype=np.float32)[np.newaxis, :]
        elif sensor == "seismic":
            if "amplitude" not in df.columns:
                return None
            arr = df["amplitude"].to_numpy(dtype=np.float32)[np.newaxis, :]
        else:
            return None

        arr = arr / ADC_SCALE[sensor]
        present = (
            df["present"].to_numpy(dtype=bool)
            if "present" in df.columns
            else np.ones(arr.shape[-1], dtype=bool)
        )
        return {"data": arr, "present": present, "native_sr": native_sr}

    # ------------------------------------------------------------------
    # Window extraction (per modality, independently)
    # ------------------------------------------------------------------

    def _get_window(
        self, sensor: str, stem: str, seg_key, w: int, interv_idx: int
    ) -> torch.Tensor:
        entry = self._cache[sensor][(stem, seg_key)]
        target_sr = entry["target_sr"]
        win_len = int(target_sr * self.cfg.sample_seconds)
        stride = max(int(target_sr * self.cfg.horizon_stride_sec), 1)
        start = w * stride
        chunk = entry["data"][:, start: start + win_len].copy()
        tensor = torch.from_numpy(chunk)                              # [1, win_len]
        if self.is_train and interv_idx > 0:
            tensor = apply_intervention(tensor, interv_idx, target_sr)
        return rms_normalize(tensor)

    def _zero_window(self, sensor: str) -> torch.Tensor:
        mod_cfg = self.cfg.modality_cfg(sensor)
        return torch.zeros(mod_cfg.n_channels, mod_cfg.window_size)

    # ------------------------------------------------------------------
    # __len__ / __getitem__
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        gkey, w, vehicle_type, det_label, audio_seg_id, seismic_seg_id = self._index[idx]
        group = self._groups[gkey]

        # Same intervention applied to both modalities — same environmental cause,
        # each sensor responds according to its own physics.
        interv_idx = _sample_interv(self.is_train)

        # Extract each modality independently; zero-pad if unavailable
        audio_avail = group["audio_stem"] is not None and w < group["audio_nw"]
        seismic_avail = group["seismic_stem"] is not None and w < group["seismic_nw"]

        if audio_avail:
            x_audio = self._get_window("audio", group["audio_stem"], group["seg_key"], w, interv_idx)
        else:
            x_audio = self._zero_window("audio")

        if seismic_avail:
            x_seismic = self._get_window("seismic", group["seismic_stem"], group["seg_key"], w, interv_idx)
        else:
            x_seismic = self._zero_window("seismic")

        # Use modality-specific segment IDs so the SSM/encoder for each
        # modality has consistent segment context within its own chain.
        segment_id = seismic_seg_id if seismic_avail else audio_seg_id

        return {
            "x_audio":         x_audio,
            "x_seismic":       x_seismic,
            "audio_avail":     audio_avail,
            "seismic_avail":   seismic_avail,
            "interv_idx":      interv_idx,
            "vehicle_type":    vehicle_type,
            "detection_label": det_label,
            "segment_id":      segment_id,
        }

    @property
    def num_segments(self) -> int:
        return self._seg_counter


# ---------------------------------------------------------------------------
# StratifiedPairDataset
# ---------------------------------------------------------------------------

# Stratum codes stored in partner_stratum_p{p}
STRATUM_CONSEC     = 0
STRATUM_SAME_TYPE  = 1
STRATUM_DIFF_TYPE  = 2
STRATUM_CROSS_DS   = 3


class StratifiedPairDataset(Dataset):
    """
    Yields (anchor, K partners) tuples where partners are drawn from four
    stratified pools pre-built at __init__ time for O(1) __getitem__.

    Anchor eligibility: both audio AND seismic must be available (so modality
    availability is a simple scalar per batch, not per-partner masks).

    Partner pools per anchor (stored as lists of flat _index positions):
        consec_pool     : [i_t+1] — always exactly one, the consecutive window
        same_type_pool  : same dataset, same vehicle_type, different gkey
        diff_type_pool  : same dataset, different vehicle_type (both valid ≥ 0)
        cross_ds_pool   : different dataset

    n_partners_* config fields control how many are drawn per __getitem__.
    Total partners K = 1 + n_partners_same_type + n_partners_diff_type + n_partners_cross_ds.
    """

    def __init__(self, sensor_dataset: SensorDataset):
        self.ds  = sensor_dataset
        cfg      = sensor_dataset.cfg
        self.n_same_type  = cfg.n_partners_same_type
        self.n_diff_type  = cfg.n_partners_diff_type
        self.n_cross_ds   = cfg.n_partners_cross_ds

        index = sensor_dataset._index
        groups = sensor_dataset._groups

        # ----------------------------------------------------------------
        # Step 1: anchor eligibility — both modalities must be present
        # ----------------------------------------------------------------
        pos_map: dict[tuple, int] = {(e[0], e[1]): i for i, e in enumerate(index)}

        # For each valid anchor, pre-resolve the consecutive partner index
        self._anchors:   list[int] = []
        self._consec_idx: list[int] = []  # parallel to _anchors

        for i, (gkey, w, vtype, det, a_sid, s_sid) in enumerate(index):
            grp = groups[gkey]
            # Anchor must have both modalities
            if grp["audio_stem"] is None or grp["seismic_stem"] is None:
                continue
            # Consecutive partner must exist in the same group
            tn_key = (gkey, w + 1)
            if tn_key not in pos_map:
                continue
            j = pos_map[tn_key]
            jgrp = groups[index[j][0]]
            if jgrp["audio_stem"] is None or jgrp["seismic_stem"] is None:
                continue
            self._anchors.append(i)
            self._consec_idx.append(j)

        # ----------------------------------------------------------------
        # Step 2: build per-stratum candidate pools over all eligible entries
        #         (any index entry where both modalities exist)
        # ----------------------------------------------------------------
        # Buckets: dataset → list of flat index positions
        # Sub-bucket: (dataset, vehicle_type) → list of flat index positions
        #   vehicle_type used only when ≥ 0 (skip background/multi)
        by_dataset:       dict[str, list[int]] = {}
        by_ds_vtype:      dict[tuple, list[int]] = {}  # (dataset, vtype) → [idx]

        for i, (gkey, w, vtype, det, a_sid, s_sid) in enumerate(index):
            grp = groups[gkey]
            if grp["audio_stem"] is None or grp["seismic_stem"] is None:
                continue
            ds_name = gkey[0]  # gkey = (dataset, vehicle, rs_node, seg_key)
            by_dataset.setdefault(ds_name, []).append(i)
            if vtype >= 0:
                by_ds_vtype.setdefault((ds_name, vtype), []).append(i)

        # ----------------------------------------------------------------
        # Step 3: for each anchor, build the three random-partner pools
        # ----------------------------------------------------------------
        # Stored as lists-of-lists; pool[anchor_pos] = list[int]
        self._same_type_pool: list[list[int]] = []
        self._diff_type_pool: list[list[int]] = []
        self._cross_ds_pool:  list[list[int]] = []

        all_eligible = [j for bucket in by_dataset.values() for j in bucket]

        # Pre-build cross-dataset pools once per dataset (O(datasets × eligible))
        # rather than O(anchors × eligible) if built per anchor.
        cross_by_ds: dict[str, list[int]] = {
            ds_name: [j for ds, bucket in by_dataset.items() if ds != ds_name for j in bucket]
            for ds_name in by_dataset
        }

        for anchor_pos, i in enumerate(self._anchors):
            gkey, w, vtype, det, _, _ = index[i]
            ds_name = gkey[0]

            # same_type: same dataset, same vehicle_type, different gkey
            if vtype >= 0:
                candidates = [j for j in by_ds_vtype.get((ds_name, vtype), [])
                              if index[j][0] != gkey]
            else:
                candidates = []
            self._same_type_pool.append(candidates if candidates else all_eligible)

            # diff_type: same dataset, different valid vehicle_type
            diff = [j for j in by_dataset.get(ds_name, [])
                    if index[j][2] >= 0 and index[j][2] != vtype and index[j][0] != gkey]
            self._diff_type_pool.append(diff if diff else all_eligible)

            # cross_ds: different dataset entirely (shared reference, not a copy)
            cross = cross_by_ds.get(ds_name, [])
            self._cross_ds_pool.append(cross if cross else all_eligible)

        n_anchors = len(self._anchors)
        print(
            f"  StratifiedPairDataset: {n_anchors} anchors | "
            f"avg pool sizes: same_type={sum(len(p) for p in self._same_type_pool)//max(n_anchors,1)} "
            f"diff_type={sum(len(p) for p in self._diff_type_pool)//max(n_anchors,1)} "
            f"cross_ds={sum(len(p) for p in self._cross_ds_pool)//max(n_anchors,1)}"
        )

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._anchors)

    def _pick(self, pool: list[int]) -> int:
        """O(1) uniform draw from a pre-built pool."""
        return pool[torch.randint(len(pool), (1,)).item()]

    def _fetch(self, flat_idx: int, interv: int) -> dict:
        """Load both modality windows for a given flat _index position."""
        gkey, w, vtype, det, _, _ = self.ds._index[flat_idx]
        grp = self.ds._groups[gkey]
        return {
            "x_audio":         self.ds._get_window("audio",   grp["audio_stem"],   grp["seg_key"], w, interv),
            "x_seismic":       self.ds._get_window("seismic", grp["seismic_stem"], grp["seg_key"], w, interv),
            "vehicle_type":    vtype,
            "detection_label": det,
        }

    def _rand_interv(self) -> int:
        return _sample_interv(self.ds.is_train)

    def __getitem__(self, idx: int) -> dict:
        i_t  = self._anchors[idx]
        i_tn = self._consec_idx[idx]

        anchor = self._fetch(i_t,  self._rand_interv())
        consec = self._fetch(i_tn, self._rand_interv())

        partners = [(consec, STRATUM_CONSEC)]

        for _ in range(self.n_same_type):
            j = self._pick(self._same_type_pool[idx])
            partners.append((self._fetch(j, self._rand_interv()), STRATUM_SAME_TYPE))

        for _ in range(self.n_diff_type):
            j = self._pick(self._diff_type_pool[idx])
            partners.append((self._fetch(j, self._rand_interv()), STRATUM_DIFF_TYPE))

        for _ in range(self.n_cross_ds):
            j = self._pick(self._cross_ds_pool[idx])
            partners.append((self._fetch(j, self._rand_interv()), STRATUM_CROSS_DS))

        item = {
            "x_audio_t":          anchor["x_audio"],
            "x_seismic_t":        anchor["x_seismic"],
            "audio_avail":        True,
            "seismic_avail":      True,
            "detection_label_t":  anchor["detection_label"],
            "vehicle_type_t":     anchor["vehicle_type"],
        }
        for p, (partner, stratum) in enumerate(partners):
            item[f"x_audio_p{p}"]          = partner["x_audio"]
            item[f"x_seismic_p{p}"]        = partner["x_seismic"]
            item[f"detection_label_p{p}"]  = partner["detection_label"]
            item[f"vehicle_type_p{p}"]     = partner["vehicle_type"]
            item[f"partner_stratum_p{p}"]  = stratum

        return item


# ---------------------------------------------------------------------------
# Collate functions
# ---------------------------------------------------------------------------

def collate_single(batch: list) -> dict:
    return {
        "x_audio":         torch.stack([b["x_audio"]    for b in batch]),
        "x_seismic":       torch.stack([b["x_seismic"]  for b in batch]),
        "audio_avail":     torch.tensor([b["audio_avail"]   for b in batch], dtype=torch.bool),
        "seismic_avail":   torch.tensor([b["seismic_avail"] for b in batch], dtype=torch.bool),
        "interv_idx":      torch.tensor([b["interv_idx"]      for b in batch], dtype=torch.long),
        "vehicle_type":    torch.tensor([b["vehicle_type"]     for b in batch], dtype=torch.long),
        "detection_label": torch.tensor([b["detection_label"]  for b in batch], dtype=torch.long),
        "segment_id":      torch.tensor([b["segment_id"]       for b in batch], dtype=torch.long),
    }


def collate_pairs(batch: list) -> dict:
    """Collate for StratifiedPairDataset. Discovers partner slots dynamically."""
    out = {
        "x_audio_t":         torch.stack([b["x_audio_t"]   for b in batch]),
        "x_seismic_t":       torch.stack([b["x_seismic_t"] for b in batch]),
        "audio_avail":       torch.tensor([b["audio_avail"]   for b in batch], dtype=torch.bool),
        "seismic_avail":     torch.tensor([b["seismic_avail"] for b in batch], dtype=torch.bool),
        "detection_label_t": torch.tensor([b["detection_label_t"] for b in batch], dtype=torch.long),
        "vehicle_type_t":    torch.tensor([b["vehicle_type_t"]    for b in batch], dtype=torch.long),
    }
    # Discover how many partner slots exist from the first sample
    p = 0
    while f"x_audio_p{p}" in batch[0]:
        out[f"x_audio_p{p}"]         = torch.stack([b[f"x_audio_p{p}"]   for b in batch])
        out[f"x_seismic_p{p}"]       = torch.stack([b[f"x_seismic_p{p}"] for b in batch])
        out[f"detection_label_p{p}"] = torch.tensor([b[f"detection_label_p{p}"] for b in batch], dtype=torch.long)
        out[f"vehicle_type_p{p}"]    = torch.tensor([b[f"vehicle_type_p{p}"]    for b in batch], dtype=torch.long)
        out[f"partner_stratum_p{p}"] = torch.tensor([b[f"partner_stratum_p{p}"] for b in batch], dtype=torch.long)
        p += 1
    out["n_partners"] = p
    return out


# Backward-compat alias: smoke_test.py and other callers import ConsecutivePairDataset.
ConsecutivePairDataset = StratifiedPairDataset
