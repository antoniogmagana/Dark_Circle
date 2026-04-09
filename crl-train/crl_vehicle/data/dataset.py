"""
SensorDataset and MultiHorizonPairDataset

Reads seismic AND audio parquet files, returning them independently so that
each modality can be processed through its own Filterbank → SSM → Encoder
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

Windows overlap: stride = cfg.horizon_stride_sec seconds (default 0.1 s).
This is ~10× more windows per recording than non-overlapping; raw waveforms
are cached once so there is no additional memory cost.

MultiHorizonPairDataset wraps SensorDataset for causal multi-horizon training:
    {
      'x_audio_t',   'x_audio_tn',
      'x_seismic_t', 'x_seismic_tn',
      'audio_avail', 'seismic_avail',
      'interv_idx_t', 'interv_idx_tn',
      'horizon_n',   — int in {1..n_horizons}
      'vehicle_type', 'detection_label', 'segment_id'
    }
"""

import re
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
    cat_str = ds_map.get(vehicle)
    if cat_str is None:
        for key in sorted(ds_map.keys(), key=len, reverse=True):
            if vehicle.startswith(key):
                cat_str = ds_map[key]
                break
    if cat_str is None:
        return None, False
    if cat_str == "multi":
        return LABEL_MULTI, True
    if cat_str == "background":
        return LABEL_BACKGROUND, True
    return CATEGORY_TO_IDX[cat_str], True


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

        result = {}
        has_segments = "scene_id" in df.columns and "run_id" in df.columns

        stride_native = max(int(native_sr * self.cfg.horizon_stride_sec), 1)
        if has_segments:
            for (scene_id, run_id), seg_df in df.groupby(["scene_id", "run_id"], sort=True):
                seg_key = (int(scene_id), int(run_id))
                cache_key = (stem, seg_key)
                if cache_key not in self._cache[sensor]:
                    entry = self._df_to_entry(seg_df, sensor, native_sr)
                    if entry is None:
                        continue
                    self._cache[sensor][cache_key] = entry
                nw = self._n_windows(self._cache[sensor][cache_key], native_sr, stride_native)
                if nw >= 1:
                    result[seg_key] = (stem, nw)
        else:
            cache_key = (stem, None)
            if cache_key not in self._cache[sensor]:
                entry = self._df_to_entry(df, sensor, native_sr)
                if entry is None:
                    return {}
                self._cache[sensor][cache_key] = entry
            nw = self._n_windows(self._cache[sensor][cache_key], native_sr, stride_native)
            if nw >= 1:
                result[None] = (stem, nw)

        return result

    def _n_windows(self, entry: dict, native_sr: int, stride_native: int | None = None) -> int:
        win_len = int(native_sr * self.cfg.sample_seconds)
        total = entry["data"].shape[-1]
        if total < win_len:
            return 0
        if stride_native is None:
            stride_native = max(int(native_sr * self.cfg.horizon_stride_sec), 1)
        return (total - win_len) // stride_native + 1

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
    # Resampling
    # ------------------------------------------------------------------

    def _resample(self, tensor: torch.Tensor, orig_sr: int, sensor: str) -> torch.Tensor:
        target_sr = _TARGET_SR[sensor]
        if orig_sr == target_sr:
            return tensor
        key = (orig_sr, target_sr)
        if key not in self._resamplers:
            self._resamplers[key] = torchaudio.transforms.Resample(
                orig_freq=orig_sr, new_freq=target_sr
            )
        return self._resamplers[key](tensor)

    # ------------------------------------------------------------------
    # Window extraction (per modality, independently)
    # ------------------------------------------------------------------

    def _get_window(
        self, sensor: str, stem: str, seg_key, w: int, interv_idx: int
    ) -> torch.Tensor:
        entry = self._cache[sensor][(stem, seg_key)]
        native_sr = entry["native_sr"]
        win_len = int(native_sr * self.cfg.sample_seconds)
        stride_native = max(int(native_sr * self.cfg.horizon_stride_sec), 1)
        start = w * stride_native
        chunk = entry["data"][:, start: start + win_len].copy()
        tensor = torch.from_numpy(chunk)                              # [1, win_len]
        tensor = self._resample(tensor, native_sr, sensor)            # [1, target_window_size]
        if self.is_train and interv_idx > 0:
            tensor = apply_intervention(tensor, interv_idx, _TARGET_SR[sensor])
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
        if self.is_train and torch.rand(1).item() < 0.60:
            interv_idx = torch.randint(1, N_INTERVENTIONS + 1, (1,)).item()
        else:
            interv_idx = 0

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
# MultiHorizonPairDataset
# ---------------------------------------------------------------------------

class MultiHorizonPairDataset(Dataset):
    """
    Wraps SensorDataset to yield (x_t, x_tn) pairs where n ∈ {1..n_horizons}
    is sampled uniformly at each call.  Both windows come from the same segment.

    Windows are spaced by horizon_stride_sec (from cfg), so x_t and x_tn are
    n * horizon_stride_sec seconds apart in the original recording.

    Interventions at t and t+n are assigned independently — the causal model
    must infer which latent variables changed without knowing the noise type
    at the target time step.
    """

    def __init__(self, sensor_dataset: SensorDataset, n_horizons: int | None = None):
        self.ds = sensor_dataset
        self.n_horizons = n_horizons if n_horizons is not None else sensor_dataset.cfg.n_horizons
        self._anchors: list[int] = []   # flat index positions valid as anchors
        index = sensor_dataset._index

        for i, (gkey, w, *_) in enumerate(index):
            group = sensor_dataset._groups[gkey]
            max_w = max(group["audio_nw"], group["seismic_nw"]) - 1
            if w + self.n_horizons <= max_w:
                self._anchors.append(i)

    def __len__(self) -> int:
        return len(self._anchors)

    def __getitem__(self, idx: int) -> dict:
        i_t = self._anchors[idx]
        gkey, w_t, vehicle_type, det_label, audio_seg_id, seismic_seg_id = self.ds._index[i_t]
        group = self.ds._groups[gkey]

        n = torch.randint(1, self.n_horizons + 1, (1,)).item()
        w_tn = w_t + n

        # Independent interventions at t and t+n.
        # Intentional: the causal classifier must detect changes without
        # knowing the intervention type at the target step.
        if self.ds.is_train and torch.rand(1).item() < 0.60:
            interv_t = torch.randint(1, N_INTERVENTIONS + 1, (1,)).item()
        else:
            interv_t = 0
        if self.ds.is_train and torch.rand(1).item() < 0.60:
            interv_tn = torch.randint(1, N_INTERVENTIONS + 1, (1,)).item()
        else:
            interv_tn = 0

        audio_avail = group["audio_stem"] is not None
        seismic_avail = group["seismic_stem"] is not None

        def get(sensor, w, interv):
            if sensor == "audio" and audio_avail and w < group["audio_nw"]:
                return self.ds._get_window(sensor, group["audio_stem"], group["seg_key"], w, interv)
            if sensor == "seismic" and seismic_avail and w < group["seismic_nw"]:
                return self.ds._get_window(sensor, group["seismic_stem"], group["seg_key"], w, interv)
            return self.ds._zero_window(sensor)

        return {
            "x_audio_t":       get("audio",   w_t,  interv_t),
            "x_audio_tn":      get("audio",   w_tn, interv_tn),
            "x_seismic_t":     get("seismic", w_t,  interv_t),
            "x_seismic_tn":    get("seismic", w_tn, interv_tn),
            "audio_avail":     audio_avail,
            "seismic_avail":   seismic_avail,
            "interv_idx_t":    interv_t,
            "interv_idx_tn":   interv_tn,
            "horizon_n":       n,
            "vehicle_type":    vehicle_type,
            "detection_label": det_label,
            "segment_id":      seismic_seg_id if seismic_avail else audio_seg_id,
        }


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
    return {
        "x_audio_t":       torch.stack([b["x_audio_t"]     for b in batch]),
        "x_audio_tn":      torch.stack([b["x_audio_tn"]    for b in batch]),
        "x_seismic_t":     torch.stack([b["x_seismic_t"]   for b in batch]),
        "x_seismic_tn":    torch.stack([b["x_seismic_tn"]  for b in batch]),
        "audio_avail":     torch.tensor([b["audio_avail"]   for b in batch], dtype=torch.bool),
        "seismic_avail":   torch.tensor([b["seismic_avail"] for b in batch], dtype=torch.bool),
        "interv_idx_t":    torch.tensor([b["interv_idx_t"]    for b in batch], dtype=torch.long),
        "interv_idx_tn":   torch.tensor([b["interv_idx_tn"]   for b in batch], dtype=torch.long),
        "horizon_n":       torch.tensor([b["horizon_n"]        for b in batch], dtype=torch.long),
        "vehicle_type":    torch.tensor([b["vehicle_type"]     for b in batch], dtype=torch.long),
        "detection_label": torch.tensor([b["detection_label"]  for b in batch], dtype=torch.long),
        "segment_id":      torch.tensor([b["segment_id"]       for b in batch], dtype=torch.long),
    }
