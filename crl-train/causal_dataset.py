"""
MultiModalCausalDataset

Loads audio, seismic, and accel parquet files into RAM and serves
consecutive (t, t+1) window pairs for CRL pre-training, plus
detection and category labels for downstream head training.

Filename convention: {dataset}_{sensor}_{vehicle}_{rs}.parquet
  - dataset : "focal" | "m3nvc" | "iobt"
  - sensor  : "audio" | "seismic" | "accel"
  - vehicle : vehicle name (may contain underscores, e.g. "cx30_miata")
  - rs      : recording session, e.g. "rs1"

Label semantics:
  category_label  : -2 = multi-vehicle (ambiguous), -1 = background,
                     0-3 = pedestrian / light / sport / utility
  detection_label :  0 = vehicle not present, 1 = present
                    (derived from per-row "present" boolean column)
"""

import re
import numpy as np
import pandas as pd
import torch
import torchaudio
from pathlib import Path
from torch.utils.data import Dataset
from crl_config import (
    NATIVE_SR, REF_SR, SAMPLE_SECONDS,
    DATASET_VEHICLE_MAP, CATEGORY_TO_IDX,
    LABEL_BACKGROUND, LABEL_MULTI,
    ADC_SCALE, MODALITY_CHANNELS, MODALITIES,
)


# ---------------------------------------------------------------------------
# Filename parsing helpers
# ---------------------------------------------------------------------------

_RS_RE = re.compile(r"_rs\d+$")


def parse_filename(stem: str):
    """
    Parse a parquet stem into (dataset, sensor, vehicle, rs_node).

    Handles multi-word vehicle names by splitting on the first token
    (dataset), the second token (sensor), and stripping the trailing
    "_rs\d+" tag; everything in between is the vehicle name.

    Returns None for unrecognised stems.
    """
    parts = stem.split("_", 2)   # at most 3 parts: dataset, sensor, rest
    if len(parts) < 3:
        return None
    dataset, sensor, rest = parts
    if dataset not in NATIVE_SR:
        return None
    if sensor not in MODALITIES:
        return None
    rs_match = _RS_RE.search(rest)
    if not rs_match:
        return None
    rs_node = rest[rs_match.start() + 1:]   # e.g. "rs1"
    vehicle = rest[:rs_match.start()]        # e.g. "cx30_miata"
    return dataset, sensor, vehicle, rs_node


def _vehicle_to_labels(dataset: str, vehicle: str):
    """
    Return (category_label, is_valid).

    category_label:
      LABEL_MULTI (-2) — multi-vehicle recording, skip classification
      LABEL_BACKGROUND (-1) — background-only recording
      0-3 — pedestrian / light / sport / utility

    is_valid: False if vehicle is completely unknown (file should be skipped).
    """
    ds_map = DATASET_VEHICLE_MAP.get(dataset, {})

    # Exact match first
    category_str = ds_map.get(vehicle)

    # Prefix match for iobt names that have timestamps appended
    if category_str is None:
        sorted_keys = sorted(ds_map.keys(), key=len, reverse=True)
        for key in sorted_keys:
            if vehicle.startswith(key):
                category_str = ds_map[key]
                break

    if category_str is None:
        return None, False

    if category_str == "multi":
        return LABEL_MULTI, True
    if category_str == "background":
        return LABEL_BACKGROUND, True
    return CATEGORY_TO_IDX[category_str], True


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MultiModalCausalDataset(Dataset):
    """
    Args:
        parquet_dir        : directory containing parquet files
        filter_present     : if True, only include windows where present=True.
                             Set to False for detection evaluation (iobt) so
                             all windows (present and not-present) are included.
        include_modalities : subset of ["audio","seismic","accel"] to load;
                             None means all available
    """

    def __init__(
        self,
        parquet_dir: str,
        filter_present: bool = False,
        include_modalities: list = None,
    ):
        self.parquet_dir = Path(parquet_dir)
        self.filter_present = filter_present
        self.include_modalities = set(include_modalities or MODALITIES)

        # RAM cache: stem → {"data": np.ndarray [C,T], "present": np.ndarray bool, "native_sr": int}
        self._cache: dict = {}

        # Resampler cache: (orig_freq, ref_freq) → torchaudio.transforms.Resample
        self._resamplers: dict = {}

        # sensor_domain_id encoding: (dataset, rs_node) → int
        self._domain_to_id: dict = {}

        # Index entries:
        # (group_key, window_idx, available_mods_frozenset, category_label, domain_id)
        self._index: list = []

        # group_key → {modality: file_stem}
        self._group_files: dict = {}

        print(f"Building index from {self.parquet_dir} ...")
        self._build_index()
        print(f"  {len(self._index)} window pairs indexed across "
              f"{len(self._domain_to_id)} sensor domains.")

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _build_index(self):
        parquet_files = sorted(self.parquet_dir.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files in {self.parquet_dir}")

        # Pass 1: group files by (dataset, vehicle, rs_node)
        groups: dict = {}
        for p in parquet_files:
            parsed = parse_filename(p.stem)
            if parsed is None:
                continue
            dataset, sensor, vehicle, rs_node = parsed
            if sensor not in self.include_modalities:
                continue
            key = (dataset, vehicle, rs_node)
            groups.setdefault(key, {})[sensor] = p.stem

        # Pass 2: load and index each group
        for group_key, sensor_stems in groups.items():
            dataset, vehicle, rs_node = group_key
            category_label, valid = _vehicle_to_labels(dataset, vehicle)
            if not valid:
                continue

            domain_key = (dataset, rs_node)
            if domain_key not in self._domain_to_id:
                self._domain_to_id[domain_key] = len(self._domain_to_id)
            domain_id = self._domain_to_id[domain_key]

            available_mods = []
            min_windows = None
            for sensor, stem in sensor_stems.items():
                p = self.parquet_dir / f"{stem}.parquet"
                n_windows = self._load_file(p, stem, sensor, dataset)
                if n_windows is not None:
                    available_mods.append(sensor)
                    if min_windows is None or n_windows < min_windows:
                        min_windows = n_windows

            if not available_mods or min_windows is None or min_windows < 2:
                continue

            self._group_files[group_key] = sensor_stems
            available_mods_frozen = frozenset(available_mods)

            for w in range(min_windows - 1):
                if self.filter_present:
                    first_mod = next(iter(available_mods))
                    first_stem = sensor_stems[first_mod]
                    native_sr = self._cache[first_stem]["native_sr"]
                    win_len = int(native_sr * SAMPLE_SECONDS)
                    row_idx = w * win_len
                    present_arr = self._cache[first_stem]["present"]
                    if row_idx < len(present_arr) and not present_arr[row_idx]:
                        continue

                self._index.append(
                    (group_key, w, available_mods_frozen, category_label, domain_id)
                )

    def _load_file(self, path: Path, stem: str, sensor: str, dataset: str):
        """Load one parquet file into RAM. Returns number of complete windows."""
        if stem in self._cache:
            entry = self._cache[stem]
        else:
            try:
                df = pd.read_parquet(path)
            except Exception as e:
                print(f"  Warning: could not read {path}: {e}")
                return None

            native_sr = NATIVE_SR.get(dataset, {}).get(sensor)
            if native_sr is None:
                return None

            if sensor == "accel":
                required = ["accel_x_ew", "accel_y_ns", "accel_z_ud"]
                if not all(c in df.columns for c in required):
                    return None
                arr = df[required].to_numpy(dtype=np.float32).T  # [3, T]
                scale = ADC_SCALE["accel"]
            else:
                if "amplitude" not in df.columns:
                    return None
                arr = df["amplitude"].to_numpy(dtype=np.float32)[np.newaxis, :]  # [1, T]
                scale = ADC_SCALE.get(sensor, 1.0)

            arr = arr / scale

            present = (
                df["present"].to_numpy(dtype=bool)
                if "present" in df.columns
                else np.ones(arr.shape[-1], dtype=bool)
            )
            entry = {"data": arr, "present": present, "native_sr": native_sr}
            self._cache[stem] = entry

        native_sr = entry["native_sr"]
        window_samples = int(native_sr * SAMPLE_SECONDS)
        if window_samples == 0:
            return None
        return entry["data"].shape[-1] // window_samples

    # ------------------------------------------------------------------
    # Resampling
    # ------------------------------------------------------------------

    def _resample(self, tensor: torch.Tensor, orig_sr: int) -> torch.Tensor:
        if orig_sr == REF_SR:
            return tensor
        key = (orig_sr, REF_SR)
        if key not in self._resamplers:
            self._resamplers[key] = torchaudio.transforms.Resample(
                orig_freq=orig_sr, new_freq=REF_SR
            )
        return self._resamplers[key](tensor)

    # ------------------------------------------------------------------
    # Window extraction
    # ------------------------------------------------------------------

    def _get_window(self, stem: str, w: int) -> torch.Tensor:
        entry = self._cache[stem]
        arr = entry["data"]
        native_sr = entry["native_sr"]
        win_len = int(native_sr * SAMPLE_SECONDS)
        start = w * win_len
        chunk = arr[:, start: start + win_len]          # [C, win_len]
        tensor = torch.from_numpy(chunk.copy())
        return self._resample(tensor, native_sr)        # [C, REF_SR]

    # ------------------------------------------------------------------
    # __len__ / __getitem__
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        group_key, w, available_mods, category_label, domain_id = self._index[idx]
        dataset, vehicle, rs_node = group_key
        sensor_stems = self._group_files[group_key]

        modality_t: dict = {}
        modality_next: dict = {}
        availability: dict = {}

        for mod in MODALITIES:
            if mod in available_mods and mod in sensor_stems:
                stem = sensor_stems[mod]
                modality_t[mod]    = self._get_window(stem, w)
                modality_next[mod] = self._get_window(stem, w + 1)
                availability[mod]  = True
            else:
                modality_t[mod]    = None
                modality_next[mod] = None
                availability[mod]  = False

        # detection_label: present flag at window w
        first_mod  = next(m for m in MODALITIES if availability[m])
        first_stem = sensor_stems[first_mod]
        entry      = self._cache[first_stem]
        win_len    = int(entry["native_sr"] * SAMPLE_SECONDS)
        row_idx    = w * win_len
        present_arr = entry["present"]
        detection_label = int(present_arr[row_idx]) if row_idx < len(present_arr) else 0

        return (
            modality_t,
            modality_next,
            availability,
            domain_id,
            category_label,
            detection_label,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_sensor_domains(self) -> int:
        return len(self._domain_to_id)


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def collate_multimodal(batch):
    """
    Collate a list of (modality_t, modality_next, availability,
                        domain_id, category_label, detection_label) tuples.

    Absent modalities are zero-padded to the expected shape.
    Returns availability as a bool tensor [B, num_modalities] ordered
    by MODALITIES list.
    """
    modality_t_list, modality_next_list, avail_list, \
        domain_ids, cat_labels, det_labels = zip(*batch)

    # Determine expected shape per modality from first available sample
    mod_shapes = {}
    for mod in MODALITIES:
        for sample_dict in modality_t_list:
            if sample_dict[mod] is not None:
                mod_shapes[mod] = sample_dict[mod].shape
                break

    def stack_mod(dicts, mod):
        shape = mod_shapes.get(mod)
        if shape is None:
            return None
        tensors = [
            d[mod] if d[mod] is not None else torch.zeros(shape)
            for d in dicts
        ]
        return torch.stack(tensors)   # [B, C, T]

    batch_t    = {mod: stack_mod(modality_t_list, mod)    for mod in MODALITIES}
    batch_next = {mod: stack_mod(modality_next_list, mod) for mod in MODALITIES}

    avail_tensor = torch.tensor(
        [[a[mod] for mod in MODALITIES] for a in avail_list],
        dtype=torch.bool,
    )
    domain_ids = torch.tensor(domain_ids, dtype=torch.long)
    cat_labels = torch.tensor(cat_labels, dtype=torch.long)
    det_labels = torch.tensor(det_labels, dtype=torch.long)

    return batch_t, batch_next, avail_tensor, domain_ids, cat_labels, det_labels
