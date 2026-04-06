"""
MultiModalCausalDataset

Loads audio, seismic, and accel parquet files into RAM and serves
consecutive (t, t+1) window pairs for CRL pre-training, plus
detection and category labels for downstream head training.

Filename convention: {dataset}_{sensor}_{vehicle}_{rs}.parquet
  - dataset : "focal" | "m3nvc" | "iobt"
  - sensor  : "audio" | "seismic" | "accel"
  - vehicle : vehicle name (may contain underscores, e.g. "cx30_miata")
  - rs      : recording session identifier, e.g. "rs1"

m3nvc files contain multiple independent recording runs inside a single
parquet, identified by (scene_id, run_id) columns. The dataset splits
these into separate segments so that consecutive-window pairs (t, t+1)
never cross a run boundary, and each run gets its own sensor_domain_id
for the iVAE prior.

focal and iobt files are single continuous recordings (no scene_id /
run_id columns), treated as a single segment.

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
    NATIVE_SR,
    REF_SR,
    SAMPLE_SECONDS,
    DATASET_VEHICLE_MAP,
    CATEGORY_TO_IDX,
    LABEL_BACKGROUND,
    LABEL_MULTI,
    ADC_SCALE,
    MODALITY_CHANNELS,
    MODALITIES,
)


# Target sample rates per modality to avoid massive upsampling of low-freq sensors
TARGET_SR = {
    "audio": 16000,
    "seismic": 200,
    "accel": 200,
}

# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

_RS_RE = re.compile(r"_rs\d+$")


def parse_filename(stem: str):
    """
    Parse a parquet stem into (dataset, sensor, vehicle, rs_node).
    Returns None for unrecognised stems.
    """
    parts = stem.split("_", 2)
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
    rs_node = rest[rs_match.start() + 1 :]  # "rs1"
    vehicle = rest[: rs_match.start()]  # "cx30_miata"
    return dataset, sensor, vehicle, rs_node


def _vehicle_to_labels(dataset: str, vehicle: str):
    """
    Return (category_label, is_valid).
    category_label: LABEL_MULTI | LABEL_BACKGROUND | 0-3
    is_valid: False if vehicle is unknown (file should be skipped).
    """
    ds_map = DATASET_VEHICLE_MAP.get(dataset, {})
    category_str = ds_map.get(vehicle)

    # Prefix match for iobt names with timestamp suffixes (e.g. "polaris0150pm")
    if category_str is None:
        for key in sorted(ds_map.keys(), key=len, reverse=True):
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
# Segment key type
# ---------------------------------------------------------------------------

# For m3nvc: (scene_id, run_id) int tuple.
# For focal / iobt: None (single segment per file).
SegKey = type(None) | tuple


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class MultiModalCausalDataset(Dataset):
    """
    Args:
        parquet_dir        : directory containing parquet files
        filter_present     : if True, skip windows where present=False.
                             Set to False for detection evaluation so all
                             windows (present and not-present) are included.
        include_modalities : subset of ["audio","seismic","accel"] to load;
                             None means all available
    """

    def __init__(
        self,
        parquet_dir: str,
        filter_present: bool = False,
        include_modalities: list = None,
        domain_map: dict = None,
    ):
        self.parquet_dir = Path(parquet_dir)
        self.filter_present = filter_present
        self.include_modalities = set(include_modalities or MODALITIES)
        self.is_train = domain_map is None

        # RAM cache: (stem, seg_key) → {"data": np.ndarray [C,T],
        #                               "present": np.ndarray bool,
        #                               "native_sr": int}
        # seg_key = (scene_id, run_id) for m3nvc, None for focal/iobt
        self._cache: dict = {}

        # Resampler cache: (orig_freq, ref_freq) → Resample transform
        self._resamplers: dict = {}

        # (dataset, rs_node[, run_id]) → int
        self._domain_to_id: dict = (
            domain_map.copy() if domain_map is not None else {"__UNKNOWN__": 0}
        )

        # Index: (group_key, window_idx, available_mods, category_label, domain_id)
        # group_key = (dataset, vehicle, rs_node, seg_key)
        self._index: list = []

        # group_key → {modality: (stem, seg_key)}
        self._group_files: dict = {}

        print(f"Building index from {self.parquet_dir} ...")
        self._build_index()
        print(
            f"  {len(self._index)} window pairs indexed across "
            f"{len(self._domain_to_id)} sensor domains."
        )

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _build_index(self):
        parquet_files = sorted(self.parquet_dir.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files in {self.parquet_dir}")

        # Pass 1: group parquet files by (dataset, vehicle, rs_node)
        file_groups: dict = {}  # (dataset, vehicle, rs_node) → {sensor: stem}
        for p in parquet_files:
            parsed = parse_filename(p.stem)
            if parsed is None:
                continue
            dataset, sensor, vehicle, rs_node = parsed
            if sensor not in self.include_modalities:
                continue
            fkey = (dataset, vehicle, rs_node)
            file_groups.setdefault(fkey, {})[sensor] = p.stem

        # Pass 2: load files, split m3nvc by (scene_id, run_id), build index
        for (dataset, vehicle, rs_node), sensor_stems in file_groups.items():
            category_label, valid = _vehicle_to_labels(dataset, vehicle)
            if not valid:
                continue

            # Load each sensor modality; collect all (stem, seg_key, n_windows)
            # per modality.  seg_key = None for focal/iobt.
            mod_segments: dict = {}  # modality → {seg_key: (stem, n_windows)}
            for sensor, stem in sensor_stems.items():
                p = self.parquet_dir / f"{stem}.parquet"
                segs = self._load_file(p, stem, sensor, dataset)
                if segs:
                    mod_segments[sensor] = segs

            if not mod_segments:
                continue

            # Collect all segment keys that appear in *any* loaded modality
            all_seg_keys: set = set()
            for segs in mod_segments.values():
                all_seg_keys.update(segs.keys())

            for seg_key in sorted(all_seg_keys):
                group_key = (dataset, vehicle, rs_node, seg_key)

                # Build domain ID: include scene_id and run_id for m3nvc
                if seg_key is not None:
                    scene_id, run_id = seg_key
                    dkey = (dataset, rs_node, scene_id, run_id)
                else:
                    dkey = (dataset, rs_node)

                if dkey not in self._domain_to_id:
                    if self.is_train:
                        self._domain_to_id[dkey] = len(self._domain_to_id)

                domain_id = self._domain_to_id.get(dkey, 0)

                # Determine available modalities for this seg_key and min windows
                available_mods = []
                min_windows = None
                group_file_map = {}
                for sensor, segs in mod_segments.items():
                    if seg_key in segs:
                        stem, n_windows = segs[seg_key]
                        available_mods.append(sensor)
                        group_file_map[sensor] = (stem, seg_key)
                        if min_windows is None or n_windows < min_windows:
                            min_windows = n_windows

                if not available_mods or min_windows is None or min_windows < 2:
                    continue

                self._group_files[group_key] = group_file_map
                avail_frozen = frozenset(available_mods)

                for w in range(min_windows - 1):
                    if self.filter_present:
                        first_mod = next(iter(available_mods))
                        stem, sk = group_file_map[first_mod]
                        entry = self._cache[(stem, sk)]
                        win_len = int(entry["native_sr"] * SAMPLE_SECONDS)
                        row_idx = w * win_len
                        if (
                            row_idx < len(entry["present"])
                            and not entry["present"][row_idx]
                        ):
                            continue

                    self._index.append(
                        (group_key, w, avail_frozen, category_label, domain_id)
                    )

    def _load_file(
        self,
        path: Path,
        stem: str,
        sensor: str,
        dataset: str,
    ) -> dict:
        """
        Load a parquet file into RAM, splitting m3nvc files by (scene_id, run_id).

        Returns:
            {seg_key: (stem, n_windows)}
            seg_key = (scene_id, run_id) for m3nvc, None for focal/iobt.
            Empty dict on error.
        """
        try:
            df = pd.read_parquet(path)
        except Exception as e:
            print(f"  Warning: could not read {path}: {e}")
            return {}

        native_sr = NATIVE_SR.get(dataset, {}).get(sensor)
        if native_sr is None:
            return {}

        # For seismic m3nvc files there is a single-value 'channel' column;
        # we just ignore it — amplitude is already per-channel.

        has_segments = "scene_id" in df.columns and "run_id" in df.columns

        result = {}

        if has_segments:
            for (scene_id, run_id), seg_df in df.groupby(
                ["scene_id", "run_id"], sort=True
            ):
                seg_key = (int(scene_id), int(run_id))
                cache_key = (stem, seg_key)
                if cache_key not in self._cache:
                    entry = self._df_to_entry(seg_df, sensor, native_sr)
                    if entry is None:
                        continue
                    self._cache[cache_key] = entry
                entry = self._cache[cache_key]
                win_len = int(native_sr * SAMPLE_SECONDS)
                n_windows = entry["data"].shape[-1] // win_len
                if n_windows >= 2:
                    result[seg_key] = (stem, n_windows)
        else:
            seg_key = None
            cache_key = (stem, None)
            if cache_key not in self._cache:
                entry = self._df_to_entry(df, sensor, native_sr)
                if entry is None:
                    return {}
                self._cache[cache_key] = entry
            entry = self._cache[cache_key]
            win_len = int(native_sr * SAMPLE_SECONDS)
            n_windows = entry["data"].shape[-1] // win_len
            if n_windows >= 2:
                result[None] = (stem, n_windows)

        return result

    @staticmethod
    def _df_to_entry(df: pd.DataFrame, sensor: str, native_sr: int) -> dict | None:
        """Convert a (possibly segmented) DataFrame slice to a cache entry."""
        if sensor == "accel":
            required = ["accel_x_ew", "accel_y_ns", "accel_z_ud"]
            if not all(c in df.columns for c in required):
                return None
            arr = df[required].to_numpy(dtype=np.float32).T  # [3, T]
        else:
            if "amplitude" not in df.columns:
                return None
            arr = df["amplitude"].to_numpy(dtype=np.float32)[np.newaxis, :]  # [1, T]

        arr = arr / ADC_SCALE.get(sensor, 1.0)

        present = (
            df["present"].to_numpy(dtype=bool)
            if "present" in df.columns
            else np.ones(arr.shape[-1], dtype=bool)
        )
        return {"data": arr, "present": present, "native_sr": native_sr}

    # ------------------------------------------------------------------
    # Resampling
    # ------------------------------------------------------------------

    def _resample(
        self, tensor: torch.Tensor, orig_sr: int, target_sr: int
    ) -> torch.Tensor:
        if orig_sr == target_sr:
            return tensor
        key = (orig_sr, target_sr)
        if key not in self._resamplers:
            self._resamplers[key] = torchaudio.transforms.Resample(
                orig_freq=orig_sr, new_freq=target_sr
            )
        return self._resamplers[key](tensor)

    # ------------------------------------------------------------------
    # Window extraction
    # ------------------------------------------------------------------

    def _get_window(self, stem: str, seg_key: SegKey, w: int, mod: str) -> torch.Tensor:
        """Extract the w-th 1-second window within a segment and resample."""
        entry = self._cache[(stem, seg_key)]
        arr = entry["data"]
        native_sr = entry["native_sr"]
        win_len = int(native_sr * SAMPLE_SECONDS)
        start = w * win_len
        chunk = arr[:, start : start + win_len]  # [C, win_len]
        tensor = torch.from_numpy(chunk.copy())

        # Zero-mean center the 1-second window to remove DC offset / thermal drift
        tensor = tensor - tensor.mean(dim=-1, keepdim=True)

        return self._resample(tensor, native_sr, TARGET_SR[mod])  # [C, target_sr]

    # ------------------------------------------------------------------
    # __len__ / __getitem__
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        group_key, w, available_mods, category_label, domain_id = self._index[idx]
        dataset, vehicle, rs_node, seg_key = group_key
        mod_file_map = self._group_files[group_key]  # {mod: (stem, seg_key)}

        modality_t: dict = {}
        modality_next: dict = {}
        availability: dict = {}

        for mod in MODALITIES:
            if mod in available_mods and mod in mod_file_map:
                stem, sk = mod_file_map[mod]
                modality_t[mod] = self._get_window(stem, sk, w, mod)
                modality_next[mod] = self._get_window(stem, sk, w + 1, mod)
                availability[mod] = True
            else:
                modality_t[mod] = None
                modality_next[mod] = None
                availability[mod] = False

        # detection_label: present flag at window w from the first available modality
        first_mod = next(m for m in MODALITIES if availability[m])
        stem, sk = mod_file_map[first_mod]
        entry = self._cache[(stem, sk)]
        win_len = int(entry["native_sr"] * SAMPLE_SECONDS)
        row_idx = w * win_len
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
    availability is returned as a bool tensor [B, num_modalities] ordered
    by the MODALITIES list.
    """
    (
        modality_t_list,
        modality_next_list,
        avail_list,
        domain_ids,
        cat_labels,
        det_labels,
    ) = zip(*batch)

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
        return torch.stack(
            [d[mod] if d[mod] is not None else torch.zeros(shape) for d in dicts]
        )

    batch_t = {mod: stack_mod(modality_t_list, mod) for mod in MODALITIES}
    batch_next = {mod: stack_mod(modality_next_list, mod) for mod in MODALITIES}

    avail_tensor = torch.tensor(
        [[a[mod] for mod in MODALITIES] for a in avail_list],
        dtype=torch.bool,
    )
    domain_ids = torch.tensor(domain_ids, dtype=torch.long)
    cat_labels = torch.tensor(cat_labels, dtype=torch.long)
    det_labels = torch.tensor(det_labels, dtype=torch.long)

    return batch_t, batch_next, avail_tensor, domain_ids, cat_labels, det_labels
