"""SensorDataset and StratifiedPairDataset for CRL training."""
from __future__ import annotations

import hashlib
import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
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
        self._cache: dict[str, dict] = {"audio": {}, "seismic": {}}
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

        if cache_dir is not None:
            cache_path = self._cache_path(cache_dir)
            if cache_path.exists():
                self._load_from_cache(cache_path)
                return

        self._build_from_parquet(parquet_files)

        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._save_cache(self._cache_path(cache_dir))

    def _cache_key(self) -> str:
        files = sorted(self.parquet_dir.glob("*.parquet"))
        h = hashlib.md5()
        for f in files:
            h.update(f.name.encode())
            h.update(str(f.stat().st_mtime).encode())
        return h.hexdigest()

    def _cache_path(self, cache_dir: Path) -> Path:
        return cache_dir / f"{self._cache_key()}.pkl"

    def _save_cache(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({
                "cache": self._cache,
                "index": self._index,
                "groups": self._groups,
            }, f)

    def _load_from_cache(self, path: Path) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._cache  = data["cache"]
        self._index  = data["index"]
        self._groups = data["groups"]

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
                df = pd.read_parquet(a_file)
                data = df.select_dtypes(include=[np.number]).values.astype(np.float32)
                self._cache["audio"][(audio_stem, None)] = {
                    "data": data, "n_windows": len(data)
                }
                audio_nw = len(data)
                seg_id += 1
                audio_seg_id = seg_id

            if s_file:
                df = pd.read_parquet(s_file)
                data = df.select_dtypes(include=[np.number]).values.astype(np.float32)
                self._cache["seismic"][(seismic_stem, None)] = {
                    "data": data, "n_windows": len(data)
                }
                seismic_nw = len(data)
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

    # ------------------------------------------------------------------
    # Window loading
    # ------------------------------------------------------------------

    def _get_window(
        self, sensor: str, stem: str, seg_key: Any, w: int, interv_idx: int
    ) -> torch.Tensor:
        entry = self._cache[sensor].get((stem, seg_key))
        if entry is None:
            mc = self.cfg.modality_cfg(sensor)
            return torch.zeros(1, mc.window_size)
        data = entry["data"]
        if w >= len(data):
            mc = self.cfg.modality_cfg(sensor)
            return torch.zeros(1, mc.window_size)
        x = torch.from_numpy(data[w]).unsqueeze(0)  # (1, W)
        x = remove_dc(x)
        if interv_idx > 0:
            mc = self.cfg.modality_cfg(sensor)
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
    """

    def __init__(self, sensor_dataset: SensorDataset) -> None:
        self.ds = sensor_dataset
        cfg = sensor_dataset.cfg

        self._anchors: list[int] = []   # indices into sensor_dataset._index
        self._pools: dict[str, list[int]] = {
            "consec":    [],
            "same_type": [],
            "diff_type": [],
            "cross_ds":  [],
        }

        self._n_same  = cfg.n_partners_same_type
        self._n_diff  = cfg.n_partners_diff_type
        self._n_cross = cfg.n_partners_cross_ds

        self._build_index()

    def _build_index(self) -> None:
        index = self.ds._index
        groups = self.ds._groups

        # Maps for pool building
        # gkey → list of global indices in that group
        gkey_to_indices: dict[tuple, list[int]] = {}
        for i, (gkey, w, vtype, det_label, a_seg, s_seg) in enumerate(index):
            gkey_to_indices.setdefault(gkey, []).append(i)

        # (dataset, vtype) → list of global indices (excluding this gkey)
        ds_vtype_to_indices: dict[tuple, list[int]] = {}
        for i, (gkey, w, vtype, det_label, a_seg, s_seg) in enumerate(index):
            ds_name = gkey[0]
            key = (ds_name, vtype)
            ds_vtype_to_indices.setdefault(key, []).append(i)

        # dataset → list of global indices
        ds_to_indices: dict[str, list[int]] = {}
        for i, (gkey, w, vtype, det_label, a_seg, s_seg) in enumerate(index):
            ds_name = gkey[0]
            ds_to_indices.setdefault(ds_name, []).append(i)

        # Anchor eligibility: must have a consecutive window in the same group
        for gkey, idx_list in gkey_to_indices.items():
            g = groups[gkey]
            if not (g["audio_stem"] is not None and g["seismic_stem"] is not None):
                continue  # both modalities required for anchor

            # Sort by window index within group
            sorted_idx = sorted(idx_list, key=lambda i: index[i][1])
            ds_name = gkey[0]
            vtype = groups[gkey]["vehicle_type"]

            for pos, global_idx in enumerate(sorted_idx[:-1]):  # skip last (no next)
                consec_idx = sorted_idx[pos + 1]
                self._anchors.append(global_idx)

                # Same-type pool: same dataset, same vtype, different gkey
                same_type_pool = [
                    j for j in ds_vtype_to_indices.get((ds_name, vtype), [])
                    if index[j][0] != gkey
                ]
                # Diff-type pool: same dataset, different valid vtype
                diff_type_pool = [
                    j for j in ds_to_indices.get(ds_name, [])
                    if index[j][2] != vtype and index[j][2] >= 0
                    and index[j][0] != gkey
                ]
                # Cross-dataset pool: any other dataset
                cross_ds_pool = [
                    j for j in range(len(index))
                    if index[j][0][0] != ds_name
                ]

                self._pools.setdefault("consec",    {})[global_idx] = [consec_idx]
                self._pools.setdefault("same_type", {})[global_idx] = same_type_pool or [consec_idx]
                self._pools.setdefault("diff_type", {})[global_idx] = diff_type_pool or [consec_idx]
                self._pools.setdefault("cross_ds",  {})[global_idx] = cross_ds_pool  or [consec_idx]

        # Rebuild as dicts if they were set as lists at class level
        if isinstance(self._pools.get("consec"), list):
            self._pools = {"consec": {}, "same_type": {}, "diff_type": {}, "cross_ds": {}}
            self._build_index()

    def _build_index(self) -> None:  # type: ignore[override]
        """Rebuild properly with dict-based pools."""
        index = self.ds._index
        groups = self.ds._groups

        self._anchors = []
        self._consec_pool:    dict[int, list[int]] = {}
        self._same_type_pool: dict[int, list[int]] = {}
        self._diff_type_pool: dict[int, list[int]] = {}
        self._cross_ds_pool:  dict[int, list[int]] = {}

        gkey_to_sorted: dict[tuple, list[int]] = {}
        for i, (gkey, w, vtype, det, a_seg, s_seg) in enumerate(index):
            gkey_to_sorted.setdefault(gkey, []).append(i)
        for gkey in gkey_to_sorted:
            gkey_to_sorted[gkey].sort(key=lambda i: index[i][1])

        ds_vtype_idx: dict[tuple, list[int]] = {}
        ds_idx: dict[str, list[int]] = {}
        for i, (gkey, w, vtype, det, a_seg, s_seg) in enumerate(index):
            ds = gkey[0]
            ds_vtype_idx.setdefault((ds, vtype), []).append(i)
            ds_idx.setdefault(ds, []).append(i)

        for gkey, sorted_list in gkey_to_sorted.items():
            g = groups[gkey]
            if not (g["audio_stem"] is not None and g["seismic_stem"] is not None):
                continue
            ds_name = gkey[0]
            vtype = g["vehicle_type"]

            for pos, global_idx in enumerate(sorted_list[:-1]):
                consec_idx = sorted_list[pos + 1]
                self._anchors.append(global_idx)

                same = [j for j in ds_vtype_idx.get((ds_name, vtype), [])
                        if index[j][0] != gkey]
                diff = [j for j in ds_idx.get(ds_name, [])
                        if index[j][2] != vtype and index[j][2] >= 0
                        and index[j][0] != gkey]
                cross = [j for j in range(len(index)) if index[j][0][0] != ds_name]

                self._consec_pool[global_idx]    = [consec_idx]
                self._same_type_pool[global_idx] = same  or [consec_idx]
                self._diff_type_pool[global_idx] = diff  or [consec_idx]
                self._cross_ds_pool[global_idx]  = cross or [consec_idx]

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

        partners: list[tuple[int, int]] = []  # (pool_idx, stratum)
        # p0: consecutive
        partners.append((random.choice(self._consec_pool[anchor_idx]),    STRATUM_CONSEC))
        # p1..p{n_same}: same type
        for _ in range(self._n_same):
            partners.append((random.choice(self._same_type_pool[anchor_idx]), STRATUM_SAME_TYPE))
        # next: diff type
        for _ in range(self._n_diff):
            partners.append((random.choice(self._diff_type_pool[anchor_idx]), STRATUM_DIFF_TYPE))
        # next: cross dataset
        for _ in range(self._n_cross):
            partners.append((random.choice(self._cross_ds_pool[anchor_idx]),  STRATUM_CROSS_DS))

        for p, (pidx, stratum) in enumerate(partners):
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
