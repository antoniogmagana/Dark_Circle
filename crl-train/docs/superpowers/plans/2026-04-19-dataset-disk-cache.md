# Dataset Disk Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a disk cache to `SensorDataset` so the 21GB parquet parse only happens once — subsequent runs mmap the serialized arrays from disk in milliseconds.

**Architecture:** On first run, after `_build_index()` completes, serialize `_cache` (numpy arrays per modality) and `_index`/`_groups`/`_segment_id_map` (Python structures) to a versioned cache directory. On subsequent runs, detect the cache, load it via `numpy.load(..., mmap_mode='r')` for arrays and pickle for metadata, and skip `_build_index()` entirely. Cache is keyed by a hash of the parquet directory contents (filenames + mtimes) so it auto-invalidates when data changes. DataLoader workers are safe because mmap'd arrays are read-only and page-shared by the OS.

**Tech Stack:** Python stdlib (`hashlib`, `pickle`, `pathlib`), `numpy` (already a dependency via pandas/torchaudio), no new dependencies.

---

## File Structure

- **Modify:** `crl_vehicle/data/dataset.py` — add `_cache_dir`, `_compute_cache_key`, `_save_cache`, `_load_cache` methods to `SensorDataset`; wrap `_build_index` call with cache check in `__init__`
- **Create:** `tests/data/test_disk_cache.py` — tests for cache save/load/invalidation

---

### Task 1: Add cache key computation

**Files:**
- Modify: `crl_vehicle/data/dataset.py` (inside `SensorDataset`)
- Test: `tests/data/test_disk_cache.py`

- [ ] **Step 1: Create test file**

```python
# tests/data/test_disk_cache.py
import hashlib
import time
from pathlib import Path
import numpy as np
import pytest

from crl_vehicle.data.dataset import SensorDataset, _compute_dir_hash
from crl_vehicle.config import CRLConfig


def test_cache_key_stable(tmp_path):
    """Same directory contents → same hash."""
    (tmp_path / "a.parquet").write_bytes(b"x")
    (tmp_path / "b.parquet").write_bytes(b"y")
    h1 = _compute_dir_hash(tmp_path)
    h2 = _compute_dir_hash(tmp_path)
    assert h1 == h2


def test_cache_key_changes_on_new_file(tmp_path):
    """Adding a file → different hash."""
    (tmp_path / "a.parquet").write_bytes(b"x")
    h1 = _compute_dir_hash(tmp_path)
    (tmp_path / "b.parquet").write_bytes(b"y")
    h2 = _compute_dir_hash(tmp_path)
    assert h1 != h2


def test_cache_key_changes_on_mtime(tmp_path):
    """Touching a file → different hash."""
    f = tmp_path / "a.parquet"
    f.write_bytes(b"x")
    h1 = _compute_dir_hash(tmp_path)
    time.sleep(0.01)
    f.touch()
    h2 = _compute_dir_hash(tmp_path)
    assert h1 != h2
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd crl-train && poetry run pytest tests/data/test_disk_cache.py -v
```

Expected: `ImportError` — `_compute_dir_hash` doesn't exist yet.

- [ ] **Step 3: Add `_compute_dir_hash` as a module-level function in `dataset.py`**

Add after the existing imports, before `_parse_stem`:

```python
import hashlib
import pickle

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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
poetry run pytest tests/data/test_disk_cache.py::test_cache_key_stable \
    tests/data/test_disk_cache.py::test_cache_key_changes_on_new_file \
    tests/data/test_disk_cache.py::test_cache_key_changes_on_mtime -v
```

Expected: all 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add crl_vehicle/data/dataset.py tests/data/test_disk_cache.py
git commit -m "feat: add _compute_dir_hash for dataset cache key"
```

---

### Task 2: Implement cache save

**Files:**
- Modify: `crl_vehicle/data/dataset.py`
- Test: `tests/data/test_disk_cache.py`

The cache layout on disk for a given `cache_dir` and `cache_key`:

```
<cache_dir>/<cache_key>/
    audio.npy          # stacked float32 array: (N_audio_segs, 1, T_audio)
    seismic.npy        # stacked float32 array: (N_seismic_segs, 1, T_seismic)
    audio_meta.pkl     # list of (stem, seg_key) in same order as audio.npy rows
    seismic_meta.pkl   # list of (stem, seg_key) in same order as seismic.npy rows
    index.pkl          # _index list
    groups.pkl         # _groups dict
    segment_id_map.pkl # _segment_id_map dict + _seg_counter int
```

Arrays are stored as separate `.npy` files per modality so they can be `mmap`'d independently. Sequences within a modality can differ in length — store as object arrays of 1D views is **not** safe for mmap. Instead, pack all segments of the same modality into a single `.npy` after zero-padding to the maximum length, and store the actual length per segment in the metadata. `_get_window` already slices by `start:start+win_len`, so it just needs to know the true length to avoid reading padding.

- [ ] **Step 1: Write failing test**

Add to `tests/data/test_disk_cache.py`:

```python
def _make_minimal_cache(tmp_path):
    """Build minimal _cache and _index structures matching SensorDataset internals."""
    audio_data = np.zeros((1, 16000), dtype=np.float32)
    seismic_data = np.zeros((1, 200), dtype=np.float32)
    cache = {
        "audio":   {("stem_a", None): {"data": audio_data,   "present": np.ones(16000, dtype=bool), "native_sr": 16000, "target_sr": 16000}},
        "seismic": {("stem_s", None): {"data": seismic_data, "present": np.ones(200,   dtype=bool), "native_sr": 200,   "target_sr": 200}},
    }
    index = [((  "ds", "veh", "rs", None), 0, 0, 1, 0, 0)]
    groups = {("ds", "veh", "rs", None): {
        "audio_stem": "stem_a", "seismic_stem": "stem_s", "seg_key": None,
        "audio_nw": 1, "seismic_nw": 1, "vehicle_type": 0,
        "audio_seg_id": 0, "seismic_seg_id": 0,
    }}
    segment_id_map = {("audio", "stem_a", None): 0, ("seismic", "stem_s", None): 0}
    seg_counter = 1
    return cache, index, groups, segment_id_map, seg_counter


def test_save_cache_creates_files(tmp_path):
    from crl_vehicle.data.dataset import _save_cache
    cache, index, groups, sim, sc = _make_minimal_cache(tmp_path)
    cache_dir = tmp_path / "cache"
    _save_cache(cache_dir, "abc123", cache, index, groups, sim, sc)
    slot = cache_dir / "abc123"
    assert (slot / "audio.npy").exists()
    assert (slot / "seismic.npy").exists()
    assert (slot / "audio_meta.pkl").exists()
    assert (slot / "seismic_meta.pkl").exists()
    assert (slot / "index.pkl").exists()
    assert (slot / "groups.pkl").exists()
    assert (slot / "segment_id_map.pkl").exists()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
poetry run pytest tests/data/test_disk_cache.py::test_save_cache_creates_files -v
```

Expected: `ImportError` — `_save_cache` not defined.

- [ ] **Step 3: Implement `_save_cache` in `dataset.py`**

Add after `_compute_dir_hash`:

```python
def _save_cache(
    cache_dir: Path,
    cache_key: str,
    sensor_cache: dict,
    index: list,
    groups: dict,
    segment_id_map: dict,
    seg_counter: int,
) -> None:
    slot = cache_dir / cache_key
    slot.mkdir(parents=True, exist_ok=True)

    for sensor in ("audio", "seismic"):
        entries = sensor_cache[sensor]
        if not entries:
            np.save(slot / f"{sensor}.npy", np.empty((0,), dtype=np.float32))
            with open(slot / f"{sensor}_meta.pkl", "wb") as f:
                pickle.dump([], f)
            continue

        keys = list(entries.keys())   # list of (stem, seg_key)
        arrays = [entries[k]["data"]  for k in keys]  # each (1, T_i)
        max_len = max(a.shape[-1] for a in arrays)

        # Pad to uniform length so all rows stack into a single 2D array
        # Shape: (N, max_len) — channel dim squeezed; restored on load.
        padded = np.zeros((len(arrays), max_len), dtype=np.float32)
        true_lens = []
        for i, a in enumerate(arrays):
            L = a.shape[-1]
            padded[i, :L] = a[0]
            true_lens.append(L)

        np.save(slot / f"{sensor}.npy", padded)

        meta = [
            {
                "stem":       k[0],
                "seg_key":    k[1],
                "true_len":   true_lens[idx],
                "present":    entries[k]["present"],
                "native_sr":  entries[k]["native_sr"],
                "target_sr":  entries[k]["target_sr"],
            }
            for idx, k in enumerate(keys)
        ]
        with open(slot / f"{sensor}_meta.pkl", "wb") as f:
            pickle.dump(meta, f)

    with open(slot / "index.pkl", "wb") as f:
        pickle.dump(index, f)
    with open(slot / "groups.pkl", "wb") as f:
        pickle.dump(groups, f)
    with open(slot / "segment_id_map.pkl", "wb") as f:
        pickle.dump({"map": segment_id_map, "counter": seg_counter}, f)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
poetry run pytest tests/data/test_disk_cache.py::test_save_cache_creates_files -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crl_vehicle/data/dataset.py tests/data/test_disk_cache.py
git commit -m "feat: implement _save_cache for dataset disk cache"
```

---

### Task 3: Implement cache load

**Files:**
- Modify: `crl_vehicle/data/dataset.py`
- Test: `tests/data/test_disk_cache.py`

Loading reconstructs `_cache` from mmap'd arrays. Each entry in `_cache[sensor]` must have the same shape as what `_get_window` expects: `{"data": ndarray (1, T), "present": ..., "native_sr": ..., "target_sr": ...}`. The mmap'd array is `(N, max_len)` — reconstruct the `(1, true_len)` view per entry by slicing `arr[row_idx:row_idx+1, :true_len]`. This is a zero-copy view into the mmap'd file.

- [ ] **Step 1: Write failing test**

Add to `tests/data/test_disk_cache.py`:

```python
def test_roundtrip_cache(tmp_path):
    from crl_vehicle.data.dataset import _save_cache, _load_cache
    cache, index, groups, sim, sc = _make_minimal_cache(tmp_path)
    cache_dir = tmp_path / "cache"
    _save_cache(cache_dir, "abc123", cache, index, groups, sim, sc)

    loaded_cache, loaded_index, loaded_groups, loaded_sim, loaded_sc = \
        _load_cache(cache_dir, "abc123")

    # Data round-trips correctly
    orig = cache["audio"][("stem_a", None)]["data"]
    loaded = loaded_cache["audio"][("stem_a", None)]["data"]
    np.testing.assert_array_equal(orig, loaded)

    # Metadata preserved
    assert loaded_index == index
    assert loaded_groups == groups
    assert loaded_sim == sim
    assert loaded_sc == sc


def test_load_cache_returns_none_for_missing_key(tmp_path):
    from crl_vehicle.data.dataset import _load_cache
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    result = _load_cache(cache_dir, "missing")
    assert result is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
poetry run pytest tests/data/test_disk_cache.py::test_roundtrip_cache \
    tests/data/test_disk_cache.py::test_load_cache_returns_none_for_missing_key -v
```

Expected: `ImportError` — `_load_cache` not defined.

- [ ] **Step 3: Implement `_load_cache` in `dataset.py`**

Add after `_save_cache`:

```python
def _load_cache(cache_dir: Path, cache_key: str):
    """
    Returns (sensor_cache, index, groups, segment_id_map, seg_counter)
    or None if the cache slot does not exist.
    """
    slot = cache_dir / cache_key
    if not slot.exists():
        return None

    try:
        sensor_cache = {"audio": {}, "seismic": {}}
        for sensor in ("audio", "seismic"):
            arr = np.load(slot / f"{sensor}.npy", mmap_mode="r")  # (N, max_len)
            with open(slot / f"{sensor}_meta.pkl", "rb") as f:
                meta = pickle.load(f)
            for row_idx, m in enumerate(meta):
                L = m["true_len"]
                # Zero-copy slice: view into mmap'd array, reshape to (1, L)
                data_view = arr[row_idx:row_idx + 1, :L]
                sensor_cache[sensor][(m["stem"], m["seg_key"])] = {
                    "data":      data_view,
                    "present":   m["present"],
                    "native_sr": m["native_sr"],
                    "target_sr": m["target_sr"],
                }

        with open(slot / "index.pkl", "rb") as f:
            index = pickle.load(f)
        with open(slot / "groups.pkl", "rb") as f:
            groups = pickle.load(f)
        with open(slot / "segment_id_map.pkl", "rb") as f:
            sid = pickle.load(f)

        return sensor_cache, index, groups, sid["map"], sid["counter"]

    except Exception as e:
        print(f"  Warning: disk cache load failed ({e}), rebuilding.")
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
poetry run pytest tests/data/test_disk_cache.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add crl_vehicle/data/dataset.py tests/data/test_disk_cache.py
git commit -m "feat: implement _load_cache with mmap for dataset disk cache"
```

---

### Task 4: Wire cache into `SensorDataset.__init__`

**Files:**
- Modify: `crl_vehicle/data/dataset.py` — `SensorDataset.__init__` and `_get_window`
- Test: `tests/data/test_disk_cache.py`

Two wiring points:
1. `__init__`: accept `cache_dir` param, check cache before `_build_index`, save after.
2. `_get_window`: the mmap'd `data_view` is shape `(1, L)` — same as what was stored before, so `_get_window` needs no changes. But `data_view` is read-only (mmap `mode='r'`), and `_get_window` does `.copy()` already (`entry["data"][:, start:start+win_len].copy()`), so workers get writable copies. No change needed to `_get_window`.

- [ ] **Step 1: Write failing test**

Add to `tests/data/test_disk_cache.py`:

```python
def test_sensor_dataset_writes_cache(tmp_path, monkeypatch):
    """SensorDataset writes a cache on first construction."""
    import crl_vehicle.data.dataset as ds_mod

    # Stub out _build_index to avoid needing real parquet files
    built = []
    def fake_build(self):
        self._cache = {
            "audio":   {("stem_a", None): {"data": np.zeros((1, 16000), dtype=np.float32), "present": np.ones(16000, dtype=bool), "native_sr": 16000, "target_sr": 16000}},
            "seismic": {("stem_s", None): {"data": np.zeros((1, 200),   dtype=np.float32), "present": np.ones(200,   dtype=bool), "native_sr": 200,   "target_sr": 200}},
        }
        self._index = []
        self._groups = {}
        self._segment_id_map = {}
        self._seg_counter = 0
        built.append(True)

    monkeypatch.setattr(ds_mod.SensorDataset, "_build_index", fake_build)
    monkeypatch.setattr(ds_mod, "_compute_dir_hash", lambda p: "testhash")

    cache_dir = tmp_path / "cache"
    cfg = CRLConfig()

    # Patch glob so no FileNotFoundError on missing parquet dir
    real_parquet_dir = tmp_path / "parquet"
    real_parquet_dir.mkdir()

    ds = ds_mod.SensorDataset(str(real_parquet_dir), cfg, cache_dir=cache_dir)
    assert len(built) == 1
    assert (cache_dir / "testhash" / "audio.npy").exists()

    # Second construction should NOT call _build_index
    built.clear()
    ds2 = ds_mod.SensorDataset(str(real_parquet_dir), cfg, cache_dir=cache_dir)
    assert len(built) == 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
poetry run pytest tests/data/test_disk_cache.py::test_sensor_dataset_writes_cache -v
```

Expected: FAIL — `SensorDataset.__init__` does not accept `cache_dir`.

- [ ] **Step 3: Modify `SensorDataset.__init__` to accept and use `cache_dir`**

Replace the existing `__init__` signature and the `_build_index()` call block:

```python
def __init__(
    self, parquet_dir: str, config: CRLConfig, is_train: bool = True,
    cache_dir: Path | None = None,
):
    self.parquet_dir = Path(parquet_dir)
    self.cfg = config
    self.is_train = is_train

    self._cache: dict[str, dict] = {m: {} for m in MODALITIES}
    self._resamplers: dict = {}
    self._segment_id_map: dict = {}
    self._seg_counter = 0
    self._groups: dict = {}
    self._index: list = []

    loaded = False
    if cache_dir is not None:
        cache_key = _compute_dir_hash(self.parquet_dir)
        result = _load_cache(cache_dir, cache_key)
        if result is not None:
            self._cache, self._index, self._groups, self._segment_id_map, self._seg_counter = result
            print(f"  SensorDataset [{self.parquet_dir.name}]: loaded from disk cache ({cache_key})")
            loaded = True

    if not loaded:
        self._build_index()
        if cache_dir is not None:
            print(f"  SensorDataset [{self.parquet_dir.name}]: saving disk cache ({cache_key})")
            _save_cache(cache_dir, cache_key, self._cache, self._index, self._groups,
                        self._segment_id_map, self._seg_counter)

    print(
        f"  SensorDataset [{self.parquet_dir.name}]: "
        f"{len(self._index)} windows, "
        f"{sum(1 for e in self._index if e[4] >= 0)} with audio, "
        f"{sum(1 for e in self._index if e[5] >= 0)} with seismic"
    )
```

- [ ] **Step 4: Run all cache tests to verify they pass**

```bash
poetry run pytest tests/data/test_disk_cache.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add crl_vehicle/data/dataset.py tests/data/test_disk_cache.py
git commit -m "feat: wire disk cache into SensorDataset.__init__"
```

---

### Task 5: Expose `cache_dir` through `train.py`

**Files:**
- Modify: `train.py` — add `--cache-dir` CLI flag, pass to `build_crl_loaders` and `build_downstream_loaders`
- Modify: `crl_vehicle/data/dataset.py` — `build_crl_loaders` and `build_downstream_loaders` are in `train.py`, not dataset.py; pass `cache_dir` through to `SensorDataset`

- [ ] **Step 1: Add `--cache-dir` argument to `parse_args()` in `train.py`**

```python
p.add_argument("--cache-dir", default=None,
               help="Directory for pre-built dataset cache. Skips parquet parse on hit.")
```

- [ ] **Step 2: Pass `cache_dir` into `build_crl_loaders` and `build_downstream_loaders`**

Update `build_crl_loaders` signature and body:

```python
def build_crl_loaders(
    data_dir: str,
    val_dir: str,
    config: CRLConfig,
    cache_dir: Path | None = None,
) -> tuple[DataLoader, DataLoader]:
    train_ds = StratifiedPairDataset(SensorDataset(data_dir, config, is_train=True,  cache_dir=cache_dir))
    val_ds   = StratifiedPairDataset(SensorDataset(val_dir,  config, is_train=False, cache_dir=cache_dir))
    # ... rest unchanged
```

Update `build_downstream_loaders` signature and body:

```python
def build_downstream_loaders(
    data_dir: str,
    val_dir: str,
    config: CRLConfig,
    cache_dir: Path | None = None,
) -> tuple[DataLoader, DataLoader]:
    train_ds = SensorDataset(data_dir, config, is_train=True,  cache_dir=cache_dir)
    val_ds   = SensorDataset(val_dir,  config, is_train=False, cache_dir=cache_dir)
    # ... rest unchanged
```

- [ ] **Step 3: Wire `cache_dir` in `main()`**

```python
cache_dir = Path(args.cache_dir) if args.cache_dir else None

# In the crl phase block:
crl_train_loader, crl_val_loader = build_crl_loaders(
    args.data_dir, args.val_dir, cfg, cache_dir=cache_dir)

# In the downstream phase block:
ds_train_loader, ds_val_loader = build_downstream_loaders(
    args.data_dir, args.val_dir, cfg, cache_dir=cache_dir)
```

- [ ] **Step 4: Smoke test — verify train.py still parses args without error**

```bash
poetry run python train.py --help
```

Expected: help text shows `--cache-dir` option, no errors.

- [ ] **Step 5: Commit**

```bash
git add train.py
git commit -m "feat: add --cache-dir flag to train.py for dataset disk cache"
```

---

### Task 6: End-to-end validation with smoke test data

**Files:**
- Read: `smoke_test.py` — understand what data it uses
- Test: manual run

- [ ] **Step 1: Check smoke test data exists**

```bash
ls saved_crl/smoke_test/
poetry run python smoke_test.py --help 2>&1 | head -5
```

- [ ] **Step 2: Run smoke test with cache enabled (first run — cache miss)**

```bash
poetry run python train.py \
    --phase crl \
    --data-dir ./saved_crl/smoke_test \
    --cache-dir /tmp/crl_cache \
    --crl-epochs 1 \
    --steps-per-epoch 2 \
    2>&1 | tee /tmp/cache_first_run.log
```

Expected output includes: `saving disk cache` and training completes without error. Check `/tmp/crl_cache/` has subdirectory with `.npy` files.

- [ ] **Step 3: Run smoke test again (cache hit)**

```bash
poetry run python train.py \
    --phase crl \
    --data-dir ./saved_crl/smoke_test \
    --cache-dir /tmp/crl_cache \
    --crl-epochs 1 \
    --steps-per-epoch 2 \
    2>&1 | tee /tmp/cache_second_run.log
```

Expected output includes: `loaded from disk cache` — no parquet parsing.

- [ ] **Step 4: Commit**

```bash
git add .
git commit -m "feat: dataset disk cache complete and validated"
```

---

## Self-Review

**Spec coverage:**
- ✅ Hash-based cache key (Task 1)
- ✅ Save on first run (Task 2)
- ✅ mmap load on subsequent runs (Task 3)
- ✅ Wired into `SensorDataset.__init__` (Task 4)
- ✅ Exposed via CLI (Task 5)
- ✅ End-to-end validation (Task 6)
- ✅ Worker safety: `_get_window` calls `.copy()` on every window slice — mmap read-only arrays are safe across forked workers

**Placeholder scan:** None found.

**Type consistency:**
- `_save_cache` / `_load_cache` use the same key tuple `(stem, seg_key)` throughout
- `data_view` shape `(1, L)` matches what `_get_window` expects (`entry["data"][:, start:start+win_len]`)
- `cache_dir: Path | None` is consistent across all call sites
