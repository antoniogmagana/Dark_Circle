# In-Distribution Split Schema (`--use-id-split`)

**Date:** 2026-04-25
**Status:** Design — pending implementation plan

## Motivation

The current pipeline routes parquet files to train/val/test by physical
directory placement under `data_files/parsed/{train,val,test}/`. Held-out
recordings (different sessions, sometimes different sensors or geometries)
sit in val and test, so the resulting metric is a cross-recording
generalization (OOD) score. That OOD signal is what the capstone targets,
but it leaves an open question: when the model underperforms, is it failing
to generalize, or has it failed to learn even an in-distribution mapping?

This spec adds an opt-in **in-distribution (ID) split schema** that derives
val/test from the *same* recordings the model trained on, halving each
"split" recording in time. It serves as a minimum-viable-correlation floor:
if the model cannot achieve healthy metrics on ID, the OOD result is
uninterpretable.

## Goals & Non-Goals

**Goals**

- Add a single CLI flag (`--use-id-split`) that switches the loader's
  split-routing source from on-disk directory layout to the
  `DATASET_VEHICLE_MAP` in `crl_vehicle/config.py`.
- Support two new markers in `DATASET_VEHICLE_MAP`: `"split"` (whole-window
  half/half by time) and `"split_runs"` (run-aware ~50/25/25 partition).
- Keep audio/seismic windows perfectly paired through every step.
- Persist a small manifest cache under `saved_crl/id_cache/` so the
  per-file routing only needs to be computed once per `(mapping, sources)`
  state.

**Non-goals**

- No changes to model, loss, trainer, or `StratifiedPairDataset`
  partner-sampling logic.
- No changes to the existing `.pt` shared-memory window cache under
  `saved_crl/cache/`. Raw window data is reused across both schemas.
- No removal or migration of the existing OOD schema. ID is opt-in;
  default behavior is unchanged.
- No new physical data layout on disk (no `parsed_id/` directories).

## Conceptual Model

Under the OOD (default) schema, the loader takes one parquet directory
(`--data-dir` for train, `--val-dir` for val) and ingests every file in
it. The marker triple in `DATASET_VEHICLE_MAP[ds][vehicle]` supplies
`(category, model, split_label)` — but the `split_label` is informational
only; the *directory* the file lives in determines whether that file
contributes to train, val, or test.

Under the ID schema, the source-of-truth flips. The loader scans **all
three** of `parsed/{train,val,test}/`, dedupes by file stem, and uses the
mapping's `split_label` (or `"split"` / `"split_runs"`) to decide which
windows from that file go to which role. Directory placement is ignored.

## Marker Semantics (under `--use-id-split`)

| Marker | Effect |
|---|---|
| `"train"` | All windows of the file go to **train**. |
| `"val"` | All windows of the file go to **val**. |
| `"test"` | All windows of the file go to **test**. |
| `"split"` | Windows `[0, N//2)` → **val**; windows `[N//2, N)` → **test**. File contributes nothing to train. |
| `"split_runs"` | Per-run partition (see "Run-Aware Splitting" below). |
| no `(category, model, split)` triple — m3nvc `"background"` entry | Routes to **train**. (Background is not a vehicle category and isn't meaningful as an ID eval target.) |

`N` is the **paired** window count: `N = min(audio_n_windows, seismic_n_windows)`
when both sensors exist for the group, otherwise the single available
sensor's count. This matches the existing rule in
`SensorDataset._build_from_parquet`.

A group whose paired window count is 0 — or where one sensor file is
missing entirely — is dropped exactly as today.

## Pairing Rule (Critical Invariant)

Audio and seismic must be sample-paired at the window level. Every split
decision is made against the **paired** window range, never against a
sensor-specific count.

- For `"split"`: val = `[0, N_pair // 2)`, test = `[N_pair // 2, N_pair)`.
  Both sensors use the same window indices.
- For `"split_runs"`: per-run window ranges are computed separately for
  audio and seismic from each parquet's `(scene_id, run_id)` columns,
  then **intersected**: `paired_range = [max(start_a, start_s), min(end_a, end_s))`.
  - If the paired range is empty, the run is dropped from the partition
    entirely.
  - If a run appears in only one sensor's parquet, the run is dropped.
  - Dropped runs are logged with the run key and the reason
    (`empty_intersection` or `single_sensor`).

The 50/25/25 greedy operates only on the surviving paired runs.

## Run-Aware Splitting (`"split_runs"`)

m3nvc parquet files contain `scene_id` and `run_id` columns. A "run" is
the contiguous block of samples sharing a `(scene_id, run_id)` key. Per
on-disk inspection, these blocks are perfectly contiguous — no
interleaving — so detection is a single pass.

**Algorithm (per `"split_runs"` file pair):**

1. Read `scene_id`, `run_id` columns from the audio parquet. Verify each
   `(scene, run)` key occupies one contiguous sample range; abort
   manifest build with an explicit error if not (so silent reordering is
   impossible).
2. Repeat for the seismic parquet.
3. Convert each run's sample range to a window range using the sensor's
   window size. Use `start_window = ceil(start_sample / W)` and
   `end_window = floor(end_sample / W)` so any windows that straddle a
   run boundary are excluded — runs are clean at the window level, no
   leakage.
4. For each `(scene, run)` key present in both sensors, compute the
   paired window range
   `[max(w_start_a, w_start_s), min(w_end_a, w_end_s))`.
   Drop runs that fail the pairing test (logged at WARNING).
5. Sort surviving runs deterministically by `(scene_id, run_id)` to
   guarantee reproducibility.
6. **Greedy 50/25/25 partition over paired window counts.** Process runs
   in descending paired-window-count order; assign each to whichever
   bucket (train, val, test) is currently furthest below its target
   ratio. Break ties in favor of train, then val.
7. **Floor:** if there are ≥3 surviving runs, val and test must each
   receive at least one run. The greedy already produces this in
   practice; if not, swap the smallest train run with whichever bucket
   is empty.
8. Record per-file `{run_key: split}` in the manifest along with the
   paired window ranges.

Per-run window counts are typically tens to hundreds (the sample file
inspected had 8 runs of 60–200 s each), so the greedy is cheap and the
50/25/25 ratio is easy to approximate well.

## Cache Format

**Location:** `saved_crl/id_cache/`

**Layout:** one JSON manifest per cache state.

```
saved_crl/id_cache/
  manifest_<hash>.json
```

`<hash>` is the sha256 of:

- The `DATASET_VEHICLE_MAP` dict, serialized canonically (sorted keys).
- `cfg.modality_cfg("audio").window_size`,
  `cfg.modality_cfg("seismic").window_size`.
- A sorted list of `(stem, source_parquet_mtime_ns)` tuples for every
  `"split"` and `"split_runs"` parquet that contributed to the manifest.

This makes the cache self-invalidating: any change to the mapping, to
window sizes, or to the underlying parquet files produces a new hash and
forces a recompute. (`"train"`/`"val"`/`"test"` files don't go into the
hash because their routing is trivial — no computation to cache.)

**Manifest schema:**

The manifest is keyed by **group** (`(dataset, vehicle, rs_node)`),
not by per-sensor stem. Because `split_assignments` are in *paired
window* coordinates, audio and seismic share the same intervals — so
storing them per-group eliminates duplication and the possibility of
audio/seismic drifting out of sync from a partial edit. This matches
the existing `gkey` structure in `_build_from_parquet`.

```json
{
  "schema_version": 1,
  "created_unix": 1745571834,
  "config_window_sizes": {"audio": 16000, "seismic": 200},
  "groups": {
    "m3nvc__cx30__rs1": {
      "dataset": "m3nvc",
      "vehicle": "cx30",
      "rs_node": "rs1",
      "marker": "split_runs",
      "split_assignments": {
        "train": [[0, 181], [453, 634]],
        "val":   [[181, 245]],
        "test":  [[245, 365]]
      },
      "run_meta": {
        "1_6": {"split": "train", "n_windows_paired": 181},
        "1_7": {"split": "val",   "n_windows_paired": 60}
      },
      "dropped_runs": [
        {"run_key": "3_2", "reason": "single_sensor"}
      ]
    },
    "iobt__silverado0315pm__rs1": {
      "dataset": "iobt",
      "vehicle": "silverado0315pm",
      "rs_node": "rs1",
      "marker": "split",
      "split_assignments": {
        "val":  [[0, 90]],
        "test": [[90, 180]]
      }
    }
  }
}
```

Group key format: `"{dataset}__{vehicle}__{rs_node}"` (double-underscore
separator since vehicle names may contain single underscores).
`split_assignments` are lists of `[w_start, w_end)` half-open intervals
in **paired window** coordinates and apply to both audio and seismic
of the pair. The dataset enumerates only the windows whose index falls
inside one of the intervals for the active role. Dropped runs are
recorded in `dropped_runs` for post-hoc inspection in addition to the
WARN log emitted at build time.

## `SensorDataset` API Changes

**New constructor signature** (additions only — existing args unchanged):

```python
SensorDataset(
    parquet_dir: str | Path,        # OOD path; ignored when use_id_split=True
    config: CRLConfig,
    is_train: bool = True,
    cache_dir: Path | None = None,  # raw .pt shared-mem cache (unchanged)
    *,
    use_id_split: bool = False,
    role: str = "train",            # "train" | "val" | "test", required when use_id_split
    id_root: str | Path | None = None,   # parent of train/val/test/ — required when use_id_split
    id_cache_dir: Path | None = None,
)
```

When `use_id_split=False`: behavior is byte-identical to today.
`role`, `id_root`, and `id_cache_dir` are unused.

When `use_id_split=True`:

1. `parquet_dir` is ignored (a WARNING is emitted if it points anywhere
   meaningful, to avoid silent confusion).
2. The loader globs `id_root / "*" / "*.parquet"`, deduplicates by
   stem, and ignores which subdir each file lives in.
3. The loader builds (or loads from `id_cache_dir`) the manifest for
   every `"split"` / `"split_runs"` file required by the active `role`.
4. Inside `_build_from_parquet`, after the existing `n_windows = min(...)`
   line, the routing logic checks the file's marker:
   - `"train"` / `"val"` / `"test"`: include all `n_windows` iff the
     marker matches `role`; else skip the group.
   - `"split"` / `"split_runs"`: include only the windows whose index
     falls in any `[w_start, w_end)` interval listed under
     `manifest.groups[group_key].split_assignments[role]`, where
     `group_key = f"{dataset}__{vehicle}__{rs_node}"`.
5. The `_index` list is populated identically to today otherwise. The
   downstream `_preload_shared`, `__getitem__`, presence-vector logic,
   and `StratifiedPairDataset` are all unchanged — they operate on the
   same `(gkey, w, ...)` index entries.

A group is dropped (logged at INFO) if its paired-window count for the
active role is zero.

## `train.py` Changes

- Add `--use-id-split` boolean flag (default False).
- Add `--id-root` argument (default `../data_files/parsed/`) — the
  parent directory containing `train/`, `val/`, and `test/`. Used only
  when `--use-id-split` is set.
- When `--use-id-split` is set:
  - Construct
    `train_ds = SensorDataset(parquet_dir=args.data_dir, config=cfg,
        is_train=True, cache_dir=cache_dir, use_id_split=True,
        role="train", id_root=args.id_root,
        id_cache_dir=Path("saved_crl/id_cache"))`.
  - Construct `val_ds` likewise with `is_train=False, role="val"`.
  - `--data-dir` and `--val-dir` are ignored (warn at startup).
- When unset: zero behavior change.

`compute_class_weights` operates on `_index` and is automatically correct
under the new schema.

## Failure Modes & Handling

| Condition | Handling |
|---|---|
| Manifest hash present, file readable | Load and use. |
| Manifest hash present, file missing or corrupt | Recompute, overwrite atomically. |
| `(scene, run)` block non-contiguous in a `"split_runs"` parquet | Abort manifest build with explicit error naming the file and key. (Silent ordering bugs are far worse than a hard stop.) |
| `"split_runs"` parquet has no `scene_id` or `run_id` column | Abort manifest build with explicit error. |
| `"split_runs"` run appears in only one sensor | Drop the run, WARN with `(stem_pair, run_key, reason="single_sensor")`. |
| `"split_runs"` paired window range empty | Drop the run, WARN with reason `"empty_intersection"`. |
| Group has 0 paired windows for the active role | Skip group, INFO log. |
| Surviving run count < 3 for a `"split_runs"` file | Apply 50/25/25 best-effort: 1 run → train; 2 runs → train+val; ≥3 runs → enforce val and test each receive ≥1. |
| Marker is `"split"` and `N_pair < 2` | Skip the group, INFO log. (Cannot split a 1-window file in half.) |

## Testing

The implementation plan should add unit tests covering:

- `"split"` marker: exact half/half on even and odd `N_pair`.
- `"split_runs"` marker: 50/25/25 ratio achieved within tolerance on a
  synthetic file with N runs of varied size.
- Pairing intersection: synthetic case where one sensor has a run the
  other lacks → run is dropped, WARN fires.
- Manifest cache hit / miss: changing the mapping, window size, or
  source mtime changes the hash; identical inputs hit cache.
- Identity test: with `--use-id-split` off, `SensorDataset` behavior
  matches today's exactly (golden index comparison on a small fixture).

End-to-end smoke: `train.py --use-id-split --epochs 1 --max-batches 4`
on the existing fixture data, asserting val and test sets are non-empty
and disjoint.

## Out of Scope (Future Work)

- A `role="test"` test-only loader path is straightforward to add to
  `eval.py` once this schema lands; not part of this spec.
- Mixing markers (e.g., one file split by run, another by half) within a
  single dataset is already supported by the design — the manifest is
  per-file — but no eval comparison across mixtures is planned here.
- The existing `eval.py --include-datasets` flag composes naturally with
  `--use-id-split` (intersection of constraints) and needs no change.

## Files Touched

- `crl-train/crl_vehicle/config.py` — add `use_id_split: bool = False`
  to `CRLConfig` (cosmetic; CLI flag is the real switch).
- `crl-train/crl_vehicle/data/dataset.py` — new manifest module + branch
  in `_build_from_parquet`.
- `crl-train/train.py` — add CLI flag + dataset wiring.
- `crl-train/saved_crl/id_cache/` — new cache directory (created on
  first use).
- Tests under `crl-train/tests/` — new unit + smoke tests.
