"""Post-hoc analysis helpers for completed CRL runs.

Used by compare_runs.py, compare_ablations.py, compare_cross_location.py,
plot_run.py, plot_aggregate.py. All functions are read-only — they never
modify run dirs or write artifacts. Scripts that own output do their own
writing.

Run directory layout (what's expected on disk):

    <run>/
    ├── crl/
    │   ├── meta.json                    # config + sensors
    │   ├── crl_metrics.csv              # per-epoch training curves
    │   ├── crl_checkpoint_summary.json  # best ELBO, best aux_type_f1
    │   ├── learnable_morlet_freqs.csv   # only for learnable morlet runs
    │   └── crl_best.pth / crl_best_aux_type.pth / crl_final.pth
    ├── downstream/<probe_mode>__<ckpt>/
    │   ├── downstream_metrics.csv       # per-epoch probe training
    │   └── meta.json
    └── eval/<probe_mode>__<ckpt>/<split>/
        └── eval_report.json             # presence + type metrics per split

`<split>` is one of {iobt, focal, m3nvc, full} — identifies which dataset
the eval ran on.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

# The probe config treated as "canonical" for the leaderboard. Chosen because
# it's the minimal-capacity probe on the ELBO-selected checkpoint — everything
# else is a variation.
CANONICAL_PROBE = "linear_ztype__crl_best"

# val_ref_elbo threshold above which a run is considered diverged. Based on
# empirical observation: healthy morlet runs land in [0.2, 5], divergent ones
# shoot past 500. 50 is a comfortable margin.
DIVERGED_ELBO_THRESHOLD = 50.0

# Shippability thresholds from the capstone deliverable.
SHIP_PRES_F1 = 0.85
SHIP_TYPE_F1 = 0.70

# Frontend → color family (for plotting). Ordered so related variants get
# adjacent hues.
FRONTEND_COLORS = {
    "multiscale":              "#1f77b4",   # blue
    "morlet":                  "#ff7f0e",   # orange
    "morlet_per_sensor":       "#d62728",   # red
    "morlet_fused":            "#9467bd",   # purple
    "morlet_learnable":        "#2ca02c",   # green
    "morlet_learnable_fused":  "#8c564b",   # brown
}

_FRONTEND_FAMILY = {
    "multiscale":             "multiscale",
    "morlet":                 "morlet",
    "morlet_per_sensor":      "morlet",
    "morlet_fused":           "morlet",
    "morlet_learnable":       "morlet_learnable",
    "morlet_learnable_fused": "morlet_learnable",
}

# Hardcoded ablation axes. Each axis name MUST be a CRLConfig field (or
# meta-level field like "stage2"); the comparator looks up the value from
# RunMetrics.config (with stage2 as a special case pulled from the top-level
# meta.json).
ABLATION_AXES = (
    "frontend_type",
    "morlet_use_phase",
    "prior_type",
    "training_mode",
    "stage2",
    "morlet_learnable_w0",
)


# --------------------------------------------------------------------------
# Dataclass
# --------------------------------------------------------------------------

@dataclass
class RunMetrics:
    """One run's aggregated metrics. All fields optional — missing files
    become empty dicts / None, never hard failures."""

    name: str
    path: Path
    config: dict                          # cfg fields from meta.json
    sensors: list[str]

    # CRL training-time metrics (from crl_metrics.csv + checkpoint_summary.json)
    best_val_ref_elbo: float | None = None
    best_val_aux_type_f1: float | None = None
    best_aux_type_epoch: int | None = None
    final_val_recon: float | None = None
    final_val_raw_kl: float | None = None
    epochs_completed: int = 0

    # Downstream probe metrics (from canonical probe's downstream_metrics.csv)
    best_pres_f1: float | None = None
    best_type_f1: float | None = None
    best_pres_acc: float | None = None
    best_type_acc: float | None = None
    final_val_loss: float | None = None
    probe_mode_used: str | None = None    # which probe config was canonical

    # Cross-location eval (from eval/<probe>/<split>/eval_report.json)
    per_dataset_type_f1: dict[str, float] = field(default_factory=dict)
    per_dataset_pres_f1: dict[str, float] = field(default_factory=dict)
    min_dataset_type_f1: float | None = None
    worst_dataset: str | None = None

    # Calibrated metrics if available (balanced_acc, MCC) from canonical eval
    calibrated_type_f1: float | None = None
    balanced_accuracy: float | None = None
    mcc: float | None = None

    # Stage-2 attribution
    stage2: bool = False
    init_from_run: str | None = None

    # Derived
    diverged: bool = False
    shippable: bool = False


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------

def discover_runs(root: Path) -> list[Path]:
    """Find all <run>/ dirs under `root` that have crl/meta.json. Returns
    a list of run dir paths (not meta.json paths)."""
    root = Path(root)
    if not root.exists():
        return []
    runs = [p.parent.parent for p in root.glob("*/crl/meta.json")]
    return sorted(runs)


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------

def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _read_csv_columns(path: Path) -> dict[str, list]:
    """Read a CSV into a dict-of-lists with numeric coercion. Strings stay
    strings if they fail to parse. Returns empty dict for missing files."""
    if not path.exists():
        return {}
    with open(path) as f:
        reader = csv.DictReader(f)
        columns = {k: [] for k in (reader.fieldnames or [])}
        for row in reader:
            for k, v in row.items():
                try:
                    columns[k].append(float(v))
                except (TypeError, ValueError):
                    columns[k].append(v)
    return columns


def _pick_canonical_probe_dir(run_dir: Path, preferred: str = CANONICAL_PROBE) -> Path | None:
    """Find the downstream subdir for the canonical probe config. Falls
    back to any available probe if the preferred one is missing."""
    downstream = run_dir / "downstream"
    if not downstream.is_dir():
        return None
    pref_path = downstream / preferred
    if pref_path.is_dir() and (pref_path / "downstream_metrics.csv").exists():
        return pref_path
    # Fallback: first available probe config.
    for sub in sorted(downstream.iterdir()):
        if sub.is_dir() and (sub / "downstream_metrics.csv").exists():
            return sub
    return None


def _best_from_ds_metrics(cols: dict) -> dict:
    """Extract best pres_f1, type_f1, acc from downstream metrics CSV."""
    out = {}
    if not cols:
        return out
    for key, op in [
        ("val_pres_f1", max), ("val_pres_acc", max),
        ("val_type_f1", max), ("val_type_acc", max),
        ("val_loss",    min),
    ]:
        values = [v for v in cols.get(key, []) if isinstance(v, (int, float))]
        if values:
            out[key] = op(values)
    return out


def _cross_location_metrics(run_dir: Path, preferred: str = CANONICAL_PROBE) -> dict:
    """Walk eval/<canonical-or-fallback>/<split>/eval_report.json and collect
    per-dataset type_f1 + pres_f1. 'full' (all datasets pooled) is excluded
    from per-dataset — that's an aggregate, not a cross-location data point."""
    eval_dir = run_dir / "eval"
    if not eval_dir.is_dir():
        return {"per_dataset_type_f1": {}, "per_dataset_pres_f1": {}}

    probe_dir = eval_dir / preferred
    if not probe_dir.is_dir():
        # Fallback to any available probe.
        candidates = [p for p in sorted(eval_dir.iterdir()) if p.is_dir()]
        if not candidates:
            return {"per_dataset_type_f1": {}, "per_dataset_pres_f1": {}}
        probe_dir = candidates[0]

    per_type: dict[str, float] = {}
    per_pres: dict[str, float] = {}
    calibrated: dict = {}
    for split_dir in sorted(probe_dir.iterdir()):
        if not split_dir.is_dir():
            continue
        split = split_dir.name
        report = _read_json(split_dir / "eval_report.json")
        if not report:
            continue
        # type metric: prefer macro_f1_support_only (correct for missing classes).
        type_block = report.get("type", {})
        f1 = type_block.get("macro_f1_support_only") or type_block.get("macro_f1")
        if f1 is not None and split != "full":
            per_type[split] = float(f1)
        # presence metric
        pres_block = report.get("presence", {})
        pres_f1 = pres_block.get("f1")
        if pres_f1 is not None and split != "full":
            per_pres[split] = float(pres_f1)
        # calibrated metrics — read from 'full' if present.
        if split == "full":
            calibrated["pres_balanced_accuracy"] = pres_block.get("balanced_accuracy")
            calibrated["pres_mcc"] = pres_block.get("mcc")
            calibrated["type_macro_f1_support_only"] = type_block.get("macro_f1_support_only")

    out: dict = {
        "per_dataset_type_f1": per_type,
        "per_dataset_pres_f1": per_pres,
    }
    if calibrated:
        out["calibrated"] = calibrated
    return out


def load_run_metrics(run_dir: Path) -> RunMetrics:
    """Aggregate everything we know about one run into a RunMetrics."""
    run_dir = Path(run_dir)
    name = run_dir.name

    # meta.json is authoritative for config.
    meta = _read_json(run_dir / "crl" / "meta.json")
    if not meta:
        # Fallback: top-level meta.json from train.py's flat layout.
        meta = _read_json(run_dir / "meta.json")
    config = meta.get("config", {})
    sensors = meta.get("sensors", config.get("sensors", []))
    stage2 = bool(meta.get("stage2", False))
    init_from_run = meta.get("init_from_run")

    rm = RunMetrics(
        name=name, path=run_dir, config=config, sensors=list(sensors),
        stage2=stage2, init_from_run=init_from_run,
    )

    # CRL metrics — summary.json is easier than parsing the CSV.
    summary = _read_json(run_dir / "crl" / "crl_checkpoint_summary.json")
    if summary:
        rm.best_val_ref_elbo = summary.get("best_ref_elbo")
        rm.best_val_aux_type_f1 = summary.get("best_aux_type_f1")
        rm.best_aux_type_epoch = summary.get("best_aux_type_epoch")

    # Pull final-epoch values + epoch count from the CSV.
    crl_csv = _read_csv_columns(run_dir / "crl" / "crl_metrics.csv")
    if crl_csv:
        rm.epochs_completed = len(crl_csv.get("epoch", []))
        if crl_csv.get("val_recon"):
            rm.final_val_recon = _last_numeric(crl_csv["val_recon"])
        if crl_csv.get("val_raw_kl"):
            rm.final_val_raw_kl = _last_numeric(crl_csv["val_raw_kl"])
        # Summary may lack ref_elbo best (older runs); recompute from CSV.
        if rm.best_val_ref_elbo is None and crl_csv.get("val_ref_elbo"):
            vals = [v for v in crl_csv["val_ref_elbo"] if isinstance(v, (int, float))]
            if vals:
                rm.best_val_ref_elbo = min(vals)

    # Downstream probes.
    probe_dir = _pick_canonical_probe_dir(run_dir)
    if probe_dir is not None:
        rm.probe_mode_used = probe_dir.name
        ds_cols = _read_csv_columns(probe_dir / "downstream_metrics.csv")
        best = _best_from_ds_metrics(ds_cols)
        rm.best_pres_f1  = best.get("val_pres_f1")
        rm.best_pres_acc = best.get("val_pres_acc")
        rm.best_type_f1  = best.get("val_type_f1")
        rm.best_type_acc = best.get("val_type_acc")
        rm.final_val_loss = best.get("val_loss")

    # Cross-location eval.
    cl = _cross_location_metrics(run_dir)
    rm.per_dataset_type_f1 = cl.get("per_dataset_type_f1", {})
    rm.per_dataset_pres_f1 = cl.get("per_dataset_pres_f1", {})
    if rm.per_dataset_type_f1:
        worst = min(rm.per_dataset_type_f1.items(), key=lambda kv: kv[1])
        rm.worst_dataset = worst[0]
        rm.min_dataset_type_f1 = worst[1]
    calibrated = cl.get("calibrated") or {}
    rm.calibrated_type_f1 = calibrated.get("type_macro_f1_support_only")
    rm.balanced_accuracy = calibrated.get("pres_balanced_accuracy")
    rm.mcc = calibrated.get("pres_mcc")

    # Derived flags.
    if rm.best_val_ref_elbo is not None:
        rm.diverged = rm.best_val_ref_elbo > DIVERGED_ELBO_THRESHOLD
    if rm.best_pres_f1 is not None and rm.best_type_f1 is not None:
        rm.shippable = (
            rm.best_pres_f1 >= SHIP_PRES_F1
            and rm.best_type_f1 >= SHIP_TYPE_F1
        )

    return rm


def _last_numeric(seq: list) -> float | None:
    """Last numeric value in a list (some rows may have stringified NaN)."""
    for v in reversed(seq):
        if isinstance(v, (int, float)):
            return float(v)
    return None


# --------------------------------------------------------------------------
# Time series (for plotting)
# --------------------------------------------------------------------------

def load_crl_timeseries(run_dir: Path) -> dict[str, list[float]]:
    """Return per-epoch training curves from crl_metrics.csv. Non-numeric
    columns (like beta_event) are dropped. Empty dict on missing file."""
    cols = _read_csv_columns(Path(run_dir) / "crl" / "crl_metrics.csv")
    return {
        k: [v for v in vs if isinstance(v, (int, float))]
        for k, vs in cols.items()
    }


def load_downstream_timeseries(
    run_dir: Path, probe: str = CANONICAL_PROBE,
) -> dict[str, list[float]]:
    """Per-epoch downstream probe curves. Falls back to any available probe
    if the preferred one is absent."""
    probe_dir = _pick_canonical_probe_dir(Path(run_dir), preferred=probe)
    if probe_dir is None:
        return {}
    cols = _read_csv_columns(probe_dir / "downstream_metrics.csv")
    return {
        k: [v for v in vs if isinstance(v, (int, float))]
        for k, vs in cols.items()
    }


def load_morlet_freq_history(run_dir: Path) -> dict[str, dict[int, list[float]]] | None:
    """Parse learnable_morlet_freqs.csv into
        {sensor: {filter_idx: [freq_hz_per_epoch, ...]}}.
    Returns None for runs without the file (non-learnable variants)."""
    path = Path(run_dir) / "crl" / "learnable_morlet_freqs.csv"
    if not path.exists():
        return None
    # CSV schema (from trainer.py): epoch, sensor, filter_idx, freq_hz, w0
    history: dict[str, dict[int, list[float]]] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            sensor = row["sensor"]
            idx = int(row["filter_idx"])
            freq = float(row["freq_hz"])
            history.setdefault(sensor, {}).setdefault(idx, []).append(freq)
    return history if history else None


# --------------------------------------------------------------------------
# Filtering
# --------------------------------------------------------------------------

def _parse_filter_value(raw: str):
    """Try to interpret a CLI --filter value as bool/int/float/str."""
    if raw.lower() in ("true", "false"):
        return raw.lower() == "true"
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def parse_filter_args(pairs: list[str] | None) -> dict:
    """Convert ['key=val', ...] into a dict with type coercion."""
    if not pairs:
        return {}
    out: dict = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--filter expects key=value, got {pair!r}")
        k, v = pair.split("=", 1)
        out[k.strip()] = _parse_filter_value(v.strip())
    return out


def _lookup_run_value(rm: RunMetrics, key: str):
    """Find `key` in a RunMetrics by looking at top-level dataclass fields,
    config dict, or the meta.json-derived stage2 flag."""
    if key == "stage2":
        return rm.stage2
    if key in rm.config:
        return rm.config[key]
    return getattr(rm, key, None)


def apply_filters(
    runs: list[RunMetrics],
    filters: dict,
    exclude_diverged: bool = True,
) -> list[RunMetrics]:
    """Drop runs that don't match every key=val in `filters`, and optionally
    drop diverged runs. Missing keys on a run = no match (filtered out)."""
    out: list[RunMetrics] = []
    for rm in runs:
        if exclude_diverged and rm.diverged:
            continue
        skip = False
        for key, expected in filters.items():
            actual = _lookup_run_value(rm, key)
            if actual != expected:
                skip = True
                break
        if not skip:
            out.append(rm)
    return out


# --------------------------------------------------------------------------
# Ablation axis extraction
# --------------------------------------------------------------------------

def axis_signature(rm: RunMetrics) -> dict:
    """Return the {axis: value} dict for a run across all ABLATION_AXES.
    Missing axes (e.g., morlet_learnable_w0 on a pre-Checkpoint-4 run)
    take a sentinel None value so pairings are still meaningful."""
    return {axis: _lookup_run_value(rm, axis) for axis in ABLATION_AXES}


def frontend_family(rm: RunMetrics) -> str:
    """Coarse frontend group for plot coloring: multiscale / morlet /
    morlet_learnable."""
    return _FRONTEND_FAMILY.get(rm.config.get("frontend_type", ""), "unknown")


def frontend_color(rm: RunMetrics) -> str:
    return FRONTEND_COLORS.get(rm.config.get("frontend_type", ""), "#888888")
