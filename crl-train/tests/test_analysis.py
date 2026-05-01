"""Tests for crl_vehicle/analysis.py — the shared loader used by all
comparison and plotting scripts. Uses tmp_path fixtures rather than real
saved_crl data so the tests run without training artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from crl_vehicle import analysis as A

# --------------------------------------------------------------------------
# Fixture helpers
# --------------------------------------------------------------------------


def _write_crl_metrics(crl_dir: Path, rows: list[dict]) -> None:
    """Write a crl_metrics.csv with the given rows. Columns inferred from
    rows[0]."""
    crl_dir.mkdir(parents=True, exist_ok=True)
    with open(crl_dir / "crl_metrics.csv", "w", newline="") as f:
        if not rows:
            return
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _write_meta(
    crl_dir: Path,
    frontend: str,
    *,
    morlet_use_phase: bool = True,
    prior_type: str = "standard",
    training_mode: str = "vae",
    stage2: bool = False,
    d_model: int = 64,
    morlet_learnable_w0: bool = False,
    sensors: list[str] | None = None,
) -> None:
    crl_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "frontend_type": frontend,
        "morlet_use_phase": morlet_use_phase,
        "prior_type": prior_type,
        "training_mode": training_mode,
        "d_model": d_model,
        "morlet_learnable_w0": morlet_learnable_w0,
    }
    meta = {
        "config": cfg,
        "sensors": sensors or ["audio", "seismic"],
        "stage2": stage2,
    }
    (crl_dir / "meta.json").write_text(json.dumps(meta))


def _write_summary(crl_dir: Path, *, best_elbo: float = 1.0, best_type_f1: float = 0.7) -> None:
    crl_dir.mkdir(parents=True, exist_ok=True)
    (crl_dir / "crl_checkpoint_summary.json").write_text(
        json.dumps(
            {
                "best_ref_elbo": best_elbo,
                "best_aux_type_f1": best_type_f1,
                "best_aux_type_epoch": 42,
            }
        )
    )


def _write_downstream(
    run_dir: Path,
    probe: str = A.CANONICAL_PROBE,
    *,
    best_pres_f1: float = 0.8,
    best_type_f1: float = 0.6,
) -> None:
    probe_dir = run_dir / "downstream" / probe
    probe_dir.mkdir(parents=True, exist_ok=True)
    with open(probe_dir / "downstream_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
                "val_pres_f1",
                "val_pres_acc",
                "val_type_f1",
                "val_type_acc",
            ],
        )
        w.writeheader()
        # Two epochs with best values on epoch 1.
        w.writerow(
            {
                "epoch": 0,
                "train_loss": 1.0,
                "val_loss": 0.9,
                "val_pres_f1": 0.5,
                "val_pres_acc": 0.6,
                "val_type_f1": 0.4,
                "val_type_acc": 0.5,
            }
        )
        w.writerow(
            {
                "epoch": 1,
                "train_loss": 0.5,
                "val_loss": 0.4,
                "val_pres_f1": best_pres_f1,
                "val_pres_acc": 0.9,
                "val_type_f1": best_type_f1,
                "val_type_acc": 0.7,
            }
        )


def _write_eval(
    run_dir: Path,
    probe: str = A.CANONICAL_PROBE,
    *,
    per_dataset: dict[str, float] | None = None,
) -> None:
    per = per_dataset or {"iobt": 0.65, "focal": 0.45, "m3nvc": 0.55}
    for split, f1 in per.items():
        d = run_dir / "eval" / probe / split
        d.mkdir(parents=True, exist_ok=True)
        (d / "eval_report.json").write_text(
            json.dumps(
                {
                    "split": split,
                    "presence": {"f1": f1 + 0.1, "balanced_accuracy": 0.7, "mcc": 0.4},
                    "type": {"macro_f1_support_only": f1, "macro_f1": f1 - 0.05},
                }
            )
        )


def _build_fake_run(
    root: Path,
    name: str,
    frontend: str = "morlet_per_sensor",
    **meta_overrides,
) -> Path:
    """One-stop fixture: writes meta + crl_metrics + summary + downstream + eval."""
    run_dir = root / name
    crl_dir = run_dir / "crl"
    _write_meta(crl_dir, frontend, **meta_overrides)
    _write_summary(crl_dir)
    _write_crl_metrics(
        crl_dir,
        [
            {
                "epoch": 0,
                "beta": 0.02,
                "beta_event": "↑",
                "val_recon": 2.0,
                "val_raw_kl": 0.5,
                "val_ref_elbo": 2.5,
                "val_aux_type_f1": 0.3,
            },
            {
                "epoch": 1,
                "beta": 0.04,
                "beta_event": "↑",
                "val_recon": 1.5,
                "val_raw_kl": 0.6,
                "val_ref_elbo": 2.1,
                "val_aux_type_f1": 0.5,
            },
        ],
    )
    _write_downstream(run_dir)
    _write_eval(run_dir)
    return run_dir


# --------------------------------------------------------------------------
# discover_runs
# --------------------------------------------------------------------------


class TestDiscoverRuns:
    def test_finds_all_runs_with_crl_meta(self, tmp_path):
        _build_fake_run(tmp_path, "a")
        _build_fake_run(tmp_path, "b")
        (tmp_path / "not-a-run").mkdir()  # no crl/meta.json
        runs = A.discover_runs(tmp_path)
        assert sorted(r.name for r in runs) == ["a", "b"]

    def test_empty_root(self, tmp_path):
        assert A.discover_runs(tmp_path) == []

    def test_missing_root_returns_empty(self, tmp_path):
        assert A.discover_runs(tmp_path / "nope") == []


# --------------------------------------------------------------------------
# load_run_metrics
# --------------------------------------------------------------------------


class TestLoadRunMetrics:
    def test_extracts_config_and_sensors(self, tmp_path):
        run = _build_fake_run(tmp_path, "r1", frontend="multiscale")
        rm = A.load_run_metrics(run)
        assert rm.name == "r1"
        assert rm.config["frontend_type"] == "multiscale"
        assert rm.sensors == ["audio", "seismic"]

    def test_picks_up_summary_json(self, tmp_path):
        run = _build_fake_run(tmp_path, "r1")
        rm = A.load_run_metrics(run)
        assert rm.best_val_ref_elbo == 1.0
        assert rm.best_val_aux_type_f1 == 0.7

    def test_epochs_from_csv(self, tmp_path):
        run = _build_fake_run(tmp_path, "r1")
        rm = A.load_run_metrics(run)
        assert rm.epochs_completed == 2

    def test_downstream_best_extracted(self, tmp_path):
        run = _build_fake_run(tmp_path, "r1")
        rm = A.load_run_metrics(run)
        assert rm.best_pres_f1 == 0.8
        assert rm.best_type_f1 == 0.6
        assert rm.probe_mode_used == A.CANONICAL_PROBE

    def test_cross_location_minimum(self, tmp_path):
        run = _build_fake_run(tmp_path, "r1")
        rm = A.load_run_metrics(run)
        # _write_eval defaults to iobt=0.65, focal=0.45, m3nvc=0.55.
        assert rm.per_dataset_type_f1 == {"iobt": 0.65, "focal": 0.45, "m3nvc": 0.55}
        assert rm.min_dataset_type_f1 == 0.45
        assert rm.worst_dataset == "focal"

    def test_diverged_flag(self, tmp_path):
        run = _build_fake_run(tmp_path, "r1")
        # Override summary to be diverged.
        (run / "crl" / "crl_checkpoint_summary.json").write_text(
            json.dumps(
                {
                    "best_ref_elbo": 700.0,
                    "best_aux_type_f1": 0.4,
                }
            )
        )
        rm = A.load_run_metrics(run)
        assert rm.diverged is True

    def test_shippable_flag(self, tmp_path):
        run = _build_fake_run(tmp_path, "r1")
        _write_downstream(run, best_pres_f1=0.88, best_type_f1=0.72)
        rm = A.load_run_metrics(run)
        assert rm.shippable is True

    def test_not_shippable_when_pres_too_low(self, tmp_path):
        run = _build_fake_run(tmp_path, "r1")
        _write_downstream(run, best_pres_f1=0.80, best_type_f1=0.72)
        rm = A.load_run_metrics(run)
        assert rm.shippable is False

    def test_tolerates_missing_eval_dir(self, tmp_path):
        run = _build_fake_run(tmp_path, "r1")
        # Remove eval artifacts entirely.
        import shutil

        shutil.rmtree(run / "eval")
        rm = A.load_run_metrics(run)
        # Must not crash; cross-location fields just stay empty.
        assert rm.per_dataset_type_f1 == {}
        assert rm.min_dataset_type_f1 is None

    def test_stage2_attribution(self, tmp_path):
        run = _build_fake_run(tmp_path, "r1", stage2=True)
        (run / "crl" / "meta.json").write_text(
            json.dumps(
                {
                    "config": json.loads((run / "crl" / "meta.json").read_text())["config"],
                    "sensors": ["audio", "seismic"],
                    "stage2": True,
                    "init_from_run": "/path/to/source",
                }
            )
        )
        rm = A.load_run_metrics(run)
        assert rm.stage2 is True
        assert rm.init_from_run == "/path/to/source"


# --------------------------------------------------------------------------
# Time series loaders
# --------------------------------------------------------------------------


class TestTimeseries:
    def test_crl_timeseries_drops_non_numeric(self, tmp_path):
        run = _build_fake_run(tmp_path, "r1")
        ts = A.load_crl_timeseries(run)
        # beta_event column is string → must be dropped.
        assert "val_recon" in ts
        assert all(isinstance(v, float) for v in ts["val_recon"])
        assert "beta_event" in ts
        assert ts["beta_event"] == []  # strings stripped

    def test_morlet_freq_history_none_for_non_learnable(self, tmp_path):
        run = _build_fake_run(tmp_path, "r1", frontend="morlet_per_sensor")
        assert A.load_morlet_freq_history(run) is None

    def test_morlet_freq_history_parses_csv(self, tmp_path):
        run = _build_fake_run(tmp_path, "r1", frontend="morlet_learnable")
        # Write a synthetic learnable_morlet_freqs.csv.
        path = run / "crl" / "learnable_morlet_freqs.csv"
        path.write_text(
            "epoch,sensor,filter_idx,freq_hz,w0\n"
            "0,audio,0,20.0,6.0\n"
            "1,audio,0,20.5,6.0\n"
            "0,audio,1,40.0,6.0\n"
            "1,audio,1,40.5,6.0\n"
            "0,seismic,0,2.0,6.0\n"
            "1,seismic,0,2.1,6.0\n"
        )
        history = A.load_morlet_freq_history(run)
        assert set(history.keys()) == {"audio", "seismic"}
        assert history["audio"][0] == [20.0, 20.5]
        assert history["audio"][1] == [40.0, 40.5]
        assert history["seismic"][0] == [2.0, 2.1]


# --------------------------------------------------------------------------
# Filtering
# --------------------------------------------------------------------------


class TestFilters:
    def test_parse_filter_args_type_coerce(self):
        parsed = A.parse_filter_args(
            [
                "morlet_use_phase=true",
                "d_model=64",
                "lr=3e-4",
                "frontend_type=multiscale",
            ]
        )
        assert parsed == {
            "morlet_use_phase": True,
            "d_model": 64,
            "lr": 3e-4,
            "frontend_type": "multiscale",
        }

    def test_parse_filter_args_malformed_raises(self):
        with pytest.raises(ValueError, match="expects key=value"):
            A.parse_filter_args(["no-equals"])

    def test_apply_filters_matches_config_field(self, tmp_path):
        r1 = _build_fake_run(tmp_path, "a", frontend="multiscale")
        r2 = _build_fake_run(tmp_path, "b", frontend="morlet_per_sensor")
        runs = [A.load_run_metrics(r1), A.load_run_metrics(r2)]
        filtered = A.apply_filters(runs, {"frontend_type": "multiscale"})
        assert [r.name for r in filtered] == ["a"]

    def test_apply_filters_matches_stage2(self, tmp_path):
        r1 = _build_fake_run(tmp_path, "a", stage2=False)
        r2 = _build_fake_run(tmp_path, "b", stage2=True)
        (r2 / "crl" / "meta.json").write_text(
            json.dumps(
                {
                    "config": json.loads((r2 / "crl" / "meta.json").read_text())["config"],
                    "sensors": ["audio", "seismic"],
                    "stage2": True,
                }
            )
        )
        runs = [A.load_run_metrics(r1), A.load_run_metrics(r2)]
        filtered = A.apply_filters(runs, {"stage2": True})
        assert [r.name for r in filtered] == ["b"]

    def test_exclude_diverged_default(self, tmp_path):
        r1 = _build_fake_run(tmp_path, "a")
        r2 = _build_fake_run(tmp_path, "b")
        # Force r2 to be diverged.
        (r2 / "crl" / "crl_checkpoint_summary.json").write_text(
            json.dumps(
                {
                    "best_ref_elbo": 800.0,
                    "best_aux_type_f1": 0.3,
                }
            )
        )
        runs = [A.load_run_metrics(r1), A.load_run_metrics(r2)]
        filtered = A.apply_filters(runs, {}, exclude_diverged=True)
        assert [r.name for r in filtered] == ["a"]

    def test_include_diverged_when_flag_off(self, tmp_path):
        r1 = _build_fake_run(tmp_path, "a")
        (r1 / "crl" / "crl_checkpoint_summary.json").write_text(
            json.dumps(
                {
                    "best_ref_elbo": 800.0,
                    "best_aux_type_f1": 0.3,
                }
            )
        )
        runs = [A.load_run_metrics(r1)]
        filtered = A.apply_filters(runs, {}, exclude_diverged=False)
        assert len(filtered) == 1


# --------------------------------------------------------------------------
# Axis signatures
# --------------------------------------------------------------------------


class TestAxisSignature:
    def test_all_axes_present(self, tmp_path):
        run = _build_fake_run(tmp_path, "r1")
        rm = A.load_run_metrics(run)
        sig = A.axis_signature(rm)
        assert set(sig.keys()) == set(A.ABLATION_AXES)

    def test_stage2_axis_value(self, tmp_path):
        run = _build_fake_run(tmp_path, "r1", stage2=True)
        (run / "crl" / "meta.json").write_text(
            json.dumps(
                {
                    "config": json.loads((run / "crl" / "meta.json").read_text())["config"],
                    "sensors": ["audio", "seismic"],
                    "stage2": True,
                }
            )
        )
        rm = A.load_run_metrics(run)
        sig = A.axis_signature(rm)
        assert sig["stage2"] is True

    def test_frontend_family_grouping(self, tmp_path):
        r1 = _build_fake_run(tmp_path, "a", frontend="multiscale")
        r2 = _build_fake_run(tmp_path, "b", frontend="morlet_per_sensor")
        r3 = _build_fake_run(tmp_path, "c", frontend="morlet_learnable_fused")
        families = [A.frontend_family(A.load_run_metrics(r)) for r in (r1, r2, r3)]
        assert families == ["multiscale", "morlet", "morlet_learnable"]
