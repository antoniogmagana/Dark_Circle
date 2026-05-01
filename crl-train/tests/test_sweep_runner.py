"""Tests for the YAML-driven sweep mode in run_experiments.py.

Covers the parts that don't require real training: YAML loading/validation,
argv construction, summary writing. Actual subprocess execution is out of
scope for unit tests — that's a smoke-test responsibility (Task #19).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

# run_experiments.py lives at the repo root, not inside a package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_experiments as runner

# ---------------------------------------------------------------------------
# load_sweep_yaml
# ---------------------------------------------------------------------------


class TestLoadSweepYaml:
    def _write(self, tmp_path: Path, content: str) -> Path:
        p = tmp_path / "sweep.yaml"
        p.write_text(content)
        return p

    def test_roundtrip_valid_yaml(self, tmp_path):
        p = self._write(
            tmp_path,
            """
base_config:
  n_epochs: 50
  lr: 0.001
runs:
  - name: a
    overrides: {frontend_type: multiscale}
  - name: b
    overrides: {frontend_type: morlet_learnable, morlet_learnable_w0: true}
""",
        )
        base, runs = runner.load_sweep_yaml(p)
        assert base == {"n_epochs": 50, "lr": 0.001}
        assert len(runs) == 2
        assert runs[0]["name"] == "a"
        assert runs[1]["overrides"]["morlet_learnable_w0"] is True

    def test_missing_runs_key_raises(self, tmp_path):
        p = self._write(tmp_path, "base_config: {n_epochs: 10}\n")
        with pytest.raises(ValueError, match="missing required 'runs'"):
            runner.load_sweep_yaml(p)

    def test_empty_runs_raises(self, tmp_path):
        p = self._write(tmp_path, "runs: []\n")
        with pytest.raises(ValueError, match="non-empty list"):
            runner.load_sweep_yaml(p)

    def test_base_config_optional(self, tmp_path):
        """Missing base_config is fine — empty dict default."""
        p = self._write(
            tmp_path,
            """
runs:
  - name: solo
    overrides: {frontend_type: multiscale}
""",
        )
        base, runs = runner.load_sweep_yaml(p)
        assert base == {}
        assert len(runs) == 1

    def test_run_missing_name_raises(self, tmp_path):
        p = self._write(
            tmp_path,
            """
runs:
  - overrides: {frontend_type: multiscale}
""",
        )
        with pytest.raises(ValueError, match="missing 'name'"):
            runner.load_sweep_yaml(p)

    def test_overrides_must_be_dict(self, tmp_path):
        p = self._write(
            tmp_path,
            """
runs:
  - name: bad
    overrides: not_a_dict
""",
        )
        with pytest.raises(ValueError, match="must be a dict"):
            runner.load_sweep_yaml(p)


# ---------------------------------------------------------------------------
# _build_train_argv
# ---------------------------------------------------------------------------


class TestBuildTrainArgv:
    def test_flagged_fields_go_to_named_flags(self, tmp_path):
        argv = runner._build_train_argv(
            python_exe="python",
            train_script=Path("train.py"),
            merged_cfg={"frontend_type": "multiscale", "lr": 0.001, "n_epochs": 50},
            run_save_dir=tmp_path / "run1",
            extra_cli={},
        )
        assert "--frontend" in argv
        assert argv[argv.index("--frontend") + 1] == "multiscale"
        assert "--lr" in argv
        assert argv[argv.index("--lr") + 1] == "0.001"
        assert "--crl-epochs" in argv
        assert argv[argv.index("--crl-epochs") + 1] == "50"

    def test_unflagged_fields_become_config_overrides_json(self, tmp_path):
        argv = runner._build_train_argv(
            python_exe="python",
            train_script=Path("train.py"),
            # morlet_use_phase and lambda_interv have no dedicated flag in train.py
            merged_cfg={
                "frontend_type": "morlet_per_sensor",
                "morlet_use_phase": True,
                "lambda_interv": 2.0,
            },
            run_save_dir=tmp_path / "run1",
            extra_cli={},
        )
        assert "--config-overrides-json" in argv
        json_str = argv[argv.index("--config-overrides-json") + 1]
        overrides = json.loads(json_str)
        assert overrides == {"morlet_use_phase": True, "lambda_interv": 2.0}

    def test_store_true_flag_only_when_true(self, tmp_path):
        argv_true = runner._build_train_argv(
            python_exe="python",
            train_script=Path("train.py"),
            merged_cfg={"morlet_learnable_w0": True},
            run_save_dir=tmp_path / "a",
            extra_cli={},
        )
        assert "--morlet-learnable-w0" in argv_true

        argv_false = runner._build_train_argv(
            python_exe="python",
            train_script=Path("train.py"),
            merged_cfg={"morlet_learnable_w0": False},
            run_save_dir=tmp_path / "b",
            extra_cli={},
        )
        # store_true flag absent when False — train.py defaults to False.
        assert "--morlet-learnable-w0" not in argv_false

    def test_extra_cli_appended(self, tmp_path):
        argv = runner._build_train_argv(
            python_exe="python",
            train_script=Path("train.py"),
            merged_cfg={},
            run_save_dir=tmp_path / "r",
            extra_cli={
                "--steps-per-epoch": 2,
                "--ds-epochs": 1,
                "--cache-dir": "/tmp/cache",
            },
        )
        assert "--steps-per-epoch" in argv
        assert argv[argv.index("--steps-per-epoch") + 1] == "2"
        assert "--cache-dir" in argv

    def test_save_dir_included(self, tmp_path):
        save_dir = tmp_path / "myrun"
        argv = runner._build_train_argv(
            python_exe="python",
            train_script=Path("train.py"),
            merged_cfg={},
            run_save_dir=save_dir,
            extra_cli={},
        )
        assert "--save-dir" in argv
        assert argv[argv.index("--save-dir") + 1] == str(save_dir)


# ---------------------------------------------------------------------------
# write_sweep_summary
# ---------------------------------------------------------------------------


class TestWriteSweepSummary:
    def test_csv_has_one_row_per_run(self, tmp_path):
        summaries = [
            {
                "name": "a",
                "overrides": {"frontend_type": "multiscale"},
                "best_val_ref_elbo": 0.5,
                "val_type_f1": 0.7,
                "returncode": 0,
                "elapsed_min": 1.0,
            },
            {
                "name": "b",
                "overrides": {"frontend_type": "morlet_learnable"},
                "best_val_ref_elbo": 0.6,
                "val_type_f1": 0.65,
                "returncode": 0,
                "elapsed_min": 1.2,
            },
        ]
        runner.write_sweep_summary(summaries, tmp_path)

        csv_path = tmp_path / "summary.csv"
        assert csv_path.exists()
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert rows[0]["name"] == "a"
        # overrides are JSON-encoded in the CSV for round-trip.
        assert json.loads(rows[0]["overrides"])["frontend_type"] == "multiscale"

    def test_json_written_alongside_csv(self, tmp_path):
        summaries = [{"name": "solo", "overrides": {}, "returncode": 0}]
        runner.write_sweep_summary(summaries, tmp_path)

        json_path = tmp_path / "summary.json"
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert data == {"runs": summaries}

    def test_missing_metrics_write_empty_string(self, tmp_path):
        """A failed run has no metrics — CSV row should be fine with blanks,
        not crash."""
        summaries = [{"name": "failed", "overrides": {}, "returncode": 1, "error": "NaN in loss"}]
        runner.write_sweep_summary(summaries, tmp_path)

        with open(tmp_path / "summary.csv") as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["name"] == "failed"
        assert rows[0]["returncode"] == "1"
        assert rows[0]["val_type_f1"] == ""


# ---------------------------------------------------------------------------
# Pre-existing helpers (sanity coverage)
# ---------------------------------------------------------------------------


class TestMetricReaders:
    """Sanity-check the metric readers the sweep runner uses to pull summaries
    from completed run dirs."""

    def test_best_crl_elbo_from_csv(self, tmp_path):
        csv_path = tmp_path / "crl_metrics.csv"
        csv_path.write_text("epoch,val_ref_elbo\n" "0,2.5\n" "1,1.8\n" "2,2.0\n")
        best, epoch = runner._best_crl_elbo(tmp_path)
        assert best == 1.8
        assert epoch == 1

    def test_best_crl_elbo_missing_file(self, tmp_path):
        best, epoch = runner._best_crl_elbo(tmp_path)
        assert best == float("inf")
        assert epoch == -1

    def test_collect_run_metrics_default_on_empty_dir(self, tmp_path):
        metrics = runner._collect_run_metrics(tmp_path)
        assert metrics["best_val_ref_elbo"] is None  # inf → None serialization
        assert metrics["val_type_f1"] == 0.0
