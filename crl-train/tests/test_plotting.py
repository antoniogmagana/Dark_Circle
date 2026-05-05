"""Tests for crl_vehicle/plotting.py — the shared poster-style helpers.

These tests construct RunMetrics directly (they're a dataclass) so they
don't need real saved_crl artifacts. The save tests render to tmp_path.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from crl_vehicle import analysis as A
from crl_vehicle import plotting as P


# --------------------------------------------------------------------------
# Fixture helpers
# --------------------------------------------------------------------------


def _rm(
    name: str,
    frontend: str,
    *,
    best_type_f1: float | None = 0.7,
    best_val_ref_elbo: float | None = 1.0,
    min_dataset_type_f1: float | None = 0.5,
    diverged: bool = False,
) -> A.RunMetrics:
    return A.RunMetrics(
        name=name,
        path=Path(f"/tmp/{name}"),
        config={"frontend_type": frontend},
        sensors=["audio", "seismic"],
        best_type_f1=best_type_f1,
        best_val_ref_elbo=best_val_ref_elbo,
        min_dataset_type_f1=min_dataset_type_f1,
        diverged=diverged,
    )


# --------------------------------------------------------------------------
# top_n_runs
# --------------------------------------------------------------------------


class TestTopN:
    def test_returns_at_most_n(self):
        rms = [_rm(f"r{i}", "multiscale", best_type_f1=0.5 + i * 0.01) for i in range(10)]
        out = P.top_n_runs(rms, n=5)
        assert len(out) == 5

    def test_returns_fewer_when_pool_smaller(self):
        rms = [_rm(f"r{i}", "multiscale", best_type_f1=0.5 + i * 0.01) for i in range(3)]
        out = P.top_n_runs(rms, n=5)
        assert len(out) == 3

    def test_sorts_descending_for_f1(self):
        rms = [
            _rm("low", "multiscale", best_type_f1=0.4),
            _rm("hi", "multiscale", best_type_f1=0.9),
            _rm("mid", "multiscale", best_type_f1=0.6),
        ]
        out = P.top_n_runs(rms, n=3, by="best_type_f1")
        assert [rm.name for rm in out] == ["hi", "mid", "low"]

    def test_sorts_ascending_for_ref_elbo(self):
        rms = [
            _rm("hi", "multiscale", best_val_ref_elbo=10.0),
            _rm("lo", "multiscale", best_val_ref_elbo=0.5),
            _rm("mid", "multiscale", best_val_ref_elbo=2.0),
        ]
        out = P.top_n_runs(rms, n=3, by="best_val_ref_elbo")
        # Lower is better → "lo" comes first.
        assert [rm.name for rm in out] == ["lo", "mid", "hi"]

    def test_excludes_diverged_by_default(self):
        rms = [
            _rm("good", "multiscale", best_type_f1=0.5, diverged=False),
            _rm("bad", "multiscale", best_type_f1=0.9, diverged=True),
        ]
        out = P.top_n_runs(rms, n=5)
        assert [rm.name for rm in out] == ["good"]

    def test_includes_diverged_when_requested(self):
        rms = [
            _rm("good", "multiscale", best_type_f1=0.5, diverged=False),
            _rm("bad", "multiscale", best_type_f1=0.9, diverged=True),
        ]
        out = P.top_n_runs(rms, n=5, exclude_diverged=False)
        # bad has higher f1 → first.
        assert [rm.name for rm in out] == ["bad", "good"]

    def test_drops_runs_with_none_metric(self):
        rms = [
            _rm("a", "multiscale", best_type_f1=None),
            _rm("b", "multiscale", best_type_f1=0.7),
        ]
        out = P.top_n_runs(rms, n=5)
        assert [rm.name for rm in out] == ["b"]


# --------------------------------------------------------------------------
# assign_run_styles
# --------------------------------------------------------------------------


class TestAssignRunStyles:
    def test_one_entry_per_run(self):
        rms = [_rm(f"r{i}", "multiscale") for i in range(3)]
        styles = P.assign_run_styles(rms)
        assert len(styles) == 3

    def test_distinct_colors(self):
        rms = [_rm(f"r{i}", "multiscale") for i in range(5)]
        styles = P.assign_run_styles(rms)
        colors = [s["color"] for s in styles]
        assert len(set(colors)) == 5

    def test_linestyle_groups_by_frontend_family(self):
        # Two morlet variants in the same family + multiscale → should get
        # two distinct linestyles total (morlet family + multiscale family).
        rms = [
            _rm("a", "morlet"),
            _rm("b", "morlet_per_sensor"),
            _rm("c", "multiscale"),
        ]
        styles = P.assign_run_styles(rms)
        # a (morlet) and b (morlet_per_sensor) are both morlet family.
        assert styles[0]["linestyle"] == styles[1]["linestyle"]
        # c (multiscale) is a different family.
        assert styles[2]["linestyle"] != styles[0]["linestyle"]

    def test_label_includes_frontend(self):
        rms = [_rm("v3", "morlet_per_sensor")]
        styles = P.assign_run_styles(rms)
        assert "v3" in styles[0]["label"]
        assert "morlet_per_sensor" in styles[0]["label"]

    def test_handles_duplicate_run_names(self):
        # Two runs share basename — must each get an independent slot.
        rms = [
            _rm("v3_lowfreq", "multiscale"),
            _rm("v3_lowfreq", "multiscale"),
        ]
        styles = P.assign_run_styles(rms)
        assert len(styles) == 2
        # Distinct colors despite the name collision.
        assert styles[0]["color"] != styles[1]["color"]


# --------------------------------------------------------------------------
# poster_save
# --------------------------------------------------------------------------


class TestPosterSave:
    def test_writes_png_and_pdf(self, tmp_path: Path):
        P.apply_poster_style()
        fig, ax = plt.subplots()
        ax.plot([0, 1, 2], [0, 1, 4])
        out_stem = tmp_path / "subdir" / "test_fig"
        P.poster_save(fig, out_stem)
        assert out_stem.with_suffix(".png").exists()
        assert out_stem.with_suffix(".pdf").exists()
        assert out_stem.with_suffix(".png").stat().st_size > 1000

    def test_creates_parent_dir(self, tmp_path: Path):
        fig, ax = plt.subplots()
        ax.plot([0, 1])
        # Two-deep nested dir that doesn't exist.
        out_stem = tmp_path / "a" / "b" / "fig"
        P.poster_save(fig, out_stem)
        assert out_stem.with_suffix(".png").exists()


# --------------------------------------------------------------------------
# apply_poster_style
# --------------------------------------------------------------------------


class TestApplyPosterStyle:
    def test_sets_font_size_to_16(self):
        P.apply_poster_style()
        assert matplotlib.rcParams["font.size"] == 16
        assert matplotlib.rcParams["font.weight"] == "bold"

    def test_sets_line_width_to_2(self):
        P.apply_poster_style()
        assert matplotlib.rcParams["lines.linewidth"] == 2.0

    def test_sets_savefig_dpi_to_300(self):
        P.apply_poster_style()
        assert matplotlib.rcParams["savefig.dpi"] == 300


# --------------------------------------------------------------------------
# plot_confusion_matrix
# --------------------------------------------------------------------------


class TestPlotConfusionMatrix:
    def test_writes_file(self, tmp_path: Path):
        P.apply_poster_style()
        cm = [[10, 2], [1, 7]]
        out = tmp_path / "cm.png"
        P.plot_confusion_matrix(cm, ["a", "b"], "test", out)
        assert out.exists()
        assert out.stat().st_size > 1000

    def test_handles_3_class(self, tmp_path: Path):
        P.apply_poster_style()
        cm = [[5, 1, 0], [1, 4, 1], [0, 0, 6]]
        out = tmp_path / "cm3.png"
        P.plot_confusion_matrix(cm, ["x", "y", "z"], "test 3-class", out)
        assert out.exists()
