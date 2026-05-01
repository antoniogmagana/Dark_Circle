"""Helpers for two-stage CRL training.

Stage 2 loads a converged fixed-Morlet checkpoint (morlet_per_sensor or
morlet_fused) into a learnable model (morlet_learnable or
morlet_learnable_fused) and fine-tunes scales against the already-trained
encoder, instead of chasing a moving target from scratch.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

# Topology pairs: target learnable frontend → source fixed frontend(s) that
# are compatible for stage-2 loading. Must match on late-fusion vs
# early-fusion structure (encoder/decoder shape differs).
_TOPOLOGY_PAIRS = {
    "morlet_learnable": ("morlet_per_sensor",),
    "morlet_learnable_fused": ("morlet_fused",),
}


@dataclass
class CandidateEvaluation:
    """Result of checking one candidate run against the target config."""

    path: Path
    frontend: str | None
    reasons: list[str]  # empty == compatible

    @property
    def compatible(self) -> bool:
        return not self.reasons


def _floats_close(a: float, b: float, tol: float = 1e-9) -> bool:
    """Tolerant float comparison for YAML/JSON round-tripped values."""
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)


def _params_equal(a: dict, b: dict) -> tuple[bool, str]:
    """Deep-equal morlet_per_sensor_params dicts with tolerant floats."""
    if set(a.keys()) != set(b.keys()):
        return False, f"sensor keys differ: {sorted(a.keys())} vs {sorted(b.keys())}"
    for sensor in a:
        sa, sb = a[sensor], b[sensor]
        if set(sa.keys()) != set(sb.keys()):
            return False, f"{sensor}: field keys differ"
        for k in sa:
            va, vb = sa[k], sb[k]
            if isinstance(va, float) and isinstance(vb, float):
                if not _floats_close(va, vb):
                    return False, f"{sensor}.{k}: {va} != {vb}"
            elif va != vb:
                return False, f"{sensor}.{k}: {va} != {vb}"
    return True, ""


def _evaluate_candidate(
    meta_path: Path,
    target_frontend: str,
    target_sensors: list[str],
    target_cfg: dict,
) -> CandidateEvaluation:
    """Inspect one candidate run's meta.json and return why it does or
    does not match the target config."""
    run_path = meta_path.parent.parent  # <run>/crl/meta.json → <run>
    try:
        meta = json.loads(meta_path.read_text())
    except Exception as e:
        return CandidateEvaluation(
            path=run_path,
            frontend=None,
            reasons=[f"could not parse meta.json: {e}"],
        )

    src_cfg = meta.get("config", {})
    src_frontend = src_cfg.get("frontend_type")
    reasons: list[str] = []

    allowed_sources = _TOPOLOGY_PAIRS.get(target_frontend, ())
    if src_frontend not in allowed_sources:
        reasons.append(
            f"frontend_type {src_frontend!r} not a stage-1 match for "
            f"{target_frontend!r} (need one of {list(allowed_sources)})"
        )

    src_sensors = meta.get("sensors") or src_cfg.get("sensors") or []
    if list(src_sensors) != list(target_sensors):
        reasons.append(f"sensors differ: {src_sensors} vs {target_sensors}")

    if src_cfg.get("d_model") != target_cfg.get("d_model"):
        reasons.append(f"d_model differs: {src_cfg.get('d_model')} vs {target_cfg.get('d_model')}")

    if bool(src_cfg.get("morlet_use_phase")) != bool(target_cfg.get("morlet_use_phase")):
        reasons.append(
            f"morlet_use_phase differs: {src_cfg.get('morlet_use_phase')} "
            f"vs {target_cfg.get('morlet_use_phase')}"
        )

    src_params = src_cfg.get("morlet_per_sensor_params", {})
    tgt_params = target_cfg.get("morlet_per_sensor_params", {})
    ok, reason = _params_equal(src_params, tgt_params)
    if not ok:
        reasons.append(f"morlet_per_sensor_params differ: {reason}")

    return CandidateEvaluation(
        path=run_path,
        frontend=src_frontend,
        reasons=reasons,
    )


def find_compatible_run(
    target_frontend: str,
    target_sensors: list[str],
    target_cfg: dict,
    search_root: Path,
    verbose: bool = True,
) -> Path:
    """Return the path to the most recent run compatible with the target config.

    Searches search_root/*/crl/meta.json, newest first. Raises with a full
    candidate breakdown if zero runs match — makes debugging obvious.

    The target_cfg is a dict (not a CRLConfig instance) so the helper stays
    testable without constructing the full config machinery.
    """
    candidates: list[Path] = sorted(
        search_root.glob("*/crl/meta.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No candidate runs found under {search_root} " f"(expected <run>/crl/meta.json)"
        )

    evaluations: list[CandidateEvaluation] = []
    for meta_path in candidates:
        ev = _evaluate_candidate(
            meta_path,
            target_frontend,
            target_sensors,
            target_cfg,
        )
        evaluations.append(ev)
        if verbose:
            status = "✓" if ev.compatible else "✗"
            print(f"  {status} {ev.path.name} ({ev.frontend})")
            for reason in ev.reasons:
                print(f"      {reason}")
        if ev.compatible:
            return ev.path

    # No match — raise with the full breakdown.
    lines = [
        f"No compatible stage-1 run found under {search_root} for target "
        f"{target_frontend!r} (sensors={target_sensors})."
    ]
    for ev in evaluations:
        lines.append(f"  - {ev.path.name} ({ev.frontend}):")
        for reason in ev.reasons:
            lines.append(f"      {reason}")
    raise RuntimeError("\n".join(lines))


def resolve_source_checkpoint(run_dir: Path) -> Path:
    """Return the CRL checkpoint to load from a stage-1 run dir. Prefers
    crl_best.pth; falls back to crl_final.pth."""
    crl_dir = run_dir / "crl"
    if not crl_dir.is_dir():
        # Support the flat layout too — some runs put files at top level.
        crl_dir = run_dir
    for name in ("crl_best.pth", "crl_final.pth"):
        p = crl_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No CRL checkpoint found in {crl_dir} " f"(looked for crl_best.pth, crl_final.pth)"
    )
