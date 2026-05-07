"""Pre-flight: for every legacy run (no m3nvc eval on any probe), try to
load the model state_dict from the *first* available probe checkpoint.
If it fails with an architecture mismatch, mark the run as dead. If it
loads, mark it as eligible for full re-evaluation.

This is much faster than running eval.py — no inference, no data loading.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAVED = ROOT / "saved_crl" / "runs"

sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from crl_vehicle.config import CRLConfig  # noqa: E402
from training.trainer import CRLModel  # noqa: E402


def _probe_has_m3nvc(probe_dir: Path) -> bool:
    eval_root = probe_dir.parent.parent / "eval" / probe_dir.name
    if not eval_root.is_dir():
        return False
    for cand in (
        eval_root / "m3nvc" / "eval_report.json",
        eval_root / "pres" / "m3nvc" / "eval_report.json",
        eval_root / "type" / "m3nvc" / "eval_report.json",
    ):
        if cand.exists():
            return True
    return False


def _ckpt_path(probe_dir: Path) -> Path | None:
    for name in (
        "downstream_best_pres.pth",
        "downstream_best_type.pth",
        "downstream_best.pth",
    ):
        p = probe_dir / name
        if p.exists():
            return p
    return None


def _load_probe(probe_dir: Path) -> tuple[bool, str]:
    """Try to load the probe's checkpoint into a freshly-built CRLModel.
    Returns (ok, message)."""
    meta_path = probe_dir / "meta.json"
    if not meta_path.exists():
        return False, "no probe meta.json"
    meta = json.loads(meta_path.read_text())
    cfg_dict = meta.get("config", {})
    sensors = meta.get("sensors", ["audio", "seismic"])
    probe_mode = meta.get("probe_mode", "linear_ztype")
    try:
        cfg = CRLConfig(
            **{
                k: v
                for k, v in cfg_dict.items()
                if hasattr(CRLConfig, k) or k in CRLConfig.__dataclass_fields__
            }
        )
    except Exception as e:
        return False, f"CRLConfig() raised: {e}"

    ckpt = _ckpt_path(probe_dir)
    if ckpt is None:
        return False, "no checkpoint .pth found"

    try:
        model = CRLModel(cfg, sensors=sensors, probe_mode=probe_mode)
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        return True, ckpt.name
    except Exception as e:
        msg = str(e)
        # Trim very long state-dict-mismatch messages.
        if len(msg) > 600:
            msg = msg[:600] + "…"
        return False, msg


def main() -> int:
    runs_to_check: list[tuple[Path, list[Path]]] = []
    for meta in sorted(SAVED.rglob("crl/meta.json")):
        run = meta.parent.parent
        ds = run / "downstream"
        if not ds.is_dir():
            continue
        legacy_probes = [
            p for p in sorted(ds.iterdir())
            if p.is_dir() and not _probe_has_m3nvc(p) and _ckpt_path(p) is not None
        ]
        if legacy_probes:
            runs_to_check.append((run, legacy_probes))

    print(f"Pre-flighting {len(runs_to_check)} legacy runs.\n")
    eligible: list[Path] = []
    dead: list[tuple[Path, str]] = []
    for run, probes in runs_to_check:
        rel = run.relative_to(SAVED)
        # Try the first probe.
        ok, msg = _load_probe(probes[0])
        if ok:
            print(f"[ OK ] {rel}  (via {probes[0].name}/{msg})")
            eligible.append(run)
        else:
            print(f"[DEAD] {rel}")
            print(f"       reason: {msg}")
            dead.append((run, msg))

    print("\n" + "=" * 80)
    print(f"Eligible for re-eval: {len(eligible)}")
    print(f"Dead (architecture mismatch): {len(dead)}")
    if dead:
        print("\nDead runs:")
        for run, _ in dead:
            print(f"  {run.relative_to(SAVED)}")
    if eligible:
        print("\nEligible runs (will produce real m3nvc numbers when re-eval'd):")
        for run in eligible:
            print(f"  {run.relative_to(SAVED)}")

    out = ROOT / "scratch" / "reeval_preflight_results.json"
    out.write_text(json.dumps(
        {
            "eligible": [str(r.relative_to(SAVED)) for r in eligible],
            "dead": [{"run": str(r.relative_to(SAVED)), "reason": m} for r, m in dead],
        },
        indent=2,
    ))
    print(f"\nResults JSON: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
