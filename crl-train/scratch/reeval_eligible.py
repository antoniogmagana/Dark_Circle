"""Re-evaluate the 3 pre-flight-eligible legacy runs on the current test
set. For each run, only the per-head winning probe is re-evaluated (per
the leaderboard schema), with two filters: m3nvc-only (the new dataset)
and unfiltered (full focal+iobt+m3nvc).

Streams output so progress is visible. Skips runs that fail mid-eval.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL_PY = ROOT / "eval.py"
SAVED = ROOT / "saved_crl" / "runs"

# (run_path, pres_probe, type_probe) — picked from leaderboard winners.
ELIGIBLE = [
    ("multiscale/vae/v3_lowfreq", "linear_fullz__crl_best", "mlp_ztype__crl_best_aux_type"),
    ("multiscale/disentangled/v3_lowfreq", "linear_fullz__crl_best_aux_type", "linear_signal__crl_best_aux_type"),
    ("morlet_per_sensor/vae/phase_v1_diag", "mlp_ztype__crl_best_aux_type", "mlp_ztype__crl_best_aux_type"),
]


def run_one(probe_dir: Path, head: str, filter_arg: list[str] | None) -> tuple[bool, str]:
    cmd = [
        sys.executable,
        str(EVAL_PY),
        "--save-dir",
        str(probe_dir),
        "--head",
        head,
        "--num-workers",
        "4",
    ]
    if filter_arg:
        cmd += ["--include-datasets", *filter_arg]
    print(f"  $ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=ROOT, timeout=1800)
    return proc.returncode == 0, ""


def main() -> int:
    results: list[dict] = []
    total = len(ELIGIBLE) * 2 * 2  # 3 runs × 2 heads × 2 filters
    i = 0
    for run_rel, pres_probe, type_probe in ELIGIBLE:
        run = SAVED / run_rel
        for head, probe_name in (("pres", pres_probe), ("type", type_probe)):
            probe_dir = run / "downstream" / probe_name
            for filter_arg, label in ((None, "full"), (["m3nvc"], "m3nvc")):
                i += 1
                print(f"\n[{i}/{total}] {run_rel} head={head} probe={probe_name} filter={label}",
                      flush=True)
                ok, _ = run_one(probe_dir, head, filter_arg)
                results.append({
                    "run": run_rel,
                    "head": head,
                    "probe": probe_name,
                    "filter": label,
                    "ok": ok,
                })
                if not ok:
                    print(f"   ⚠ failed: {run_rel} head={head} probe={probe_name} filter={label}",
                          flush=True)

    out = ROOT / "scratch" / "reeval_eligible_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n\nWrote {out}")
    print(f"OK: {sum(1 for r in results if r['ok'])} / Failed: {sum(1 for r in results if not r['ok'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
