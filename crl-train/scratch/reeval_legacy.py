"""Re-evaluate every legacy run (no m3nvc eval artifact) on the current
test set. For each (run, probe, head) triple, runs eval.py with both
--include-datasets m3nvc and a full-split (no filter) eval. Captures
crashes per-(run, probe) — if state_dict loading fails for one head, the
other head will fail the same way, so we skip it.

Output: prints a table at the end with status per (run, probe).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL_PY = ROOT / "eval.py"
SAVED = ROOT / "saved_crl" / "runs"


def _probe_has_m3nvc(probe_dir: Path) -> bool:
    """Probe-level m3nvc check — accepts both flat and per-head layouts."""
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


def find_legacy_jobs() -> list[tuple[Path, list[str]]]:
    """Return [(probe_dir, [heads_with_ckpt])] for every probe that lacks
    an m3nvc eval. Heads include all that have a usable checkpoint
    (legacy combined or per-head)."""
    jobs: list[tuple[Path, list[str]]] = []
    for meta in sorted(SAVED.rglob("crl/meta.json")):
        run = meta.parent.parent
        ds = run / "downstream"
        if not ds.is_dir():
            continue
        for probe_dir in sorted(ds.iterdir()):
            if not probe_dir.is_dir():
                continue
            if _probe_has_m3nvc(probe_dir):
                continue
            heads: list[str] = []
            has_legacy = (probe_dir / "downstream_best.pth").exists()
            for head, ckpt_name in (
                ("pres", "downstream_best_pres.pth"),
                ("type", "downstream_best_type.pth"),
            ):
                if (probe_dir / ckpt_name).exists() or has_legacy:
                    heads.append(head)
            if heads:
                jobs.append((probe_dir, heads))
    return jobs


def run_eval(probe_dir: Path, head: str, include_datasets: list[str] | None) -> tuple[bool, str]:
    """Run eval.py once. Returns (ok, last_lines_or_error)."""
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
    if include_datasets:
        cmd += ["--include-datasets", *include_datasets]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, cwd=ROOT, timeout=900,
    )
    if proc.returncode != 0:
        # Capture the most informative tail
        err = (proc.stderr or proc.stdout or "").strip().splitlines()
        tail = "\n".join(err[-15:])
        return False, tail
    return True, ""


def main() -> int:
    jobs = find_legacy_jobs()
    print(f"Found {len(jobs)} (probe, heads) jobs across legacy runs.\n")
    results: list[dict] = []
    crashed_runs: set[str] = set()
    for i, (probe_dir, heads) in enumerate(jobs, 1):
        rel = probe_dir.relative_to(SAVED)
        run_key = str(rel.parts[0:-2])  # frontend/mode/run-id
        if run_key in crashed_runs:
            print(f"[{i}/{len(jobs)}] SKIP (run already crashed): {rel}")
            results.append({"probe_dir": str(rel), "status": "skip-run-crashed"})
            continue

        # Run m3nvc filter and full split. m3nvc first — it's the new data point.
        # If state_dict loads fail, it'll fail here and we mark the whole run crashed.
        for filter_set, label in (
            (["m3nvc"], "m3nvc"),
            (None, "full"),
        ):
            for head in heads:
                print(f"[{i}/{len(jobs)}] {rel} head={head} filter={label}", flush=True)
                ok, msg = run_eval(probe_dir, head, filter_set)
                rec = {
                    "probe_dir": str(rel),
                    "head": head,
                    "filter": label,
                    "status": "ok" if ok else "crash",
                }
                if not ok:
                    rec["error_tail"] = msg
                    print(f"   CRASH:\n{msg}\n")
                    # Mark the whole run as crashed so we skip remaining probes/heads.
                    crashed_runs.add(run_key)
                results.append(rec)
                if not ok:
                    break  # Skip remaining heads/filters for this probe.
            if not ok:
                break

    # Summary.
    print("\n" + "=" * 80)
    print(f"DONE. {len([r for r in results if r['status'] == 'ok'])} ok, "
          f"{len([r for r in results if r['status'] == 'crash'])} crashes, "
          f"{len([r for r in results if r['status'] == 'skip-run-crashed'])} skipped.")
    print("\nCRASHED RUNS (entire run unloadable on current code):")
    for run_key in sorted(crashed_runs):
        print(f"  {run_key}")

    # Write JSON.
    out = ROOT / "scratch" / "reeval_legacy_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults JSON: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
