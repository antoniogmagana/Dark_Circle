#!/usr/bin/env python3
"""Replay the split restructure on the server using manifest.csv.

Idempotent: can be rerun safely. Works regardless of which split each file
currently sits in (locates by filename, moves to target_split).

Usage (from anywhere):
    python apply_manifest.py --parsed-dir /path/to/data_files/parsed \
                             --manifest /path/to/manifest.csv \
                             [--cache-dir /path/to/saved_crl/cache] \
                             [--dry-run]

What it does:
    1. For each row in manifest.csv, finds the file in any of train/val/test
       under parsed-dir; if not in target_split, moves it there.
    2. Deletes any parquet under parsed-dir that is NOT in the manifest
       (accel, m3nvc multi-target, m3nvc background).
    3. Clears cache-dir if provided.
"""
from __future__ import annotations
import argparse, csv, shutil, sys
from pathlib import Path

SPLITS = ("train", "val", "test")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--parsed-dir",
        required=True,
        type=Path,
        help="Path to data_files/parsed/ containing train/val/test/",
    )
    ap.add_argument("--manifest", required=True, type=Path, help="Path to manifest.csv")
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional: saved_crl/cache/ to clear after moves",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="Print actions without executing"
    )
    args = ap.parse_args()

    parsed = args.parsed_dir.resolve()
    for sp in SPLITS:
        d = parsed / sp
        if not d.is_dir():
            print(f"ERROR: {d} does not exist", file=sys.stderr)
            return 2

    # 1. Load manifest.
    with open(args.manifest) as fh:
        rows = list(csv.DictReader(fh))
    expected = {r["filename"]: r["target_split"] for r in rows}
    print(f"Manifest: {len(expected)} files expected across splits.")

    # 2. Move files to target split.
    moves, missing = 0, []
    for fname, target in expected.items():
        target_path = parsed / target / fname
        if target_path.exists():
            continue
        found = None
        for sp in SPLITS:
            candidate = parsed / sp / fname
            if candidate.exists():
                found = candidate
                break
        if found is None:
            missing.append(fname)
            continue
        action = f"mv {found.relative_to(parsed)} -> {target}/{fname}"
        if args.dry_run:
            print(f"  [DRY] {action}")
        else:
            shutil.move(str(found), str(target_path))
            print(f"  {action}")
        moves += 1

    if missing:
        print(
            f"\nWARNING: {len(missing)} file(s) in manifest not found on disk:",
            file=sys.stderr,
        )
        for f in missing[:10]:
            print(f"  {f}", file=sys.stderr)
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more", file=sys.stderr)

    # 3. Delete files not in manifest (accel, m3nvc multi/background).
    deletes = 0
    for sp in SPLITS:
        for f in sorted((parsed / sp).glob("*.parquet")):
            if f.name not in expected:
                action = f"rm {sp}/{f.name}"
                if args.dry_run:
                    print(f"  [DRY] {action}")
                else:
                    f.unlink()
                    print(f"  {action}")
                deletes += 1

    # 4. Clear cache.
    if args.cache_dir and args.cache_dir.exists():
        for item in args.cache_dir.iterdir():
            action = f"rm -rf cache/{item.name}"
            if args.dry_run:
                print(f"  [DRY] {action}")
            else:
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
                print(f"  {action}")

    print(
        f"\nSummary: {moves} moves, {deletes} deletes"
        + (" (DRY RUN — no changes applied)" if args.dry_run else "")
    )

    # 5. Post-state report.
    if not args.dry_run:
        print("\nFinal counts:")
        for sp in SPLITS:
            n = len(list((parsed / sp).glob("*.parquet")))
            print(f"  {sp:6s}: {n} files")

    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
