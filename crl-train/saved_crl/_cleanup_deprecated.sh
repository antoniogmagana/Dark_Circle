#!/usr/bin/env bash
# Remove run artifacts deprecated by the 2026-05-03 cleanup pass:
#   1) Plain `frontend_type=morlet` runs — config.py now raises on construction
#      (seismic-1-token regression). These checkpoints can't be loaded.
#   2) Top-level mirrors of paths that _reorganize.sh already moved into
#      runs/. diff -rq confirmed byte-identical at flagging time.
#   3) Pre-Checkpoint-1 framework / id-split smoke output, superseded by the
#      committed framework (Checkpoints 1–3, fully validated).
#   4) An incomplete disentangled CRL run (no downstream/eval), name was
#      already "run1_incomplete".
#
# Usage:
#   bash _cleanup_deprecated.sh           # dry-run: prints every rm it would do
#   EXECUTE=1 bash _cleanup_deprecated.sh # actually delete
#
# Idempotent: skips when path is already gone.
# Run from inside saved_crl/.

set -euo pipefail

if [[ "$(basename "$PWD")" != "saved_crl" ]]; then
  echo "Error: must be run from inside saved_crl/. Currently in: $PWD" >&2
  exit 1
fi

EXECUTE="${EXECUTE:-0}"

rm_safe() {
  local target="$1"
  local reason="$2"
  if [[ ! -e "$target" ]]; then
    echo "SKIP (gone): $target"
    return
  fi
  if [[ "$EXECUTE" == "1" ]]; then
    rm -rf "$target"
    echo "REMOVED ($reason): $target"
  else
    echo "[dry-run] rm -rf $target  # $reason"
  fi
}

# ---------------------------------------------------------------------------
# 1) Plain `frontend_type=morlet` — deprecated by config.py raise
# ---------------------------------------------------------------------------
rm_safe "runs/morlet/vae/legacy_oldlayout/2026-04-20_09-07" "deprecated plain morlet"
rm_safe "runs/morlet/vae/legacy_oldlayout/2026-04-20_21-23" "deprecated plain morlet"
rm_safe "runs/morlet/vae/legacy_oldlayout"                  "empty after above"
rm_safe "runs/morlet/vae"                                   "empty after above"
rm_safe "runs/morlet"                                       "empty after above"

rm_safe "runs/_archive/experiments_sweep/baseline_morlet/2026-04-20_15-14-35" \
        "deprecated plain morlet (sweep)"
rm_safe "runs/_archive/experiments_sweep/baseline_morlet/2026-04-21_03-23-02" \
        "deprecated plain morlet (sweep)"
rm_safe "runs/_archive/experiments_sweep/baseline_morlet" \
        "empty after above"

# ---------------------------------------------------------------------------
# 2) Top-level dirs already mirrored under runs/  (diff -rq verified)
# ---------------------------------------------------------------------------
rm_safe "supervised_multiscale_idsplit"          "duplicate of runs/supervised/id_split/v1"
rm_safe "supervised_multiscale_idsplit_v2"       "duplicate of runs/supervised/id_split/v2"
rm_safe "supervised_multiscale_idsplit_v2_fast"  "duplicate of runs/supervised/id_split/v2_fast"
rm_safe "supervised_multiscale_filesplit_v2"     "duplicate of runs/supervised/file_split/v2"
rm_safe "smoke_test"                             "duplicate of runs/_archive/smoke_test"

# ---------------------------------------------------------------------------
# 3) Pre-Checkpoint-1 framework smokes — superseded
# ---------------------------------------------------------------------------
rm_safe "runs/_archive/framework_smoke" "pre-Checkpoint-1 framework smoke"
rm_safe "runs/_archive/id_split_smoke"  "pre-Checkpoint-1 id-split smoke"

# ---------------------------------------------------------------------------
# 4) Incomplete run
# ---------------------------------------------------------------------------
rm_safe "runs/multiscale/disentangled/run1_incomplete" \
        "incomplete (CRL only, no downstream/eval)"
