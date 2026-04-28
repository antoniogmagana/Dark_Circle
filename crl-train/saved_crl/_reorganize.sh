#!/usr/bin/env bash
# Reorganize crl-train/saved_crl/ into runs/<model>/<variant>/<run-id>/ layout.
#
# Usage:
#   bash _reorganize.sh           # dry-run: prints every mv it would do
#   EXECUTE=1 bash _reorganize.sh # actually run the moves
#
# Idempotent: skips mv when source already missing or destination already exists.
# Run from inside saved_crl/.

set -euo pipefail

if [[ "$(basename "$PWD")" != "saved_crl" ]]; then
  echo "Error: must be run from inside saved_crl/. Currently in: $PWD" >&2
  exit 1
fi

EXECUTE="${EXECUTE:-0}"

mv_safe() {
  local src="$1"
  local dst="$2"
  if [[ ! -e "$src" ]]; then
    echo "SKIP (no source): $src"
    return
  fi
  if [[ -e "$dst" ]]; then
    echo "SKIP (dst exists): $src -> $dst"
    return
  fi
  local parent
  parent="$(dirname "$dst")"
  if [[ "$EXECUTE" == "1" ]]; then
    mkdir -p "$parent"
    mv "$src" "$dst"
    echo "MOVED: $src -> $dst"
  else
    echo "[dry-run] mkdir -p $parent && mv $src $dst"
  fi
}

# ---------------------------------------------------------------------------
# Phase 1: id_split/ runs (already in crl/downstream/eval layout)
# ---------------------------------------------------------------------------

# multiscale / vae
mv_safe "id_split/multiscale_run1"            "runs/multiscale/vae/v1"
mv_safe "id_split/multiscale_run1_diag"       "runs/multiscale/vae/v1_diag"
mv_safe "id_split/multiscale_v2"              "runs/multiscale/vae/v2"
mv_safe "id_split/multiscale_v3_lowfreq"      "runs/multiscale/vae/v3_lowfreq"

# multiscale / disentangled
mv_safe "id_split/multiscale_v3_lowfreq_disentangled" \
        "runs/multiscale/disentangled/v3_lowfreq"
mv_safe "id_split/disentangled_multiscale_run1" \
        "runs/multiscale/disentangled/run1_incomplete"

# morlet_per_sensor / vae
mv_safe "id_split/morlet_per_sensor_phase_run1"      "runs/morlet_per_sensor/vae/phase_v1"
mv_safe "id_split/morlet_per_sensor_phase_run1_diag" "runs/morlet_per_sensor/vae/phase_v1_diag"

# ---------------------------------------------------------------------------
# Phase 2: oneshot/ runs (mixed frontends/modes; routed by inspected meta.json)
# ---------------------------------------------------------------------------

# multiscale runs in oneshot/
mv_safe "oneshot/2026-04-24_16-31-36"      "runs/multiscale/vae/2026-04-24_16-31"
mv_safe "oneshot/multiscale_filesplit_v2"  "runs/multiscale/vae/filesplit_v2"
mv_safe "oneshot/2026-04-25_09-41-39"      "runs/multiscale/disentangled/2026-04-25_09-41"
mv_safe "oneshot/2026-04-24_16-17-43"      "runs/multiscale/contrastive/2026-04-24_16-17"

# morlet_per_sensor runs in oneshot/
mv_safe "oneshot/2026-04-24_18-57-29"      "runs/morlet_per_sensor/vae/2026-04-24_18-57"
mv_safe "oneshot/2026-04-25_10-06-01"      "runs/morlet_per_sensor/disentangled/2026-04-25_10-06"
mv_safe "oneshot/disentangled_smoke"       "runs/morlet_per_sensor/disentangled/smoke"

# morlet_fused runs in oneshot/ (the two CRL-less ones flagged in inventory)
mv_safe "oneshot/2026-04-24_18-57-44"      "runs/morlet_fused/vae/2026-04-24_18-57"
mv_safe "oneshot/2026-04-25_06-40-58"      "runs/morlet_fused/vae/2026-04-25_06-40_dsonly"
mv_safe "oneshot/2026-04-25_06-41-49"      "runs/morlet_fused/vae/2026-04-25_06-41_dsonly"

# ---------------------------------------------------------------------------
# Phase 3: supervised baselines (non-CRL; their own bucket)
# ---------------------------------------------------------------------------

mv_safe "supervised_multiscale_idsplit"          "runs/supervised/id_split/v1"
mv_safe "supervised_multiscale_idsplit_v2"       "runs/supervised/id_split/v2"
mv_safe "supervised_multiscale_idsplit_v2_fast"  "runs/supervised/id_split/v2_fast"
mv_safe "_diag_bs256"                            "runs/supervised/id_split/bs256_diag"
mv_safe "supervised_multiscale_filesplit_v2"     "runs/supervised/file_split/v2"

# ---------------------------------------------------------------------------
# Phase 4: legacy old-layout dirs (no crl/ds/eval substructure)
# ---------------------------------------------------------------------------

# Date-stamped frontend-only dumps from before the diagnostic layout.
mv_safe "morlet/2026-04-20_09-07-39"            "runs/morlet/vae/legacy_oldlayout/2026-04-20_09-07"
mv_safe "morlet/2026-04-20_21-23-23"            "runs/morlet/vae/legacy_oldlayout/2026-04-20_21-23"
mv_safe "multiscale/2026-04-20_09-05-07"        "runs/multiscale/vae/legacy_oldlayout/2026-04-20_09-05"
mv_safe "multiscale/2026-04-20_21-22-13"        "runs/multiscale/vae/legacy_oldlayout/2026-04-20_21-22"
mv_safe "morlet_learnable/2026-04-24_23-34-56"  "runs/morlet_learnable/vae/legacy_oldlayout/2026-04-24_23-34"
mv_safe "morlet_learnable/2026-04-24_23-35-45"  "runs/morlet_learnable/vae/legacy_oldlayout/2026-04-24_23-35"

# ---------------------------------------------------------------------------
# Phase 5: archive (smoke / sweep / experimental dirs that don't fit the layout)
# ---------------------------------------------------------------------------

mv_safe "experiments"        "runs/_archive/experiments_sweep"
mv_safe "smoke_test"         "runs/_archive/smoke_test"
mv_safe "framework_smoke"    "runs/_archive/framework_smoke"
mv_safe "_id_split_smoke"    "runs/_archive/id_split_smoke"

# ---------------------------------------------------------------------------
# Phase 6: caches/ umbrella
# ---------------------------------------------------------------------------

mv_safe "cache"     "caches/waveform"
mv_safe "id_cache"  "caches/id_split"

# ---------------------------------------------------------------------------
# Phase 7: clean up empty parent dirs left behind
# ---------------------------------------------------------------------------

cleanup_empty() {
  local d="$1"
  if [[ -d "$d" ]] && [[ -z "$(ls -A "$d")" ]]; then
    if [[ "$EXECUTE" == "1" ]]; then
      rmdir "$d"
      echo "RMDIR (empty): $d"
    else
      echo "[dry-run] rmdir $d  (empty parent)"
    fi
  fi
}

cleanup_empty "id_split"
cleanup_empty "oneshot"
cleanup_empty "morlet"
cleanup_empty "multiscale"
cleanup_empty "morlet_learnable"

# ---------------------------------------------------------------------------
# Phase 8: STATUS.md notes for runs that didn't complete the full pipeline
# ---------------------------------------------------------------------------

write_status() {
  local dir="$1"
  local body="$2"
  if [[ ! -d "$dir" ]]; then
    echo "SKIP STATUS (no dir): $dir"
    return
  fi
  local file="$dir/STATUS.md"
  if [[ -e "$file" ]]; then
    echo "SKIP STATUS (exists): $file"
    return
  fi
  if [[ "$EXECUTE" == "1" ]]; then
    printf '%s\n' "$body" > "$file"
    echo "WROTE STATUS: $file"
  else
    echo "[dry-run] write $file"
  fi
}

write_status "runs/multiscale/disentangled/run1_incomplete" \
"# Status: incomplete

Pretraining stopped at epoch 82 of 100 — never finished.
No downstream training, no eval. CRL checkpoints exist (crl_best.pth,
crl_best_aux_type.pth) but reflect a partial training schedule, not a
fully converged run. Do not compare against completed runs head-to-head."

write_status "runs/morlet_fused/vae/2026-04-25_06-40_dsonly" \
"# Status: downstream-only

This directory contains downstream + eval artifacts but no CRL checkpoint
of its own. Likely a downstream-only re-run against an external CRL
checkpoint (probable parent: runs/morlet_fused/vae/2026-04-24_18-57/).
Treat results here as a probe variation, not a fresh end-to-end run."

write_status "runs/morlet_fused/vae/2026-04-25_06-41_dsonly" \
"# Status: downstream-only

Same shape as runs/morlet_fused/vae/2026-04-25_06-40_dsonly: downstream
+ eval only, no CRL of its own. Probable parent: 2026-04-24_18-57/."

write_status "runs/multiscale/vae/v1" \
"# Status: partial — CRL + downstream, no eval

Has crl/ and downstream/ but no eval/ directory. Reported leaderboard
metrics for this run come from the eval pipeline, so do not include this
run in head-to-head F1 comparisons. Use v1_diag instead, which has the
complete eval phase against this run's checkpoint."

write_status "runs/morlet_per_sensor/vae/phase_v1" \
"# Status: partial — CRL + downstream, no eval

Has crl/ and downstream/ but no eval/ directory. Use phase_v1_diag for
the leaderboard-comparable evaluation against this run's checkpoint."

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

if [[ "$EXECUTE" == "1" ]]; then
  echo
  echo "Reorganization complete. Run 'tree -L 4 -d runs/' or 'find runs -maxdepth 4 -type d' to verify."
else
  echo
  echo "(dry-run finished. Re-run with EXECUTE=1 to apply.)"
fi
