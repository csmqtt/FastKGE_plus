#!/bin/bash
# ----------------------------------------------------------------------
# multiseed.sh
#   Run a training recipe across multiple random seeds so the aggregated
#   MRR is directly comparable to the paper's "average of 5 runs" protocol.
#
#   Why this exists:
#     IJCAI'24 FastKGE reports "the average of random five running times"
#     (arxiv 2407.05705 §Experimental Setup). A single-seed baseline has
#     ±0.5~1.5pp per-snap variance on ENTITY/FACT/HYBRID/RELATION, enough
#     to make the repo's 1-seed baseline look 0.7~3.0pp worse than paper
#     Table 2/3. Averaging 5 seeds is the apples-to-apples comparison.
#
#   Usage:
#     ./multiseed.sh <mode> <tag> [extra args forwarded to the inner runner]
#
#     <mode>:
#       current   -> uses main.sh (current HEAD code + current recipe)
#                    Answers: "is my optimization a real improvement, or
#                    just seed noise?"
#       baseline  -> uses baseline.sh (paper commit via git worktree,
#                    paper HP). Answers: "does 5-seed averaging close the
#                    repo-vs-paper gap?"
#
#     <tag>:
#       Short label for this experiment, e.g. "v25_p0_extA" or "paper_bl".
#       Logs end up at ./logs/multiseed/<tag>/s<seed>/<timestamp>/<ds>.log
#
#     [extra args]:
#       Forwarded verbatim to main.sh / baseline.sh. Because both scripts
#       end every `python main.py` call with "$@", argparse's last-wins
#       semantics let us override -log_path and -random_seed without
#       editing those scripts. Don't pass -random_seed here - the loop
#       below sets it per seed.
#
#   Output:
#     ./logs/multiseed/<tag>/s<seed>/<timestamp>/<dataset>.log
#     (One timestamp dir per (seed, dataset) since main.py creates a fresh
#      timestamp on each invocation.)
#
#   Collect:
#     python multiseed_collect.py <tag>
#
#   Runtime note:
#     "current" mode runs all 6 datasets per seed (main.sh iterates them).
#     At ~1500s/dataset on a 3090Ti, one full sweep is ~12h of GPU time.
#     If you only care about e.g. HYBRID+RELATION, comment out the other
#     python lines in main.sh before launching, or interrupt once the
#     datasets of interest are done - logs written so far are valid.
# ----------------------------------------------------------------------
set -euo pipefail
export LANG=zh_CN.UTF-8

if [ $# -lt 2 ]; then
    sed -n '2,/^# ---/p' "$0" | sed 's/^# \{0,1\}//' | head -n -1
    exit 2
fi

MODE="$1"
TAG="$2"
shift 2
EXTRA=("$@")

# Default seeds. Edit here if you want a different set; keeping 3407 first
# so s3407 reproduces the existing single-seed baseline bit-for-bit.
SEEDS=(3407 42 0 1 2)

REPO_ROOT="/home/hwc/GraduationProject/FastKGE"
OUT_ROOT="${REPO_ROOT}/logs/multiseed/${TAG}"

case "$MODE" in
    current)  RUNNER="${REPO_ROOT}/main.sh" ;;
    baseline) RUNNER="${REPO_ROOT}/baseline.sh" ;;
    *) echo "mode must be 'current' or 'baseline', got '$MODE'" >&2; exit 2 ;;
esac

if [ ! -x "$RUNNER" ]; then
    chmod +x "$RUNNER" || true
fi

mkdir -p "$OUT_ROOT"

# Sanity: warn if user accidentally passed -random_seed themselves.
for a in "${EXTRA[@]:-}"; do
    if [ "$a" = "-random_seed" ]; then
        echo "ERROR: don't pass -random_seed; this script sets it per seed." >&2
        exit 2
    fi
done

echo "multiseed: mode=${MODE}  tag=${TAG}  seeds=(${SEEDS[*]})  runner=${RUNNER}"
echo "multiseed: logs at ${OUT_ROOT}"

for SEED in "${SEEDS[@]}"; do
    SEED_DIR="${OUT_ROOT}/s${SEED}"
    mkdir -p "$SEED_DIR"
    echo ""
    echo "=============================================================="
    echo "  multiseed [${MODE}] tag=${TAG}  seed=${SEED}  -> ${SEED_DIR}"
    echo "=============================================================="
    # Forward overrides after EXTRA so our -log_path / -random_seed win
    # (argparse last-occurrence wins).
    "$RUNNER" "${EXTRA[@]}" -random_seed "$SEED" -log_path "${SEED_DIR}/"
done

echo ""
echo "All seeds done for tag=${TAG}."
echo "Collect with:  python multiseed_collect.py ${TAG}"
