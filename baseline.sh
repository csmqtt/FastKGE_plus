#!/bin/bash
# ----------------------------------------------------------------------
# baseline.sh
#   Re-run the ORIGINAL, UNMODIFIED paper code (git commit 196b9a6) on
#   all 6 datasets (FB_CKGE / WN_CKGE / ENTITY / FACT / HYBRID / RELATION)
#   and write the logs to   ./logs/baseline/<timestamp>/<DATASET>.log
#
#   How it works:
#     1) Create a git worktree at commit 196b9a6 under /tmp (keeps the
#        current HEAD tree / Python code intact for your analysis).
#     2) Symlink the big data/ folder from the main workspace so we
#        don't have to re-copy ~6GB.
#     3) Point -log_path at <repo>/logs/baseline/ so all outputs land
#        there, not in the worktree.
#     4) Run with the main.sh-equivalent baseline hyperparameters the
#        paper ships for FB/WN; the same HP are reused for the 4
#        LKGE-style datasets because the original main.sh never shipped
#        commands for them.
#
#   Safe to run multiple times: idempotent worktree + logs folder.
# ----------------------------------------------------------------------
set -euo pipefail
export LANG=zh_CN.UTF-8

REPO_ROOT="/home/hwc/GraduationProject/FastKGE"
WORKTREE="/tmp/fastkge-paper-baseline"
PAPER_COMMIT="196b9a6"
BASELINE_LOGS="${REPO_ROOT}/logs/baseline"

# --- 1. worktree at pristine paper commit ----------------------------
if [ ! -d "$WORKTREE" ]; then
    git -C "$REPO_ROOT" worktree add -f "$WORKTREE" "$PAPER_COMMIT"
fi

# --- 2. wire data / checkpoint / save ---------------------------------
[ -e "$WORKTREE/data" ]       || ln -s "$REPO_ROOT/data" "$WORKTREE/data"
mkdir -p "$WORKTREE/checkpoint" "$WORKTREE/save" "$BASELINE_LOGS"

# --- 3. run all 6 datasets with the paper's baseline HP ---------------
# Matches the original main.sh exactly for FB/WN; reused verbatim for
# ENTITY/FACT/HYBRID/RELATION so the HP is one consistent "paper recipe".
COMMON=(
    -model_name        LoraKGE_Layers
    -ent_r             150
    -rel_r             20
    -num_ent_layers    2
    -num_rel_layers    1
    -learning_rate     1e-1
    -using_various_ranks True
    -gpu               0
    -log_path          "${BASELINE_LOGS}/"
)

cd "$WORKTREE"
for DS in FB_CKGE WN_CKGE ENTITY FACT HYBRID RELATION; do
    echo "===== [paper-baseline] $DS ====="
    python main.py "${COMMON[@]}" -dataset "$DS" "$@"
done

echo ""
echo "All paper-baseline runs done."
echo "Logs:         ${BASELINE_LOGS}"
echo "Code (paper): ${WORKTREE}  (git worktree @ ${PAPER_COMMIT})"
