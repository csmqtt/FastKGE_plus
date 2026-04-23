#! /bin/bash
# ---------------------------------------------------------------------------
# P1 profiling helper: attach torch.profiler to one (snapshot, epoch) pair on
# one dataset. Produces:
#   1. A chrome trace  -> logs/profile/<DATASET>_s<SNAP>_e<EPOCH>.json
#      (open at chrome://tracing or https://ui.perfetto.dev)
#   2. A top-K ops table printed to the run log under logs/profile/<run-ts>/
#
# Why profile mid-training on a late snapshot:
#   On HYBRID S3 (Plan-A log, 20260421193601) the epoch time balloons to ~670s
#   while S0 is ~87s. The suspected bottleneck is the LoRA-delta assembly cost
#   inside margin_loss (get_lora_embeddings + cat + fancy-index). Labels added
#   with record_function let us attribute time to:
#       - lora/ent_layers_forward   (num_ent_layers * A@B)
#       - lora/rel_layers_forward   (num_rel_layers * A@B)
#       - lora/assemble_full_embeddings (cat + python list + gather)
#       - lora/index_select / lora/score_fun
#       - batch/backward  batch/optimizer_step  batch/to_device
#
# Usage:
#   ./run_profile.sh HYBRID 3        # default: epoch 1, 20 batches
#   ./run_profile.sh RELATION 3 2 30 # snapshot 3, epoch 2, 30 batches
#   DATASET=HYBRID SNAP=3 EPOCH=1 ./run_profile.sh
#
# Cost: ~1 profiled epoch (~a few seconds of extra overhead) + full normal run.
# Set FAST=1 to skip past the profile epoch with tight early-stopping; total
# wall time drops from ~7 min (full HYBRID run) to ~1 min (stop training at the
# snapshot right after the profile target).
# ---------------------------------------------------------------------------
export LANG=zh_CN.UTF-8

DATASET="${1:-${DATASET:-HYBRID}}"
SNAP="${2:-${SNAP:-3}}"
EPOCH="${3:-${EPOCH:-1}}"
NBATCH="${4:-${NBATCH:-20}}"
shift $(( $# > 4 ? 4 : $# ))

# FAST mode: train just enough epochs per snapshot to reach the profile target,
# then stop via a tight patience. We still need to walk snapshots 0..SNAP-1
# because the model state matters for what gets profiled.
FAST_ARGS=""
if [[ "${FAST:-0}" = "1" ]]; then
    # epoch_num is the hard cap per snapshot. EPOCH+2 is enough to clear
    # warmup_epochs (5 default) is NOT triggered here because we keep
    # use_lr_scheduler off. Patience=1 forces stop right after the profile
    # epoch on the target snapshot.
    EP_CAP=$(( EPOCH + 2 ))
    FAST_ARGS="-epoch_num $EP_CAP -patience 1 -snapshot_num $(( SNAP + 1 ))"
    echo "run_profile.sh: FAST mode ON -> $FAST_ARGS"
fi

COMMON="-model_name LoraKGE_Layers -ent_r 150 -rel_r 20 -num_ent_layers 2 -num_rel_layers 1 -gpu 0 -learning_rate 1e-1 -using_various_ranks True"
UNIVERSAL="-loraplus_ratio 1.0 -log_lora_stats"
# Match the per-dataset recipe that main.sh uses so the profile reflects the
# actual production config (otherwise we'd be profiling a different code path
# than the one we're optimizing).
case "$DATASET" in
    FB_CKGE|WN_CKGE)
        RECIPE="-lora_init dora -use_rs_lora -use_lr_scheduler -warmup_epochs 5 -min_lr_ratio 0.3 -patience 8"
        ;;
    HYBRID|RELATION)
        RECIPE="-num_rel_layers 2 -rel_r 40 -patience 3"
        ;;
    ENTITY|FACT|*)
        RECIPE="-patience 3"
        ;;
esac

PROF_ARGS="-profile_snapshot $SNAP -profile_epoch $EPOCH -profile_num_batches $NBATCH -profile_out ./logs/profile"

echo "run_profile.sh: dataset=$DATASET snap=$SNAP epoch=$EPOCH nbatch=$NBATCH"
echo "run_profile.sh: recipe=[$RECIPE]"
echo "run_profile.sh: extra=[$*]"

python main.py $COMMON -dataset "$DATASET" $UNIVERSAL $RECIPE $FAST_ARGS $PROF_ARGS "$@"
