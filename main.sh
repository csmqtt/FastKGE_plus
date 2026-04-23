#! /bin/bash
export LANG=zh_CN.UTF-8

# ---------------------------------------------------------------------------
# V2.5b: LoRA+CKGE training recipe with DATASET-ADAPTIVE scheduling.
#
# Observation (recordforstep V2.5 vs V2.5b section): warmup+cosine scheduling
# on snapshot 0 helps when the base-embedding training naturally runs long
# (FB_CKGE with noisy validation that delays baseline early-stop; WN_CKGE
# marginally). On datasets where baseline patience=3 already stops at
# convergence (ENTITY/FACT/HYBRID/RELATION), warmup eats 5 epochs at
# sub-base lr and cosine decay cuts the effective training budget, costing
# 0.7-2.4pp on S0 with no accuracy gain on S1-S4 and near-2x training time.
#
# Strategy:
#   - Large / noisy-convergence datasets (FB_CKGE, WN_CKGE): enable scheduler
#     + patience=8, exploit the S0 gain and the early-stopping cleanup.
#   - Fast-convergence datasets (ENTITY/FACT/HYBRID/RELATION): scheduler OFF,
#     patience=3 to match baseline budget exactly. Keep all other V2 changes
#     (AdamW wd=0 ≡ Adam, LoRA+ ratio control, warmup-aware stop_epoch counter,
#     log_lora_stats diagnostics, _init_nonempty safety).
# ---------------------------------------------------------------------------

# Common to every dataset (match V1.x baseline structural hyperparameters).
COMMON="-model_name LoraKGE_Layers -ent_r 150 -rel_r 20 -num_ent_layers 2 -num_rel_layers 1 -gpu 0 -learning_rate 1e-1 -using_various_ranks True"

# V2.5 flags that apply universally (no negative impact observed on any dataset):
#   -loraplus_ratio 1.0  : lr_B = lr_A; required because lr=1e-1 * 16 = 1.6 blows up AdamW
#   -log_lora_stats      : per-epoch LoRA A/B grad/param norms for diagnostic plots
UNIVERSAL="-loraplus_ratio 1.0 -log_lora_stats"

# Scheduler recipe (used only where the S0 gain is worth the cost):
#   -use_lr_scheduler -warmup_epochs 5 -min_lr_ratio 0.3 -patience 8
SCHED_RECIPE="-use_lr_scheduler -warmup_epochs 5 -min_lr_ratio 0.3 -patience 8"

# Baseline-tight recipe (no scheduler; matches baseline patience):
#   -patience 3
# NOTE: since we don't pass -use_lr_scheduler, scheduler is None → warmup-aware
#       stop_epoch counter in main.py reduces to always-increment, identical
#       to V1.x semantics. Safe.
BASELINE_TIGHT="-patience 3"

# P0 Graph Layering (rel-side): split the single rank-20 rel LoRA adapter
# into `num_rel_layers` adapters, each still getting its own ent_r-sized
# capacity. Only meaningful on datasets that introduce new relations per
# snapshot; measured per-snap new-relation count:
#   HYBRID   : 16, 49, 58, 28       ← P0 target
#   RELATION : 47, 47, 47, 47       ← P0 target
#   FACT / ENTITY / FB / WN : 0-3   ← rel-side layering is a no-op
# -rel_r 40 is chosen so each layer has rank >= 20 (old baseline per-rel capacity)
# while keeping the total rel parameter count within 2x the baseline.
#
# 2026-04-21 efficiency pass (rollback of Plan A / patience=6):
# The first Plan-A run (20260421193601) showed Whole_MRR@S4 = 0.198, identical
# to P0-only at patience=3 (within noise); the second Plan-A run got 0.196.
# Meanwhile Plan-A added ~400s (HYBRID S3 alone: 34 epochs vs patience=3's ~21
# epochs, +13 epochs x 21 s = +273 s; S1 adds another 3 wasted epochs). Best
# MRR per snapshot was reached well before the extended patience activated, so
# those extra epochs are pure waste. We revert to patience=3 here and keep P0.
#
# Expected on HYBRID with this recipe: Whole_MRR@S4 ~0.198, total ~1050 s
# (was 1303 s with profile on, 1547 s Plan-A first run).
P0_REL_LAYER="-num_rel_layers 2 -rel_r 40"

# Dataset filter. Three equivalent ways to restrict which datasets run:
#   1) Positional args (must come BEFORE any flag):
#        ./main.sh HYBRID                           # one dataset
#        ./main.sh HYBRID RELATION                  # subset
#        ./main.sh HYBRID -patience 5               # subset + extra CLI args
#   2) Env var:
#        DATASETS="HYBRID RELATION" ./main.sh
#   3) Nothing -> run all 6 datasets (paper setup).
# Implementation: consume leading non-flag positional args as the dataset list;
# remaining args (starting with `-`) are forwarded to python via "$@".
positional_ds=""
while [[ $# -gt 0 && "$1" != -* ]]; do
    positional_ds="$positional_ds $1"
    shift
done
positional_ds="${positional_ds# }"  # trim leading space

if [[ -n "$positional_ds" ]]; then
    DATASETS="$positional_ds"
else
    DATASETS="${DATASETS:-FB_CKGE WN_CKGE ENTITY FACT HYBRID RELATION}"
fi

# Validate: warn early on typos instead of silently running nothing.
valid_ds_re="^(FB_CKGE|WN_CKGE|ENTITY|FACT|HYBRID|RELATION)$"
for DS in $DATASETS; do
    if ! [[ "$DS" =~ $valid_ds_re ]]; then
        echo "main.sh: ERROR unknown dataset '$DS' (valid: FB_CKGE WN_CKGE ENTITY FACT HYBRID RELATION)" >&2
        exit 2
    fi
done

echo "main.sh: running datasets=[${DATASETS}] extra_args=[$*]"

for DS in $DATASETS; do
    case "$DS" in
        FB_CKGE|WN_CKGE)
            # DoRA init + rank-stabilized scaling + warmup/cosine scheduler.
            python main.py $COMMON -dataset "$DS" -lora_init dora -use_rs_lora $UNIVERSAL $SCHED_RECIPE "$@"
            ;;
        ENTITY|FACT)
            # Plain xavier init; no new relations per snapshot so P0 is a no-op.
            python main.py $COMMON -dataset "$DS" $UNIVERSAL $BASELINE_TIGHT "$@"
            ;;
        HYBRID|RELATION)
            # P0 rel-side Graph Layering (keep), patience=3 baseline (restored).
            # Plan A's patience=6 was rolled back: it added ~400 s on HYBRID
            # without improving Whole_MRR@S4 above the P0-only number.
            python main.py $COMMON -dataset "$DS" $UNIVERSAL $P0_REL_LAYER $BASELINE_TIGHT "$@"
            ;;
        *)
            echo "main.sh: unknown dataset '$DS' - skipping" >&2
            ;;
    esac
done
