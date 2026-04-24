#!/bin/bash
set -euo pipefail
export LANG=zh_CN.UTF-8

REPO_ROOT="/home/hwc/GraduationProject/FastKGE"
LOG_ROOT="${REPO_ROOT}/logs/ablation_interlayer"

# 与 main.sh 保持一致的基础配方，确保本脚本只比较结构变量。
COMMON=(
    -model_name LoraKGE_Layers
    -ent_r 150
    -rel_r 20
    -num_ent_layers 2
    -num_rel_layers 1
    -gpu 0
    -learning_rate 1e-1
    -using_various_ranks True
    -loraplus_ratio 1.0
    -log_lora_stats
)

# 当前方案新增的结构超参，显式写出便于日志复现。
NEW_STRUCTURE_BASE=(
    -difficulty_rank_scale 0.5
    -difficulty_lr_scale 0.2
    -interlayer_init 0.0
    -interlayer_stopgrad on
    -log_layer_plan
)

SCHED_RECIPE=(-use_lr_scheduler -warmup_epochs 5 -min_lr_ratio 0.3 -patience 8)
BASELINE_TIGHT=(-patience 3)
P0_REL_LAYER=(-num_rel_layers 2 -rel_r 40)

INTERLAYER_ONLY=(
    -entity_layering legacy
    -ent_rank_policy legacy
    -interlayer_lora_mode residual
)

DIFFICULTY_ONLY=(
    -entity_layering difficulty
    -ent_rank_policy difficulty
    -interlayer_lora_mode off
)

BOTH_ON=(
    -entity_layering difficulty
    -ent_rank_policy difficulty
    -interlayer_lora_mode residual
)

positional_ds=""
while [[ $# -gt 0 && "$1" != -* ]]; do
    positional_ds="$positional_ds $1"
    shift
done
positional_ds="${positional_ds# }"

if [[ -n "$positional_ds" ]]; then
    DATASETS="$positional_ds"
else
    DATASETS="${DATASETS:-FB_CKGE WN_CKGE ENTITY FACT HYBRID RELATION}"
fi

EXTRA_ARGS=("$@")

valid_ds_re="^(FB_CKGE|WN_CKGE|ENTITY|FACT|HYBRID|RELATION)$"
for DS in $DATASETS; do
    if ! [[ "$DS" =~ $valid_ds_re ]]; then
        echo "ablation_interlayer.sh: 不支持的数据集 '$DS'（可选：FB_CKGE WN_CKGE ENTITY FACT HYBRID RELATION）" >&2
        exit 2
    fi
done

mkdir -p "${LOG_ROOT}"
cd "${REPO_ROOT}"

build_dataset_recipe() {
    local ds="$1"
    local out_name="$2"
    local -n extra_flags_ref="$3"
    local recipe=("${COMMON[@]}" "${NEW_STRUCTURE_BASE[@]}" "${extra_flags_ref[@]}")

    case "$ds" in
        FB_CKGE|WN_CKGE)
            recipe+=(-dataset "$ds" -lora_init dora -use_rs_lora "${SCHED_RECIPE[@]}" -log_path "${LOG_ROOT}/${out_name}/")
            ;;
        ENTITY|FACT)
            recipe+=(-dataset "$ds" "${BASELINE_TIGHT[@]}" -log_path "${LOG_ROOT}/${out_name}/")
            ;;
        HYBRID|RELATION)
            recipe+=(-dataset "$ds" "${P0_REL_LAYER[@]}" "${BASELINE_TIGHT[@]}" -log_path "${LOG_ROOT}/${out_name}/")
            ;;
        *)
            echo "ablation_interlayer.sh: 不支持的数据集 '$ds'" >&2
            exit 2
            ;;
    esac

    python main.py "${recipe[@]}" "${EXTRA_ARGS[@]}"
}

run_case() {
    local ds="$1"
    local out_name="$2"
    local flag_name="$3"
    echo "===== [${out_name}] ${ds} ====="
    build_dataset_recipe "$ds" "$out_name" "$flag_name"
}

echo "ablation_interlayer.sh: datasets=[${DATASETS}] extra_args=[$*]"
echo "ablation_interlayer.sh: logs at ${LOG_ROOT}"

for DS in $DATASETS; do
    run_case "$DS" "interlayer_only" INTERLAYER_ONLY
    run_case "$DS" "difficulty_only" DIFFICULTY_ONLY
    run_case "$DS" "interlayer_difficulty" BOTH_ON
done

echo "三组消融完成，日志目录：${LOG_ROOT}"
