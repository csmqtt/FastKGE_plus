#!/bin/bash
set -euo pipefail
export LANG=zh_CN.UTF-8

REPO_ROOT="/home/hwc/GraduationProject/FastKGE"
LOG_ROOT="${REPO_ROOT}/logs/ablation_quant_init"

# 保持与主训练脚本一致的基础配方，只比较层间协同与初始化策略。
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
    -entity_layering legacy
    -ent_rank_policy legacy
)

# 新方向统一保留的结构超参。
INTERLAYER_BASE=(
    -interlayer_init 0.0
    -interlayer_stopgrad on
)

# 量化残差初始化默认采用逐行 8 比特量化。
QUANT_INIT_BASE=(
    -lora_init quant_svd
    -quant_init_bits 8
    -quant_init_granularity row
)

SCHED_RECIPE=(-use_lr_scheduler -warmup_epochs 5 -min_lr_ratio 0.3 -patience 8)
BASELINE_TIGHT=(-patience 3)
P0_REL_LAYER=(-num_rel_layers 2 -rel_r 40)

ORIGINAL_STRUCTURE=()

INTERLAYER_ONLY=(
    -interlayer_lora_mode residual
)

QUANT_SVD_ONLY=(
    -interlayer_lora_mode off
    "${QUANT_INIT_BASE[@]}"
)

INTERLAYER_QUANT_SVD=(
    -interlayer_lora_mode residual
    "${QUANT_INIT_BASE[@]}"
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
        echo "ablation_quant_init.sh: 不支持的数据集 '$DS'（可选：FB_CKGE WN_CKGE ENTITY FACT HYBRID RELATION）" >&2
        exit 2
    fi
done

mkdir -p "${LOG_ROOT}"
cd "${REPO_ROOT}"

build_dataset_recipe() {
    local ds="$1"
    local out_name="$2"
    local -n extra_flags_ref="$3"
    local recipe=("${COMMON[@]}" "${INTERLAYER_BASE[@]}")

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
            echo "ablation_quant_init.sh: 不支持的数据集 '$ds'" >&2
            exit 2
            ;;
    esac

    python main.py "${recipe[@]}" "${extra_flags_ref[@]}" "${EXTRA_ARGS[@]}"
}

run_case() {
    local ds="$1"
    local out_name="$2"
    local flag_name="$3"
    echo "===== [${out_name}] ${ds} ====="
    build_dataset_recipe "$ds" "$out_name" "$flag_name"
}

echo "ablation_quant_init.sh: datasets=[${DATASETS}] extra_args=[$*]"
echo "ablation_quant_init.sh: logs at ${LOG_ROOT}"

for DS in $DATASETS; do
    run_case "$DS" "original_structure" ORIGINAL_STRUCTURE
    run_case "$DS" "interlayer_only" INTERLAYER_ONLY
    run_case "$DS" "quant_svd_only" QUANT_SVD_ONLY
    run_case "$DS" "interlayer_quant_svd" INTERLAYER_QUANT_SVD
done

echo "四组实验完成，日志目录：${LOG_ROOT}"
