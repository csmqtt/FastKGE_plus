#!/bin/bash
set -euo pipefail
export LANG=zh_CN.UTF-8

REPO_ROOT="/home/hwc/GraduationProject/FastKGE"

extract_precision() {
    local precision="fp64"
    local args=("$@")
    local idx=0
    while [[ $idx -lt ${#args[@]} ]]; do
        if [[ "${args[$idx]}" == "-precision" && $((idx + 1)) -lt ${#args[@]} ]]; then
            precision="${args[$((idx + 1))]}"
            break
        fi
        idx=$((idx + 1))
    done
    echo "${precision}"
}

# 与 main.sh 保持一致的基础配方，只比较是否启用层间协作。
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

INTERLAYER_BASE=(
    -interlayer_init 0.0
    -interlayer_stopgrad on
)

SCHED_RECIPE=(-use_lr_scheduler -warmup_epochs 5 -min_lr_ratio 0.3 -patience 8)
BASELINE_TIGHT=(-patience 3)
P0_REL_LAYER=(-num_rel_layers 2 -rel_r 40)

ORIGINAL_STRUCTURE=(
    -interlayer_lora_mode off
)

INTERLAYER_ONLY=(
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
PRECISION_TAG="$(extract_precision "${EXTRA_ARGS[@]}")"
LOG_ROOT="${REPO_ROOT}/logs/ablation_interlayer_efficiency/${PRECISION_TAG}"
SAVE_ROOT="${REPO_ROOT}/checkpoint/ablation_interlayer_efficiency/${PRECISION_TAG}"

valid_ds_re="^(FB_CKGE|WN_CKGE|ENTITY|FACT|HYBRID|RELATION)$"
for DS in $DATASETS; do
    if ! [[ "$DS" =~ $valid_ds_re ]]; then
        echo "ablation_interlayer_efficiency.sh: 不支持的数据集 '$DS'（可选：FB_CKGE WN_CKGE ENTITY FACT HYBRID RELATION）" >&2
        exit 2
    fi
done

mkdir -p "${LOG_ROOT}" "${SAVE_ROOT}"
cd "${REPO_ROOT}"

build_dataset_recipe() {
    local ds="$1"
    local out_name="$2"
    local -n extra_flags_ref="$3"
    local recipe=(
        "${COMMON[@]}"
        "${INTERLAYER_BASE[@]}"
        "${extra_flags_ref[@]}"
        -log_path "${LOG_ROOT}/${out_name}/"
        -save_path "${SAVE_ROOT}/${out_name}/"
    )

    case "$ds" in
        FB_CKGE|WN_CKGE)
            recipe+=(-dataset "$ds" -lora_init dora -use_rs_lora "${SCHED_RECIPE[@]}")
            ;;
        ENTITY|FACT)
            recipe+=(-dataset "$ds" "${BASELINE_TIGHT[@]}")
            ;;
        HYBRID|RELATION)
            recipe+=(-dataset "$ds" "${P0_REL_LAYER[@]}" "${BASELINE_TIGHT[@]}")
            ;;
        *)
            echo "ablation_interlayer_efficiency.sh: 不支持的数据集 '$ds'" >&2
            exit 2
            ;;
    esac

    python main.py "${recipe[@]}" "${EXTRA_ARGS[@]}"
}

run_case() {
    local ds="$1"
    local out_name="$2"
    local flag_name="$3"
    mkdir -p "${LOG_ROOT}/${out_name}" "${SAVE_ROOT}/${out_name}"
    echo "===== [${out_name}] ${ds} precision=${PRECISION_TAG} ====="
    build_dataset_recipe "$ds" "$out_name" "$flag_name"
}

echo "ablation_interlayer_efficiency.sh: datasets=[${DATASETS}] precision=[${PRECISION_TAG}] extra_args=[$*]"
echo "ablation_interlayer_efficiency.sh: logs at ${LOG_ROOT}"
echo "ablation_interlayer_efficiency.sh: checkpoints at ${SAVE_ROOT}"

for DS in $DATASETS; do
    run_case "$DS" "original_structure" ORIGINAL_STRUCTURE
    run_case "$DS" "interlayer_only" INTERLAYER_ONLY
done

echo "两组效率对照完成，日志目录：${LOG_ROOT}"
