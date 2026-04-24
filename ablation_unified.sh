#!/bin/bash
set -euo pipefail
export LANG=zh_CN.UTF-8

REPO_ROOT="/home/hwc/GraduationProject/FastKGE"
LOG_ROOT="${REPO_ROOT}/logs/ablation_unified"
DATASETS="${DATASETS:-ENTITY HYBRID RELATION}"

# 统一结构公共参数。
UNIFIED_COMMON="-model_name LoraKGE_Layers -adapter_type svd_gate -ent_r 150 -rel_r 20 -num_ent_layers 2 -num_rel_layers 2 -middle_r 8 -gate_hidden_dim 8 -gpu 0 -learning_rate 1e-1 -using_various_ranks True"
LEGACY_COMMON="-model_name LoraKGE_Layers -ent_r 150 -rel_r 20 -num_ent_layers 2 -num_rel_layers 1 -gpu 0 -learning_rate 1e-1 -using_various_ranks True"
SCHED_RECIPE="-use_lr_scheduler -warmup_epochs 5 -min_lr_ratio 0.3 -patience 8"
BASELINE_TIGHT="-patience 3"

mkdir -p "${LOG_ROOT}"
cd "${REPO_ROOT}"

run_legacy_nowbest() {
    local ds="$1"
    case "$ds" in
        FB_CKGE|WN_CKGE)
            python main.py ${LEGACY_COMMON} -dataset "$ds" -adapter_type legacy -lora_init dora -use_rs_lora ${SCHED_RECIPE} -log_path "${LOG_ROOT}/legacy_nowbest/"
            ;;
        ENTITY|FACT)
            python main.py ${LEGACY_COMMON} -dataset "$ds" -adapter_type legacy ${BASELINE_TIGHT} -log_path "${LOG_ROOT}/legacy_nowbest/"
            ;;
        HYBRID|RELATION)
            python main.py ${LEGACY_COMMON} -dataset "$ds" -adapter_type legacy -num_rel_layers 2 -rel_r 40 ${BASELINE_TIGHT} -log_path "${LOG_ROOT}/legacy_nowbest/"
            ;;
        *)
            echo "ablation_unified.sh: 不支持的数据集 ${ds}" >&2
            exit 2
            ;;
    esac
}

run_unified_case() {
    local ds="$1"
    local gate_strategy="$2"
    local out_dir="$3"
    case "$ds" in
        FB_CKGE|WN_CKGE)
            python main.py ${UNIFIED_COMMON} -dataset "$ds" -gate_strategy "${gate_strategy}" ${SCHED_RECIPE} -log_path "${LOG_ROOT}/${out_dir}/"
            ;;
        ENTITY|FACT|HYBRID|RELATION)
            python main.py ${UNIFIED_COMMON} -dataset "$ds" -gate_strategy "${gate_strategy}" ${BASELINE_TIGHT} -log_path "${LOG_ROOT}/${out_dir}/"
            ;;
        *)
            echo "ablation_unified.sh: 不支持的数据集 ${ds}" >&2
            exit 2
            ;;
    esac
}

echo "ablation_unified.sh: datasets=[${DATASETS}]"
for DS in $DATASETS; do
    echo "===== [legacy_nowbest] ${DS} ====="
    run_legacy_nowbest "$DS"
    echo "===== [unified_no_gate] ${DS} ====="
    run_unified_case "$DS" "off" "unified_no_gate"
    echo "===== [unified_gate] ${DS} ====="
    run_unified_case "$DS" "on" "unified_gate"
done

echo "消融完成，日志目录：${LOG_ROOT}"
