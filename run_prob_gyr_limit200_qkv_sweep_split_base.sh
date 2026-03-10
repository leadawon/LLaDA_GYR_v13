#!/usr/bin/env bash
set -euo pipefail

# Probabilistic GYR limit200 sweep for LLaDA (sharded base)
# - conservative -> omegaradical tiers
# - proxy_qk(qk_log_softmax/qk_softmax/qk_raw) sweep
# - shard rule: job_index % SHARD_COUNT == SHARD_INDEX

cd /workspace/LLaDA_GYR_v13

VENV_PATH=${VENV_PATH:-/workspace/venvs/real_dreamvenv/bin/activate}
source "${VENV_PATH}"

GPU_ID="${GPU_ID:?GPU_ID is required (e.g. 0)}"
TASK="${TASK:?TASK is required (e.g. gsm8k_cot)}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:?MAX_NEW_TOKENS is required}"
LIMIT="${LIMIT:?LIMIT is required}"
SHARD_INDEX="${SHARD_INDEX:?SHARD_INDEX is required (0-based)}"
SHARD_COUNT="${SHARD_COUNT:-1}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-${MAX_NEW_TOKENS}}"
ONLY_TIERS="${ONLY_TIERS:-}"
MODEL_ID="${MODEL_ID:-GSAI-ML/LLaDA-8B-Instruct}"

if ! [[ "${GPU_ID}" =~ ^[0-9]+$ ]]; then
  echo "[error] GPU_ID must be an integer: ${GPU_ID}"
  exit 1
fi
if ! [[ "${SHARD_INDEX}" =~ ^[0-9]+$ ]]; then
  echo "[error] SHARD_INDEX must be an integer: ${SHARD_INDEX}"
  exit 1
fi
if ! [[ "${SHARD_COUNT}" =~ ^[0-9]+$ ]] || (( SHARD_COUNT <= 0 )); then
  echo "[error] SHARD_COUNT must be a positive integer: ${SHARD_COUNT}"
  exit 1
fi
if (( SHARD_INDEX >= SHARD_COUNT )); then
  echo "[error] SHARD_INDEX must be < SHARD_COUNT: ${SHARD_INDEX} vs ${SHARD_COUNT}"
  exit 1
fi

if command -v accelerate >/dev/null 2>&1; then
  ACCELERATE_BIN="$(command -v accelerate)"
elif [[ -x /workspace/venvs/real_dreamvenv/bin/accelerate ]]; then
  ACCELERATE_BIN="/workspace/venvs/real_dreamvenv/bin/accelerate"
else
  echo "[error] accelerate binary not found"
  exit 127
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONUNBUFFERED=1
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export LM_EVAL_INCLUDE_PATH="${LM_EVAL_INCLUDE_PATH:-/workspace/Dream/eval_instruct/lm_eval/tasks}"

MAIN_PROCESS_PORT=$((13600 + GPU_ID))

GREEN_CONF="${GREEN_CONF:-0.0}"
YELLOW_CONF="${YELLOW_CONF:--0.1}"
# LLaDA calibration:
# - Promote current conservative baseline to very_radical.
# - Refill conservative->radical with stricter (more conservative) settings.
VERY_RADICAL_GREEN_BIAS="${VERY_RADICAL_GREEN_BIAS:-0.30}"
VERY_RADICAL_YELLOW_BIAS="${VERY_RADICAL_YELLOW_BIAS:-0.22}"
VERY_RADICAL_TAU="${VERY_RADICAL_TAU:-0.0025}"
EXTREMRADICAL_GREEN_BIAS="${EXTREMRADICAL_GREEN_BIAS:-0.03}"
EXTREMRADICAL_YELLOW_BIAS="${EXTREMRADICAL_YELLOW_BIAS:-0.015}"
EXTREMRADICAL_TAU="${EXTREMRADICAL_TAU:-0.024}"
ULTRARADICAL_GREEN_BIAS="${ULTRARADICAL_GREEN_BIAS:-0.025}"
ULTRARADICAL_YELLOW_BIAS="${ULTRARADICAL_YELLOW_BIAS:-0.010}"
ULTRARADICAL_TAU="${ULTRARADICAL_TAU:-0.026}"
HYPERRADICAL_GREEN_BIAS="${HYPERRADICAL_GREEN_BIAS:-0.018}"
HYPERRADICAL_YELLOW_BIAS="${HYPERRADICAL_YELLOW_BIAS:-0.005}"
HYPERRADICAL_TAU="${HYPERRADICAL_TAU:-0.028}"
OMEGARADICAL_GREEN_BIAS="${OMEGARADICAL_GREEN_BIAS:-0.01}"
OMEGARADICAL_YELLOW_BIAS="${OMEGARADICAL_YELLOW_BIAS:-0.00}"
OMEGARADICAL_TAU="${OMEGARADICAL_TAU:-0.03}"

TIERS=(
  "conservative|0.42|0.34|0.0010"
  "balanced|0.38|0.30|0.0014"
  "progressive|0.35|0.27|0.0018"
  "radical|0.32|0.24|0.0022"
  "very_radical|${VERY_RADICAL_GREEN_BIAS}|${VERY_RADICAL_YELLOW_BIAS}|${VERY_RADICAL_TAU}"
  "extremradical|${EXTREMRADICAL_GREEN_BIAS}|${EXTREMRADICAL_YELLOW_BIAS}|${EXTREMRADICAL_TAU}"
  "ultraradical|${ULTRARADICAL_GREEN_BIAS}|${ULTRARADICAL_YELLOW_BIAS}|${ULTRARADICAL_TAU}"
  "hyperradical|${HYPERRADICAL_GREEN_BIAS}|${HYPERRADICAL_YELLOW_BIAS}|${HYPERRADICAL_TAU}"
  "omegaradical|${OMEGARADICAL_GREEN_BIAS}|${OMEGARADICAL_YELLOW_BIAS}|${OMEGARADICAL_TAU}"
)

PROXY_NAMES=("qk_log_softmax" "qk_softmax" "qk_raw")
LAYER_SETS=("last_only" "mid_only" "early_mid_late")
HEAD_REDUCE_OPTS=("mean" "max")
SYMS=("max")

THRESH_qk_log_softmax=(0.2 0.3)
THRESH_qk_softmax=(0.4 0.5)
THRESH_qk_raw=(0.5 0.6)

ROOT_OUT="${ROOT_OUT:-out/prob_gyr_limit200_qkv_sweep_llada_${TASK}_gpu${GPU_ID}_limit${LIMIT}}"
LOG_DIR="${LOG_DIR:-out/logs/prob_gyr_limit200_qkv_sweep_llada_${TASK}_gpu${GPU_ID}_limit${LIMIT}}"
mkdir -p "${ROOT_OUT}" "${LOG_DIR}"

RUN_TAG="prob_gyr_limit200_qkv_sweep_llada_${TASK}_g${GPU_ID}_s${SHARD_INDEX}of${SHARD_COUNT}_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/${RUN_TAG}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

SUMMARY_TSV="${ROOT_OUT}/${RUN_TAG}_summary.tsv"
echo -e "run_name\ttier\tmethod\tproxy_name\tlayer_set\thead_reduce\tsym\tthreshold\tgreen_bias\tyellow_bias\ttau\tgreen_conf_eff\tyellow_conf_eff\tflex_em\tavg_forward_pass\tefficiency\tout_dir" > "${SUMMARY_TSV}"

sanitize_float() {
  local v="$1"
  echo "${v}" | sed -e 's/-/m/g' -e 's/\./p/g'
}

add_float() {
  local a="$1"
  local b="$2"
  awk -v x="${a}" -v y="${b}" 'BEGIN { printf "%.10g", (x + y) }'
}

has_results() {
  local out_dir="$1"
  if [[ "${FORCE_RERUN:-0}" == "1" ]]; then
    return 1
  fi
  find "${out_dir}" -type f -name 'results_*.json' -print -quit 2>/dev/null | grep -q .
}

is_my_shard() {
  local idx="$1"
  (( idx % SHARD_COUNT == SHARD_INDEX ))
}

keep_sweep_index() {
  local idx="$1"
  # Keep 1st, 3rd, 5th... sweep entries (1-based indexing).
  (( idx % 2 == 0 ))
}

tier_selected() {
  local tier="$1"
  if [[ -z "${ONLY_TIERS}" ]]; then
    return 0
  fi
  local item
  IFS=',' read -r -a items <<< "${ONLY_TIERS}"
  for item in "${items[@]}"; do
    item="$(echo "${item}" | xargs)"
    if [[ -n "${item}" && "${item}" == "${tier}" ]]; then
      return 0
    fi
  done
  return 1
}

extract_avg_fp() {
  local fp_json="$1"
  python - <<'PY' "${fp_json}"
import json,sys
p=sys.argv[1]
try:
    with open(p, 'r', encoding='utf-8') as f:
        d=json.load(f)
    v=d.get('avg_forward_passes', d.get('avg_nfe', None))
    if v is None:
        print('FAIL')
    else:
        print(float(v))
except Exception:
    print('FAIL')
PY
}

run_one() {
  local tier="$1"
  local green_bias="$2"
  local yellow_bias="$3"
  local tau="$4"
  local proxy_name="$5"
  local layer_set="$6"
  local head_reduce="$7"
  local sym="$8"
  local threshold="$9"

  local run_name out_dir thr_tag
  thr_tag=$(sanitize_float "${threshold}")
  run_name="v13_${tier}_${proxy_name}_ls${layer_set}_hr${head_reduce}_sym${sym}_th${thr_tag}"
  out_dir="${ROOT_OUT}/${tier}/${TASK}/${run_name}"

  local green_conf_eff yellow_conf_eff
  green_conf_eff=$(add_float "${GREEN_CONF}" "${green_bias}")
  yellow_conf_eff=$(add_float "${YELLOW_CONF}" "${yellow_bias}")

  local speed_jsonl fp_stats_json
  speed_jsonl="${out_dir}/speed/nfe_stats.jsonl"
  fp_stats_json="${out_dir}/step_stats/fp_stats.json"

  mkdir -p "${out_dir}" "$(dirname "${speed_jsonl}")" "$(dirname "${fp_stats_json}")"

  if has_results "${out_dir}"; then
    echo "[skip] ${run_name} (results exist)"
    local score_skip avg_fp_skip eff_skip
    score_skip=$(python tools/check_flex_em.py "${out_dir}" --task "${TASK}" --quiet 2>/dev/null || echo "FAIL")
    avg_fp_skip=$(extract_avg_fp "${fp_stats_json}")
    eff_skip=$(python - <<'PY' "${score_skip}" "${avg_fp_skip}"
import sys
try:
    s=float(sys.argv[1]); f=float(sys.argv[2]); print(s/f if f>0 else 'FAIL')
except Exception:
    print('FAIL')
PY
)
    echo -e "${run_name}\t${tier}\tproxy_qk\t${proxy_name}\t${layer_set}\t${head_reduce}\t${sym}\t${threshold}\t${green_bias}\t${yellow_bias}\t${tau}\t${green_conf_eff}\t${yellow_conf_eff}\t${score_skip}\t${avg_fp_skip}\t${eff_skip}\t${out_dir}" >> "${SUMMARY_TSV}"
    return 0
  fi

  local proxy_cfg
  proxy_cfg="name=${proxy_name};layer_set=${layer_set};head_reduce=${head_reduce};layer_reduce=mean;threshold=${threshold};sym=${sym};yg_mode=directed;outgoing_lambda=0.1;outgoing_metric=max;outgoing_include_green=false"

  local model_args=(
    "model_path=${MODEL_ID}"
    "gen_length=${MAX_NEW_TOKENS}"
    "steps=${DIFFUSION_STEPS}"
    "block_length=${MAX_NEW_TOKENS}"
    "temperature=0.1"
    "top_p=0.9"
    "alg=entropy"
    "show_speed=True"
    "outp_path=${speed_jsonl}"
    "fp_stats_path=${fp_stats_json}"
    "apply_chat_template=True"

    "enable_green_red_policy=1"
    "enable_yellow_policy=1"
    "green_conf_thresh=${green_conf_eff}"
    "yellow_conf_thresh=${yellow_conf_eff}"
    "enable_probabilistic_gyr_policy=1"
    "prob_gyr_tau=${tau}"
    "yellow_dependency_method=proxy_qk"
    "yellow_proxy_cfg=${proxy_cfg}"
    "enable_early_stop_when_no_mask=1"
    "early_stop_only_when_gr_enabled=0"
  )

  echo "[run] ${run_name}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${ACCELERATE_BIN}" launch --main_process_port "${MAIN_PROCESS_PORT}" eval_llada.py \
    --model llada_dist \
    --model_args "$(IFS=,; echo "${model_args[*]}")" \
    --tasks "${TASK}" \
    --device cuda \
    --batch_size 1 \
    --num_fewshot 0 \
    --limit "${LIMIT}" \
    --output_path "${out_dir}" \
    --log_samples \
    --confirm_run_unsafe_code \
    --apply_chat_template

  local score avg_fp eff
  score=$(python tools/check_flex_em.py "${out_dir}" --task "${TASK}" --quiet 2>/dev/null || echo "FAIL")
  avg_fp=$(extract_avg_fp "${fp_stats_json}")
  eff=$(python - <<'PY' "${score}" "${avg_fp}"
import sys
try:
    s=float(sys.argv[1]); f=float(sys.argv[2]); print(s/f if f>0 else 'FAIL')
except Exception:
    print('FAIL')
PY
)

  echo "[done] ${run_name} flex_em=${score} avg_fp=${avg_fp} efficiency=${eff}"
  echo -e "${run_name}\t${tier}\tproxy_qk\t${proxy_name}\t${layer_set}\t${head_reduce}\t${sym}\t${threshold}\t${green_bias}\t${yellow_bias}\t${tau}\t${green_conf_eff}\t${yellow_conf_eff}\t${score}\t${avg_fp}\t${eff}\t${out_dir}" >> "${SUMMARY_TSV}"
}

echo "[start] ${RUN_TAG}"
echo "[info] model=${MODEL_ID} task=${TASK} gpu=${CUDA_VISIBLE_DEVICES} shard=${SHARD_INDEX}/${SHARD_COUNT}"
echo "[info] max_new_tokens=${MAX_NEW_TOKENS} diffusion_steps=${DIFFUSION_STEPS} limit=${LIMIT}"
if [[ -n "${ONLY_TIERS}" ]]; then
  echo "[info] only_tiers=${ONLY_TIERS}"
fi

JOB_INDEX=0
SHARD_JOB_COUNT=0
SHARD_KEPT_JOB_COUNT=0

for T in "${TIERS[@]}"; do
  IFS='|' read -r TIER GREEN_BIAS YELLOW_BIAS TAU <<< "${T}"

  if ! tier_selected "${TIER}"; then
    continue
  fi

  echo ""
  echo "================ Tier: ${TIER} (tau=${TAU}) ================"

  for NAME in "${PROXY_NAMES[@]}"; do
    THR_VAR="THRESH_${NAME}[@]"
    THR_LIST=("${!THR_VAR}")

    for LSET in "${LAYER_SETS[@]}"; do
      for HR in "${HEAD_REDUCE_OPTS[@]}"; do
        for SYM in "${SYMS[@]}"; do
          for THR in "${THR_LIST[@]}"; do
            if keep_sweep_index "${JOB_INDEX}" && is_my_shard "${JOB_INDEX}"; then
              SHARD_KEPT_JOB_COUNT=$((SHARD_KEPT_JOB_COUNT + 1))
              SHARD_JOB_COUNT=$((SHARD_JOB_COUNT + 1))
              run_one "${TIER}" "${GREEN_BIAS}" "${YELLOW_BIAS}" "${TAU}" \
                "${NAME}" "${LSET}" "${HR}" "${SYM}" "${THR}"
            fi
            JOB_INDEX=$((JOB_INDEX + 1))
          done
        done
      done
    done
  done
done

echo ""
echo "[info] total_jobs=${JOB_INDEX}, shard_jobs=${SHARD_JOB_COUNT}, shard_kept_jobs=${SHARD_KEPT_JOB_COUNT}"
echo "[summary] ${SUMMARY_TSV}"
python3 - <<'PY' "${SUMMARY_TSV}"
import csv, sys
p = sys.argv[1]
rows = []
with open(p, newline='', encoding='utf-8') as f:
    r = csv.DictReader(f, delimiter='\t')
    for row in r:
        try:
            score = float(row['flex_em'])
            avg_fp = float(row['avg_forward_pass'])
            eff = score / avg_fp if avg_fp > 0 else -1.0
        except Exception:
            continue
        rows.append((eff, score, avg_fp, row))
rows.sort(key=lambda x: x[0], reverse=True)
print('[top10-efficiency]')
for eff, score, avg_fp, row in rows[:10]:
    print(f"{eff:.6f}\tflex={score:.4f}\tavg_fp={avg_fp:.3f}\t{row['run_name']}")
PY

echo "[done]"
