#!/usr/bin/env bash
set -euo pipefail

cd /workspace/LLaDA_GYR_v13

VENV_PATH="${VENV_PATH:-/workspace/venvs/real_dreamvenv/bin/activate}"
source "${VENV_PATH}"

GPU_ID="${GPU_ID:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"
# DIFFUSION_STEPS is intentionally tied to task gen_length per run.
LIMIT_GSM8K="${LIMIT_GSM8K:-9999}"
# 9-tier index order from conservative -> omegaradical:
# 1,9,5,3,7 => conservative, omegaradical, very_radical, progressive, ultraradical
TIER_SEQUENCE="${TIER_SEQUENCE:-conservative,omegaradical,very_radical,progressive,ultraradical}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
MODEL_ID="${MODEL_ID:-GSAI-ML/LLaDA-8B-Instruct}"

FORCE_RERUN_EFFECTIVE="${FORCE_RERUN:-0}"
if [[ "${SKIP_EXISTING}" == "0" ]]; then
  FORCE_RERUN_EFFECTIVE="1"
fi

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

ROOT_OUT_GSM8K="${ROOT_OUT_GSM8K:-out/prob_gyr_limit9999_qkv_sweep_llada_alltiers_gpu${GPU_ID}_gsm8k_limit${LIMIT_GSM8K}}"
LOG_DIR_GSM8K="${LOG_DIR_GSM8K:-out/logs/prob_gyr_limit9999_qkv_sweep_llada_alltiers_gpu${GPU_ID}_gsm8k_limit${LIMIT_GSM8K}}"

SCRIPT_LOG_DIR="${SCRIPT_LOG_DIR:-out/logs/prob_gyr_limit9999_qkv_sweep_llada_gpu${GPU_ID}_part1_1}"
mkdir -p "${SCRIPT_LOG_DIR}" "${ROOT_OUT_GSM8K}" "${LOG_DIR_GSM8K}"
RUN_TAG="prob_gyr_limit9999_qkv_sweep_llada_gpu${GPU_ID}_part1_1_s${SHARD_INDEX}of${SHARD_COUNT}_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${SCRIPT_LOG_DIR}/${RUN_TAG}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONUNBUFFERED=1
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export LM_EVAL_INCLUDE_PATH="${LM_EVAL_INCLUDE_PATH:-/workspace/Dream/eval_instruct/lm_eval/tasks}"
MAIN_PROCESS_PORT=$((13600 + GPU_ID))

GREEN_CONF="${GREEN_CONF:-0.0}"
YELLOW_CONF="${YELLOW_CONF:--0.1}"

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

SUMMARY_TSV="${SCRIPT_LOG_DIR}/${RUN_TAG}_summary.tsv"
echo -e "task\ttier\trun_name\tlimit\tgreen_bias\tyellow_bias\ttau\tscore\tout_dir" > "${SUMMARY_TSV}"

add_float() {
  local a="$1"
  local b="$2"
  awk -v x="${a}" -v y="${b}" 'BEGIN { printf "%.10g", (x + y) }'
}

has_results() {
  local out_dir="$1"
  if [[ "${FORCE_RERUN_EFFECTIVE}" == "1" ]]; then
    return 1
  fi
  find "${out_dir}" -type f -name "results_*.json" -print -quit 2>/dev/null | grep -q .
}

tier_params() {
  local tier="$1"
  case "${tier}" in
    conservative) echo "0.42|0.34|0.0010" ;;
    balanced) echo "0.38|0.30|0.0014" ;;
    progressive) echo "0.35|0.27|0.0018" ;;
    radical) echo "0.32|0.24|0.0022" ;;
    very_radical) echo "${VERY_RADICAL_GREEN_BIAS}|${VERY_RADICAL_YELLOW_BIAS}|${VERY_RADICAL_TAU}" ;;
    extremradical) echo "${EXTREMRADICAL_GREEN_BIAS}|${EXTREMRADICAL_YELLOW_BIAS}|${EXTREMRADICAL_TAU}" ;;
    ultraradical) echo "${ULTRARADICAL_GREEN_BIAS}|${ULTRARADICAL_YELLOW_BIAS}|${ULTRARADICAL_TAU}" ;;
    hyperradical) echo "${HYPERRADICAL_GREEN_BIAS}|${HYPERRADICAL_YELLOW_BIAS}|${HYPERRADICAL_TAU}" ;;
    omegaradical) echo "${OMEGARADICAL_GREEN_BIAS}|${OMEGARADICAL_YELLOW_BIAS}|${OMEGARADICAL_TAU}" ;;
    *) return 1 ;;
  esac
}

extract_score() {
  local out_dir="$1"
  local task="$2"
  python - <<'PY' "${out_dir}" "${task}"
import glob, json, os, sys
out_dir, task = sys.argv[1], sys.argv[2]
paths = sorted(glob.glob(os.path.join(out_dir, "**", "results_*.json"), recursive=True))
if not paths:
    print("NA")
    raise SystemExit
p = paths[-1]
try:
    d = json.load(open(p))
except Exception:
    print("NA")
    raise SystemExit
res = d.get("results", {}).get(task, {})
key_map = {
    "gsm8k_cot": "exact_match,flexible-extract",
    "humaneval_instruct": "pass@1,create_test",
    "mbpp_instruct": "pass_at_1,none",
    "ifeval": "prompt_level_strict_acc,none",
}
key = key_map.get(task)
val = res.get(key) if key else None
print("NA" if val is None else val)
PY
}

run_task_tier() {
  local task="$1"
  local max_new_tokens="$2"
  local limit="$3"
  local root_out="$4"
  local log_dir="$5"
  local tier="$6"

  local params
  if ! params="$(tier_params "${tier}")"; then
    echo "[warn] unknown tier, skip: ${tier}"
    return 0
  fi

  IFS='|' read -r green_bias yellow_bias tau <<< "${params}"

  local green_conf_eff yellow_conf_eff
  green_conf_eff="$(add_float "${GREEN_CONF}" "${green_bias}")"
  yellow_conf_eff="$(add_float "${YELLOW_CONF}" "${yellow_bias}")"

  local run_name out_dir
  run_name="v13_${tier}_qk_softmax_lsearly_mid_late_hrmax_symmax_th0p4"
  out_dir="${root_out}/${tier}/${task}/${run_name}"

  mkdir -p "${out_dir}" "${out_dir}/step_stats" "${root_out}" "${log_dir}"

  if has_results "${out_dir}"; then
    local score_skip
    score_skip="$(extract_score "${out_dir}" "${task}")"
    echo "[skip] ${task} ${run_name} (results exist, score=${score_skip})"
    echo -e "${task}\t${tier}\t${run_name}\t${limit}\t${green_bias}\t${yellow_bias}\t${tau}\t${score_skip}\t${out_dir}" >> "${SUMMARY_TSV}"
    return 0
  fi

  local task_steps
  task_steps="${max_new_tokens}"

  local proxy_cfg
  proxy_cfg="name=qk_softmax;layer_set=early_mid_late;head_reduce=max;layer_reduce=mean;threshold=0.4;sym=max;yg_mode=directed;outgoing_lambda=0.1;outgoing_metric=max;outgoing_include_green=false"

  local speed_jsonl fp_stats_json
  speed_jsonl="${out_dir}/speed/nfe_stats.jsonl"
  fp_stats_json="${out_dir}/step_stats/fp_stats.json"

  local model_args=(
    "model_path=${MODEL_ID}"
    "gen_length=${max_new_tokens}"
    "steps=${task_steps}"
    "block_length=${max_new_tokens}"
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

  echo "[run] task=${task} tier=${tier} run=${run_name} limit=${limit} gen=${max_new_tokens} steps=${task_steps}"

  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${ACCELERATE_BIN}" launch --main_process_port "${MAIN_PROCESS_PORT}" eval_llada.py \
    --model llada_dist \
    --model_args "$(IFS=,; echo "${model_args[*]}")" \
    --tasks "${task}" \
    --device cuda \
    --batch_size 1 \
    --num_fewshot 0 \
    --limit "${limit}" \
    --output_path "${out_dir}" \
    --log_samples \
    --confirm_run_unsafe_code \
    --apply_chat_template

  local score
  score="$(extract_score "${out_dir}" "${task}")"
  echo "[done] ${task} ${run_name} score=${score}"
  echo -e "${task}\t${tier}\t${run_name}\t${limit}\t${green_bias}\t${yellow_bias}\t${tau}\t${score}\t${out_dir}" >> "${SUMMARY_TSV}"
}

echo "[start-part1_1-limit9999] gpu=${GPU_ID} (gsm8k only, limit=${LIMIT_GSM8K})"
echo "[proxy-fixed] qk_softmax | ls=early_mid_late | hr=max | sym=max | th=0.4"
echo "[skip-existing] ${SKIP_EXISTING} (FORCE_RERUN=${FORCE_RERUN_EFFECTIVE})"
echo "[tier-sequence] ${TIER_SEQUENCE}"

IFS=',' read -r -a TIER_LIST <<< "${TIER_SEQUENCE}"
for raw_tier in "${TIER_LIST[@]}"; do
  tier="$(echo "${raw_tier}" | xargs)"
  if [[ -z "${tier}" ]]; then
    continue
  fi

  echo "============================================================"
  echo "[tier] ${tier} -> gsm8k"
  echo "============================================================"
  run_task_tier "gsm8k_cot" 256 "${LIMIT_GSM8K}" "${ROOT_OUT_GSM8K}" "${LOG_DIR_GSM8K}" "${tier}"
done

echo "[done-part1_1-limit9999] gpu=${GPU_ID} (gsm8k only)"
echo "[summary] ${SUMMARY_TSV}"
