#!/usr/bin/env bash
set -euo pipefail

# LLaDA-Instruct lm-eval harness runner
# Full evaluation preset: GPU 0 + limit 9999

GPU_ID=0
LIMIT=9999
BASE_PORT="${BASE_PORT:-12640}"
TASKS="${TASKS:-gsm8k_cot humaneval_instruct mbpp_instruct ifeval}"
STEP_RATIOS="${STEP_RATIOS:-1.0 0.75 0.5 0.25 0.125}"
STEP_MIN="${STEP_MIN:-32}"

MODEL_ID="${MODEL_ID:-GSAI-ML/LLaDA-8B-Instruct}"
OUT_ROOT="${OUT_ROOT:-output_llada_instruct_lmeval_limit${LIMIT}}"
ACCELERATE_BIN="${ACCELERATE_BIN:-}"

if [[ -z "${ACCELERATE_BIN}" ]]; then
  if command -v accelerate >/dev/null 2>&1; then
    ACCELERATE_BIN="$(command -v accelerate)"
  elif [[ -x /workspace/venvs/real_dreamvenv/bin/accelerate ]]; then
    ACCELERATE_BIN="/workspace/venvs/real_dreamvenv/bin/accelerate"
  elif [[ -x /workspace/venvs/klassvenv/bin/accelerate ]]; then
    ACCELERATE_BIN="/workspace/venvs/klassvenv/bin/accelerate"
  else
    echo "Could not find accelerate binary. Set ACCELERATE_BIN=/path/to/accelerate" >&2
    exit 127
  fi
fi

# Reuse Dream custom task definitions.
export LM_EVAL_INCLUDE_PATH="${LM_EVAL_INCLUDE_PATH:-/workspace/Dream/eval_instruct/lm_eval/tasks}"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export PYTHONPATH=.

task_max_new_tokens () {
  case "$1" in
    gsm8k_cot) echo 256 ;;
    humaneval_instruct) echo 512 ;;
    mbpp_instruct) echo 768 ;;
    ifeval) echo 768 ;;
    *)
      echo "Unknown task: $1" >&2
      return 2
      ;;
  esac
}

run_one () {
  local task="$1"
  local max_new_tokens="$2"
  local diffusion_steps="$3"
  local port="$4"

  local run_dir="${OUT_ROOT}/${task}/step_${diffusion_steps}"
  local speed_jsonl="${run_dir}/speed/nfe_stats.jsonl"
  local fp_stats_json="${run_dir}/step_stats/fp_stats.json"

  mkdir -p "${run_dir}" "$(dirname "${speed_jsonl}")" "$(dirname "${fp_stats_json}")"

  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${ACCELERATE_BIN}" launch --main_process_port "${port}" eval_llada.py \
    --model llada_dist \
    --model_args model_path=${MODEL_ID},gen_length=${max_new_tokens},steps=${diffusion_steps},block_length=${max_new_tokens},temperature=0.1,top_p=0.9,alg=entropy,show_speed=True,outp_path=${speed_jsonl},fp_stats_path=${fp_stats_json},apply_chat_template=True \
    --tasks "${task}" \
    --device cuda \
    --batch_size 1 \
    --num_fewshot 0 \
    --limit "${LIMIT}" \
    --output_path "${run_dir}" \
    --log_samples \
    --confirm_run_unsafe_code \
    --apply_chat_template
}

step_schedule_for_task () {
  local max_new_tokens="$1"
  local scheduled=()
  local ratio

  for ratio in ${STEP_RATIOS}; do
    local step
    step="$(awk -v m="${max_new_tokens}" -v r="${ratio}" 'BEGIN { v = int(m * r); if (v < 1) v = 1; print v }')"
    if (( step > STEP_MIN )); then
      step=$(( (step / 32) * 32 ))
    fi
    if (( step < STEP_MIN )); then
      step="${STEP_MIN}"
    fi
    if (( step > max_new_tokens )); then
      step="${max_new_tokens}"
    fi

    if [[ ! " ${scheduled[*]} " =~ [[:space:]]${step}[[:space:]] ]]; then
      scheduled+=("${step}")
    fi
  done

  if [[ ${#scheduled[@]} -eq 0 ]]; then
    scheduled=("${max_new_tokens}")
  fi
  echo "${scheduled[*]}"
}

idx=0
for task in ${TASKS}; do
  max_new_tokens="$(task_max_new_tokens "${task}")"
  for diffusion_steps in $(step_schedule_for_task "${max_new_tokens}"); do
    run_one "${task}" "${max_new_tokens}" "${diffusion_steps}" $((BASE_PORT + idx))
    idx=$((idx + 1))
  done
done

echo "Completed. Outputs under: ${OUT_ROOT}"
