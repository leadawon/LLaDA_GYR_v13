#!/usr/bin/env bash
set -euo pipefail

cd /workspace/LLaDA_GYR_v13

GPU_ID="${GPU_ID:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-256}"
LIMIT_GSM8K="${LIMIT_GSM8K:-9999}"
LIMIT_HUMANEVAL="${LIMIT_HUMANEVAL:-9999}"
# 9-tier index order from conservative -> omegaradical:
# 1,9,5,3,7 => conservative, omegaradical, very_radical, progressive, ultraradical
TIER_SEQUENCE="${TIER_SEQUENCE:-conservative,omegaradical,very_radical,progressive,ultraradical}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

FORCE_RERUN_EFFECTIVE="${FORCE_RERUN:-0}"
if [[ "${SKIP_EXISTING}" == "0" ]]; then
  FORCE_RERUN_EFFECTIVE="1"
fi

ROOT_OUT_GSM8K="${ROOT_OUT_GSM8K:-out/prob_gyr_limit9999_qkv_sweep_llada_alltiers_gpu${GPU_ID}_gsm8k_limit${LIMIT_GSM8K}}"
LOG_DIR_GSM8K="${LOG_DIR_GSM8K:-out/logs/prob_gyr_limit9999_qkv_sweep_llada_alltiers_gpu${GPU_ID}_gsm8k_limit${LIMIT_GSM8K}}"
ROOT_OUT_HUMANEVAL="${ROOT_OUT_HUMANEVAL:-out/prob_gyr_limit9999_qkv_sweep_llada_alltiers_gpu${GPU_ID}_humaneval_instruct_limit${LIMIT_HUMANEVAL}}"
LOG_DIR_HUMANEVAL="${LOG_DIR_HUMANEVAL:-out/logs/prob_gyr_limit9999_qkv_sweep_llada_alltiers_gpu${GPU_ID}_humaneval_instruct_limit${LIMIT_HUMANEVAL}}"

echo "[start-part1-limit9999] gpu=${GPU_ID} (gsm8k + humaneval, limits: gsm8k=${LIMIT_GSM8K}, humaneval=${LIMIT_HUMANEVAL})"
echo "[proxy-fixed] qk_softmax | ls=early_mid_late | hr=max | sym=max | th=0.4"
echo "[skip-existing] ${SKIP_EXISTING} (FORCE_RERUN=${FORCE_RERUN_EFFECTIVE})"
echo "[tier-sequence] ${TIER_SEQUENCE}"

run_patched_split_base() {
  local task="$1"
  local max_new_tokens="$2"
  local limit="$3"
  local root_out="$4"
  local log_dir="$5"
  local tier="$6"
  local tmp_script

  tmp_script="$(mktemp "/tmp/run_prob_gyr_limit200_qkv_sweep_split_base.sh.XXXXXX")"
  sed \
    -e 's/^PROXY_NAMES=.*/PROXY_NAMES=("qk_softmax")/' \
    -e 's/^LAYER_SETS=.*/LAYER_SETS=("early_mid_late")/' \
    -e 's/^HEAD_REDUCE_OPTS=.*/HEAD_REDUCE_OPTS=("max")/' \
    -e 's/^SYMS=.*/SYMS=("max")/' \
    -e 's/^THRESH_qk_softmax=.*/THRESH_qk_softmax=(0.4)/' \
    -e '/^keep_sweep_index() {/,/^}/c\
keep_sweep_index() {\
  return 0\
}' \
    "./run_prob_gyr_limit200_qkv_sweep_split_base.sh" > "${tmp_script}"
  chmod +x "${tmp_script}"

  GPU_ID="${GPU_ID}" \
  SHARD_INDEX="${SHARD_INDEX}" \
  SHARD_COUNT="${SHARD_COUNT}" \
  TASK="${task}" \
  MAX_NEW_TOKENS="${max_new_tokens}" \
  DIFFUSION_STEPS="${DIFFUSION_STEPS}" \
  LIMIT="${limit}" \
  ROOT_OUT="${root_out}" \
  LOG_DIR="${log_dir}" \
  ONLY_TIERS="${tier}" \
  FORCE_RERUN="${FORCE_RERUN_EFFECTIVE}" \
    bash "${tmp_script}"

  rm -f "${tmp_script}"
}

IFS=',' read -r -a TIER_LIST <<< "${TIER_SEQUENCE}"
for raw_tier in "${TIER_LIST[@]}"; do
  tier="$(echo "${raw_tier}" | xargs)"
  if [[ -z "${tier}" ]]; then
    continue
  fi

  echo "============================================================"
  echo "[tier] ${tier} -> gsm8k"
  echo "============================================================"
  run_patched_split_base "gsm8k_cot" 256 "${LIMIT_GSM8K}" "${ROOT_OUT_GSM8K}" "${LOG_DIR_GSM8K}" "${tier}"

  echo "============================================================"
  echo "[tier] ${tier} -> humaneval"
  echo "============================================================"
  run_patched_split_base "humaneval_instruct" 512 "${LIMIT_HUMANEVAL}" "${ROOT_OUT_HUMANEVAL}" "${LOG_DIR_HUMANEVAL}" "${tier}"
done

echo "[done-part1-limit9999] gpu=${GPU_ID} (gsm8k + humaneval)"
