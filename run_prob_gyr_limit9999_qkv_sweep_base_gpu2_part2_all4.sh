#!/usr/bin/env bash
set -euo pipefail

cd /workspace/LLaDA_GYR_v13

GPU_ID="${GPU_ID:-2}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-256}"
LIMIT_MBPP="${LIMIT_MBPP:-9999}"
LIMIT_IFEVAL="${LIMIT_IFEVAL:-9999}"
# 9-tier index order from conservative -> omegaradical:
# 1,9,5,3,7 => conservative, omegaradical, very_radical, progressive, ultraradical
TIER_SEQUENCE="${TIER_SEQUENCE:-conservative,omegaradical,very_radical,progressive,ultraradical}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

FORCE_RERUN_EFFECTIVE="${FORCE_RERUN:-0}"
if [[ "${SKIP_EXISTING}" == "0" ]]; then
  FORCE_RERUN_EFFECTIVE="1"
fi

ROOT_OUT_MBPP="${ROOT_OUT_MBPP:-out/prob_gyr_limit9999_qkv_sweep_llada_alltiers_gpu${GPU_ID}_mbpp_instruct_limit${LIMIT_MBPP}}"
LOG_DIR_MBPP="${LOG_DIR_MBPP:-out/logs/prob_gyr_limit9999_qkv_sweep_llada_alltiers_gpu${GPU_ID}_mbpp_instruct_limit${LIMIT_MBPP}}"
ROOT_OUT_IFEVAL="${ROOT_OUT_IFEVAL:-out/prob_gyr_limit9999_qkv_sweep_llada_alltiers_gpu${GPU_ID}_ifeval_limit${LIMIT_IFEVAL}}"
LOG_DIR_IFEVAL="${LOG_DIR_IFEVAL:-out/logs/prob_gyr_limit9999_qkv_sweep_llada_alltiers_gpu${GPU_ID}_ifeval_limit${LIMIT_IFEVAL}}"

echo "[start-part2-limit9999] gpu=${GPU_ID} (mbpp + ifeval, limits: mbpp=${LIMIT_MBPP}, ifeval=${LIMIT_IFEVAL})"
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
  echo "[tier] ${tier} -> mbpp"
  echo "============================================================"
  run_patched_split_base "mbpp_instruct" 768 "${LIMIT_MBPP}" "${ROOT_OUT_MBPP}" "${LOG_DIR_MBPP}" "${tier}"

  echo "============================================================"
  echo "[tier] ${tier} -> ifeval"
  echo "============================================================"
  run_patched_split_base "ifeval" 768 "${LIMIT_IFEVAL}" "${ROOT_OUT_IFEVAL}" "${LOG_DIR_IFEVAL}" "${tier}"
done

echo "[done-part2-limit9999] gpu=${GPU_ID} (mbpp + ifeval)"
