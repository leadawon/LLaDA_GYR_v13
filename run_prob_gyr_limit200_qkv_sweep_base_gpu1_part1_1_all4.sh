#!/usr/bin/env bash
set -euo pipefail

cd /workspace/LLaDA_GYR_v13

GPU_ID="${GPU_ID:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-256}"
LIMIT_GSM8K="${LIMIT_GSM8K:-200}"
ONLY_TIERS="${ONLY_TIERS:-conservative,balanced,progressive,radical,very_radical,extremradical,ultraradical,hyperradical,omegaradical}"

ROOT_OUT_GSM8K="${ROOT_OUT_GSM8K:-out/prob_gyr_limit200_qkv_sweep_llada_alltiers_gpu${GPU_ID}_gsm8k_limit${LIMIT_GSM8K}}"
LOG_DIR_GSM8K="${LOG_DIR_GSM8K:-out/logs/prob_gyr_limit200_qkv_sweep_llada_alltiers_gpu${GPU_ID}_gsm8k_limit${LIMIT_GSM8K}}"

echo "[start-part1_1] gpu=${GPU_ID} (gsm8k, limit=${LIMIT_GSM8K})"
echo "[proxy-fixed] qk_softmax | ls=early_mid_late | hr=max | sym=max | th=0.4"

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
TASK="gsm8k_cot" \
MAX_NEW_TOKENS=256 \
DIFFUSION_STEPS="${DIFFUSION_STEPS}" \
LIMIT="${LIMIT_GSM8K}" \
ROOT_OUT="${ROOT_OUT_GSM8K}" \
LOG_DIR="${LOG_DIR_GSM8K}" \
ONLY_TIERS="${ONLY_TIERS}" \
  bash "${tmp_script}"

rm -f "${tmp_script}"

echo "[done-part1_1] gpu=${GPU_ID} (gsm8k)"
