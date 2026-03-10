#!/usr/bin/env bash
set -euo pipefail

cd /workspace/LLaDA_GYR_v13

GPU_ID="${GPU_ID:-2}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-256}"
LIMIT_HUMANEVAL="${LIMIT_HUMANEVAL:-30}"
ONLY_TIERS="${ONLY_TIERS:-conservative,balanced,progressive,radical,very_radical,extremradical,ultraradical,hyperradical,omegaradical}"

ROOT_OUT_HUMANEVAL="${ROOT_OUT_HUMANEVAL:-out/prob_gyr_limit200_qkv_sweep_llada_alltiers_gpu${GPU_ID}_humaneval_instruct_limit${LIMIT_HUMANEVAL}}"
LOG_DIR_HUMANEVAL="${LOG_DIR_HUMANEVAL:-out/logs/prob_gyr_limit200_qkv_sweep_llada_alltiers_gpu${GPU_ID}_humaneval_instruct_limit${LIMIT_HUMANEVAL}}"

echo "[start-part1_2] gpu=${GPU_ID} (humaneval, limit=${LIMIT_HUMANEVAL})"
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
TASK="humaneval_instruct" \
MAX_NEW_TOKENS=512 \
DIFFUSION_STEPS="${DIFFUSION_STEPS}" \
LIMIT="${LIMIT_HUMANEVAL}" \
ROOT_OUT="${ROOT_OUT_HUMANEVAL}" \
LOG_DIR="${LOG_DIR_HUMANEVAL}" \
ONLY_TIERS="${ONLY_TIERS}" \
  bash "${tmp_script}"

rm -f "${tmp_script}"

echo "[done-part1_2] gpu=${GPU_ID} (humaneval)"
