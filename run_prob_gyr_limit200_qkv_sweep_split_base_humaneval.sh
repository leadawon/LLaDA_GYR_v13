#!/usr/bin/env bash
set -euo pipefail

cd /workspace/LLaDA_GYR_v13

GPU_ID="${GPU_ID:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-256}"
LIMIT="${LIMIT:-30}"

ROOT_OUT="${ROOT_OUT:-out/prob_gyr_limit200_qkv_sweep_llada_alltiers_gpu${GPU_ID}_humaneval_instruct_limit${LIMIT}}"
LOG_DIR="${LOG_DIR:-out/logs/prob_gyr_limit200_qkv_sweep_llada_alltiers_gpu${GPU_ID}_humaneval_instruct_limit${LIMIT}}"

GPU_ID="${GPU_ID}" \
SHARD_INDEX="${SHARD_INDEX}" \
SHARD_COUNT="${SHARD_COUNT}" \
TASK="humaneval_instruct" \
MAX_NEW_TOKENS=512 \
DIFFUSION_STEPS="${DIFFUSION_STEPS}" \
LIMIT="${LIMIT}" \
ROOT_OUT="${ROOT_OUT}" \
LOG_DIR="${LOG_DIR}" \
ONLY_TIERS="${ONLY_TIERS:-}" \
  bash ./run_prob_gyr_limit200_qkv_sweep_split_base.sh
