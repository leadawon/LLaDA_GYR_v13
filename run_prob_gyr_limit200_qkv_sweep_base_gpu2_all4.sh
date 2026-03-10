#!/usr/bin/env bash
set -euo pipefail

# Single-GPU (GPU2) all-4-benchmark launcher for LLaDA GYR sweep.
# Runs sequentially on one GPU:
#   gsm8k_cot -> humaneval_instruct -> mbpp_instruct -> ifeval
#
# Defaults follow your current benchmark policy:
#   gsm8k: max_new_tokens=256, limit=200
#   humaneval: max_new_tokens=512, limit=30
#   mbpp: max_new_tokens=768, limit=20
#   ifeval: max_new_tokens=768, limit=50

cd /workspace/LLaDA_GYR_v13

GPU_ID="${GPU_ID:-2}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-256}"
ONLY_TIERS="${ONLY_TIERS:-conservative,balanced,progressive,radical,very_radical,extremradical,ultraradical,hyperradical,omegaradical}"

LIMIT_GSM8K="${LIMIT_GSM8K:-200}"
LIMIT_HUMANEVAL="${LIMIT_HUMANEVAL:-30}"
LIMIT_MBPP="${LIMIT_MBPP:-20}"
LIMIT_IFEVAL="${LIMIT_IFEVAL:-50}"

echo "[start-all4-single] gpu=${GPU_ID} shard=${SHARD_INDEX}/${SHARD_COUNT} steps=${DIFFUSION_STEPS}"

echo "[1/4] gsm8k_cot"
GPU_ID="${GPU_ID}" \
SHARD_INDEX="${SHARD_INDEX}" \
SHARD_COUNT="${SHARD_COUNT}" \
DIFFUSION_STEPS="${DIFFUSION_STEPS}" \
LIMIT="${LIMIT_GSM8K}" \
ONLY_TIERS="${ONLY_TIERS}" \
  bash ./run_prob_gyr_limit200_qkv_sweep_split_base_gsm8k.sh

echo "[2/4] humaneval_instruct"
GPU_ID="${GPU_ID}" \
SHARD_INDEX="${SHARD_INDEX}" \
SHARD_COUNT="${SHARD_COUNT}" \
DIFFUSION_STEPS="${DIFFUSION_STEPS}" \
LIMIT="${LIMIT_HUMANEVAL}" \
ONLY_TIERS="${ONLY_TIERS}" \
  bash ./run_prob_gyr_limit200_qkv_sweep_split_base_humaneval.sh

echo "[3/4] mbpp_instruct"
GPU_ID="${GPU_ID}" \
SHARD_INDEX="${SHARD_INDEX}" \
SHARD_COUNT="${SHARD_COUNT}" \
DIFFUSION_STEPS="${DIFFUSION_STEPS}" \
LIMIT="${LIMIT_MBPP}" \
ONLY_TIERS="${ONLY_TIERS}" \
  bash ./run_prob_gyr_limit200_qkv_sweep_split_base_mbpp.sh

echo "[4/4] ifeval"
GPU_ID="${GPU_ID}" \
SHARD_INDEX="${SHARD_INDEX}" \
SHARD_COUNT="${SHARD_COUNT}" \
DIFFUSION_STEPS="${DIFFUSION_STEPS}" \
LIMIT="${LIMIT_IFEVAL}" \
ONLY_TIERS="${ONLY_TIERS}" \
  bash ./run_prob_gyr_limit200_qkv_sweep_split_base_ifeval.sh

echo "[done-all4-single] gpu=${GPU_ID}"
