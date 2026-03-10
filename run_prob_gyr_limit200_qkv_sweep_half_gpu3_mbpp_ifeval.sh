#!/usr/bin/env bash
set -euo pipefail

cd /workspace/LLaDA_GYR_v13

SHARD_INDEX="${SHARD_INDEX:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-256}"
ONLY_TIERS="${ONLY_TIERS:-conservative,balanced,progressive,radical,very_radical,extremradical,ultraradical,hyperradical,omegaradical}"

echo "[start-half] gpu2:mbpp + gpu3:ifeval"

GPU_ID="${GPU2_ID:-2}" SHARD_INDEX="${SHARD_INDEX}" SHARD_COUNT="${SHARD_COUNT}" DIFFUSION_STEPS="${DIFFUSION_STEPS}" LIMIT="${LIMIT_MBPP:-20}" ONLY_TIERS="${ONLY_TIERS}" \
  bash ./run_prob_gyr_limit200_qkv_sweep_split_base_mbpp.sh

GPU_ID="${GPU3_ID:-3}" SHARD_INDEX="${SHARD_INDEX}" SHARD_COUNT="${SHARD_COUNT}" DIFFUSION_STEPS="${DIFFUSION_STEPS}" LIMIT="${LIMIT_IFEVAL:-50}" ONLY_TIERS="${ONLY_TIERS}" \
  bash ./run_prob_gyr_limit200_qkv_sweep_split_base_ifeval.sh

echo "[done-half] gpu2:mbpp + gpu3:ifeval"
