#!/usr/bin/env bash
set -euo pipefail

cd /workspace/LLaDA_GYR_v13
GPU_ID="${GPU_ID:-2}" \
SHARD_INDEX="${SHARD_INDEX:-0}" \
SHARD_COUNT="${SHARD_COUNT:-1}" \
DIFFUSION_STEPS="${DIFFUSION_STEPS:-256}" \
LIMIT="${LIMIT:-20}" \
ONLY_TIERS="${ONLY_TIERS:-conservative,balanced,progressive,radical,very_radical,extremradical,ultraradical,hyperradical,omegaradical}" \
  bash ./run_prob_gyr_limit200_qkv_sweep_split_base_mbpp.sh
