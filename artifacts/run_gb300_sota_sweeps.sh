#!/usr/bin/env bash
set -euo pipefail

# Reproduce the GB300 SOTA sweeps used for issue #1 artifacts.
# Run from the repository root.

COMMON_ARGS=(
  --no-color
  --total-gpus 72
  --system gb300
  --top-n 1
)

uv run aiconfigurator cli default \
  "${COMMON_ARGS[@]}" \
  --model-path nvidia/Kimi-K2.5-NVFP4 \
  --backend auto \
  --save-dir artifacts/gb300_pareto/kimi

uv run aiconfigurator cli default \
  "${COMMON_ARGS[@]}" \
  --model-path nvidia/MiniMax-M2.7-NVFP4 \
  --backend auto \
  --save-dir artifacts/gb300_pareto/minimax

uv run aiconfigurator cli default \
  "${COMMON_ARGS[@]}" \
  --model-path nvidia/GLM-5-NVFP4 \
  --backend trtllm \
  --save-dir artifacts/gb300_pareto/glm_trtllm

uv run aiconfigurator cli default \
  "${COMMON_ARGS[@]}" \
  --model-path nvidia/GLM-5-NVFP4 \
  --backend vllm \
  --save-dir artifacts/gb300_pareto/glm_vllm

# DeepSeek V4 Pro did not produce plottable GB300 candidates in this run:
# the support matrix failed for the standard backend/model variants, and this
# HYBRID MegaMoE path lacked the required GB300 DSv4 MegaMoE module data.
uv run aiconfigurator cli default \
  "${COMMON_ARGS[@]}" \
  --model-path sgl-project/DeepSeek-V4-Pro-FP8 \
  --backend sglang \
  --database-mode HYBRID \
  --moe-backend megamoe \
  --save-dir artifacts/gb300_pareto/deepseek_v4_pro_hybrid

uv run python artifacts/gb300_pareto/build_pareto_artifacts.py
