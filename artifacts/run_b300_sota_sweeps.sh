#!/usr/bin/env bash
set -euo pipefail

# Reproduce the DGX-B300 SXM sweeps used for issue #1 artifacts.
# Run from the repository root.

COMMON_ARGS=(
  --no-color
  --total-gpus 8
  --system b300_sxm
  --top-n 1
)

uv run aiconfigurator cli default \
  "${COMMON_ARGS[@]}" \
  --model-path nvidia/Kimi-K2.5-NVFP4 \
  --backend auto \
  --save-dir artifacts/b300_pareto/kimi

uv run aiconfigurator cli default \
  "${COMMON_ARGS[@]}" \
  --model-path nvidia/MiniMax-M2.7-NVFP4 \
  --backend auto \
  --save-dir artifacts/b300_pareto/minimax

uv run aiconfigurator cli default \
  "${COMMON_ARGS[@]}" \
  --model-path nvidia/GLM-5-NVFP4 \
  --backend trtllm \
  --save-dir artifacts/b300_pareto/glm_trtllm

uv run aiconfigurator cli default \
  "${COMMON_ARGS[@]}" \
  --model-path nvidia/GLM-5-NVFP4 \
  --backend vllm \
  --save-dir artifacts/b300_pareto/glm_vllm

uv run aiconfigurator cli default \
  "${COMMON_ARGS[@]}" \
  --model-path nvidia/GLM-5-NVFP4 \
  --backend sglang \
  --backend-version 0.5.9 \
  --save-dir artifacts/b300_pareto/glm_sglang_059

# DeepSeek V4 Pro did not produce plottable DGX-B300 candidates in this run:
# the support matrix failed for the available variants/backends, and this
# HYBRID MegaMoE path lacked dsv4_megamoe_module perf data for b300_sxm.
uv run aiconfigurator cli default \
  "${COMMON_ARGS[@]}" \
  --model-path sgl-project/DeepSeek-V4-Pro-FP8 \
  --backend sglang \
  --database-mode HYBRID \
  --moe-backend megamoe \
  --save-dir artifacts/b300_pareto/deepseek_v4_pro_hybrid

uv run python artifacts/b300_pareto/build_pareto_artifacts.py
