#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct}"
LONGBENCH_PATH="${LONGBENCH_PATH:-zai-org/LongBench}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/longbench_stage4b}"
SAMPLE_START="${SAMPLE_START:-0}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-1}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-8192}"
RETENTION_RATIO="${RETENTION_RATIO:-0.30}"
ALPHA="${ALPHA:-0.50}"
H_BAR="${H_BAR:-0.30}"
TASKS=(qasper hotpotqa passage_retrieval_en)

COMMON_ARGS=(
  --model-path "$MODEL_PATH"
  --dataset-source "$LONGBENCH_PATH"
  --tasks "${TASKS[@]}"
  --output-root "$OUTPUT_ROOT"
  --sample-start "$SAMPLE_START"
  --sample-limit "$SAMPLE_LIMIT"
  --max-input-tokens "$MAX_INPUT_TOKENS"
  --retention-ratio "$RETENTION_RATIO"
  --window-size 32
  --kernel-size 7
  --pooling maxpool
  --local-files-only
)

python experiments/LongBench/stage4b_runner.py \
  --method full \
  --run-name full \
  "${COMMON_ARGS[@]}"

python experiments/LongBench/stage4b_runner.py \
  --method fixed \
  --run-name fixed_r0.30 \
  "${COMMON_ARGS[@]}"

python experiments/LongBench/stage4b_runner.py \
  --method entrokv \
  --run-name entrokv_r0.30_a0.50 \
  --alpha "$ALPHA" \
  --h-bar "$H_BAR" \
  "${COMMON_ARGS[@]}"

python experiments/LongBench/eval_stage4b.py \
  --output-root "$OUTPUT_ROOT" \
  --runs full fixed_r0.30 entrokv_r0.30_a0.50 \
  --tasks "${TASKS[@]}"
