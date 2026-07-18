#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

INPUT_ROOT="${INPUT_ROOT:-outputs/longbench_stage4b}"
RUN_NAME="${RUN_NAME:-entrokv_r0.30_a0.50}"
EXPECTED_SAMPLES="${EXPECTED_SAMPLES:-100}"
OUTPUT_DIR="${OUTPUT_DIR:-$INPUT_ROOT/visualization/$RUN_NAME}"
DPI="${DPI:-300}"

python experiments/LongBench/visualize_stage4c.py \
  --input-root "$INPUT_ROOT" \
  --run-name "$RUN_NAME" \
  --tasks qasper hotpotqa passage_retrieval_en \
  --expected-samples "$EXPECTED_SAMPLES" \
  --output-dir "$OUTPUT_DIR" \
  --dpi "$DPI"

echo "[PASS] Stage 4C visualization completed: $OUTPUT_DIR"
