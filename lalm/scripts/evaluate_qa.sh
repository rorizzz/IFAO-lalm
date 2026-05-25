#!/bin/bash
# LALM QA evaluation script (single GPU).
#
# Usage (from examples/lalm/):
#   bash scripts/evaluate_qa.sh
#
# Override checkpoint via env vars, e.g.:
#   ITER=5000 bash scripts/evaluate_qa.sh
#   EPOCH=3   bash scripts/evaluate_qa.sh

set -euo pipefail

# ── Environment ───────────────────────────────────────────────────────────────
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
# ── GPU ───────────────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# ── Experiment ────────────────────────────────────────────────────────────────
exp_name="qwen25omni_qwen25_7b_ds2"
exp_dir="./exp/$exp_name"

# ── Checkpoint ────────────────────────────────────────────────────────────────
iter=${ITER:-300000}
epoch=${EPOCH:-0}

# ── Decoding ──────────────────────────────────────────────────────────────────
decoding_method="beam_search"
num_beams=4
max_new_tokens=256
dtype="fp16"

# ── Data ──────────────────────────────────────────────────────────────────────
test_data_config="configs/test_qa_data_config.yaml"
max_duration=500

echo "========================================================"
echo "  Exp:      ${exp_dir}"
echo "  Iter:     ${iter}  |  Epoch: ${epoch}"
echo "  Decoding: ${decoding_method} (beams=${num_beams})"
echo "  Max new tokens: ${max_new_tokens}"
echo "========================================================"

python evaluate_qa.py \
    exp_dir="$exp_dir" \
    checkpoint.iter="$iter" \
    decoding_method="$decoding_method" \
    num_beams="$num_beams" \
    max_new_tokens="$max_new_tokens" \
    dtype="$dtype" \
    data.test_data_config="$test_data_config" \
    data.max_duration="$max_duration"
