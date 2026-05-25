#!/bin/bash
# Assemble a LALM checkpoint from pretrained LLM + audio encoder components.
#
# Run once before training. Saves a self-contained HF checkpoint to $output_dir.
# After this, load the model with:
#   AutoModelForCausalLM.from_pretrained(output_dir)
#   LALMProcessor.from_pretrained(output_dir)
#
# Usage:
#   cd examples/lalm && bash scripts/build_model.sh

set -euo pipefail

# ── Components ────────────────────────────────────────────────────────────────
llm=/path/to/Qwen2.5-7B-Instruct
encoder=/path/to/Qwen2.5-Omni-7B

# ── Downsampling ──────────────────────────────────────────────────────────────
# projector_downsample_rate: frame concat in LALMProjector (higher = fewer LLM tokens)
projector_downsample_rate=2

# ── Output ────────────────────────────────────────────────────────────────────
output_dir=./models/qwen25omni_qwen25_7b_ds${projector_downsample_rate}

echo "LLM:     $llm"
echo "Encoder: $encoder"
echo "Output:  $output_dir"

python build_model.py \
    --llm "$llm" \
    --encoder "$encoder" \
    --output_dir "$output_dir" \
    --dtype bfloat16 \
    --projector_downsample_rate "$projector_downsample_rate"

echo "Done. Checkpoint saved to $output_dir"
