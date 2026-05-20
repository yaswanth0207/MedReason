#!/usr/bin/env bash
# MedReason – QLoRA Fine-Tuning & 4-bit Quantized Export
#
# Prerequisites:
#   pip install "mlx-lm[train]"
#   python prepare_data.py          # generates data/train.jsonl & valid.jsonl
#
# Usage:
#   bash train_and_fuse.sh
set -euo pipefail

# Must match the ``model`` field in lora_config.yaml.
MODEL="mlx-community/Qwen2.5-3B-Instruct-8bit"
ADAPTER_DIR="./adapters"
FUSED_FP16_DIR="./medreason-fused-fp16"
FINAL_DIR="./medreason-4bit"

# ── Step 1: QLoRA fine-tune ─────────────────────────────────────────
echo "============================================"
echo " Step 1/3 – QLoRA Fine-Tuning"
echo "============================================"
mlx_lm.lora --config lora_config.yaml

# ── Step 2: Fuse adapters into a full-precision model ───────────────
#    --dequantize converts the 8-bit base back to FP16 before fusing
#    so no quantization noise leaks into the merged weights.
echo ""
echo "============================================"
echo " Step 2/3 – Fuse Adapters (dequantize → FP16)"
echo "============================================"
mlx_lm.fuse \
    --model "$MODEL" \
    --adapter-path "$ADAPTER_DIR" \
    --save-path "$FUSED_FP16_DIR" \
    --dequantize

# ── Step 3: Re-quantize to 4-bit for efficient inference ────────────
echo ""
echo "============================================"
echo " Step 3/3 – Quantize to 4-bit"
echo "============================================"
mlx_lm.convert \
    --hf-path "$FUSED_FP16_DIR" \
    --mlx-path "$FINAL_DIR" \
    --quantize \
    --q-bits 4

echo ""
echo "============================================"
echo " Done!"
echo " Quantized model saved to: $FINAL_DIR"
echo " Launch the UI:  python app.py"
echo "============================================"
