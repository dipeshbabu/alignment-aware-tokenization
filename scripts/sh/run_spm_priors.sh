#!/usr/bin/env bash
set -euo pipefail

# Optional SentencePiece tokenizer experiment.
# Usage: bash scripts/sh/run_spm_priors.sh [llama3|mistral7b|qwen2_7b]

FAMILY="${1:-mistral7b}"
OUT_DIR="tokenizers/spm_hazard"
DATA_CORPUS="data/unlabeled/u_train.jsonl"
ANCHORS="data/anchors/anchors_500.jsonl"
NEUTRALS="data/neutrals/neutrals_1000.jsonl"

mkdir -p "$OUT_DIR"
PREFIX="${OUT_DIR}/spm_${FAMILY}_hazard"
EXPORT_DIR="${OUT_DIR}/spm_${FAMILY}_hazard_hf"

python -m tokenizers.spm_priors \
  --corpus "$DATA_CORPUS" \
  --anchors "$ANCHORS" \
  --neutrals "$NEUTRALS" \
  --model_prefix "$PREFIX" \
  --vocab_size 50000 \
  --lambda_conf 0.3 \
  --lambda_overlap 0.25 \
  --hazard_boost 5.0 \
  --min_stem_len 3 \
  --input_sentence_size 200000 \
  --byte_fallback true \
  --hf_target "$FAMILY" \
  --export_hf_dir "$EXPORT_DIR" \
  --addl_special "<s>,</s>,<pad>"
