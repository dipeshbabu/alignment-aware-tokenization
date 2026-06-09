#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   uv run bash scripts/sh/run_lora_drift_native_bpe.sh [pythia410m|pythia1_4b|llama3_8b|qwen2_7b]

FAMILY=${1:-pythia410m}
SPLIT_DIR=${AAT_SPLIT_DIR:-data/splits/stems_seed9172}
NEUTRALS=${NEUTRALS:-data/neutrals/neutrals_1000.jsonl}
if [[ -f "$SPLIT_DIR/neutrals_train_stems.jsonl" ]]; then
  NEUTRALS="$SPLIT_DIR/neutrals_train_stems.jsonl"
fi

case "$FAMILY" in
  pythia410m)
    CFG=configs/pythia410m.yml
    TOK=${TOK:-tokenizers/native_bpe_pythia410m/hazard}
    PROBE=${PROBE:-probes/pythia410m_layer10.npy}
    OUT=${OUT:-adapters/pythia410m-native-bpe-lora-drift}
    ;;
  pythia1_4b)
    CFG=configs/pythia1_4b.yml
    TOK=${TOK:-tokenizers/native_bpe_pythia1_4b/hazard}
    PROBE=${PROBE:-probes/pythia1_4b_layer11.npy}
    OUT=${OUT:-adapters/pythia1_4b-native-bpe-lora-drift}
    ;;
  llama3_8b)
    CFG=configs/llama3_8b.yml
    TOK=${TOK:-tokenizers/native_bpe_llama3_8b/hazard}
    PROBE=${PROBE:-probes/llama3_8b_layer16.npy}
    OUT=${OUT:-adapters/llama3_8b-native-bpe-lora-drift}
    ;;
  qwen2_7b)
    CFG=configs/qwen2_7b.yml
    TOK=${TOK:-tokenizers/native_bpe_qwen2_7b/hazard}
    PROBE=${PROBE:-probes/qwen2_7b_layer16.npy}
    OUT=${OUT:-adapters/qwen2_7b-native-bpe-lora-drift}
    ;;
  *)
    echo "Usage: $0 [pythia410m|pythia1_4b|llama3_8b|qwen2_7b]" >&2
    exit 2
    ;;
esac

if [[ ! -d "$TOK" ]]; then
  echo "Missing tokenizer $TOK. Run scripts/sh/run_native_bpe_search.sh $FAMILY first." >&2
  exit 2
fi
if [[ ! -f "$PROBE" ]]; then
  echo "Missing probe $PROBE. Run probe scripts first." >&2
  exit 2
fi

uv run python scripts/run_lora_drift_native_bpe.py \
  --config "$CFG" \
  --tokenizer "$TOK" \
  --probe "$PROBE" \
  --save "$OUT" \
  --neutrals "$NEUTRALS"
