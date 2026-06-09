#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/sh/run_lora_drift_spm.sh [mistral7b|llama3_8b|qwen2_7b]

FAMILY=${1:-mistral7b}
SPLIT_DIR=${AAT_SPLIT_DIR:-data/splits/stems_seed9172}
NEUTRALS=${NEUTRALS:-data/neutrals/neutrals_1000.jsonl}
if [[ -f "$SPLIT_DIR/neutrals_train_stems.jsonl" ]]; then
  NEUTRALS="$SPLIT_DIR/neutrals_train_stems.jsonl"
fi

case "$FAMILY" in
  mistral7b)
    CFG=configs/mistral7b_spm.yml
    PROBE=probes/mistral7b_layer16.npy
    OUT=adapters/mistral7b-spm-lora-drift
    ;;
  llama3_8b)
    CFG=configs/llama3_8b_spm.yml
    PROBE=probes/llama3_8b_layer16.npy
    OUT=adapters/llama3_8b-spm-lora-drift
    ;;
  qwen2_7b)
    CFG=configs/qwen2_7b_spm.yml
    PROBE=probes/qwen2_7b_layer16.npy
    OUT=adapters/qwen2_7b-spm-lora-drift
    ;;
  *)
    echo "Usage: bash scripts/sh/run_lora_drift_spm.sh [mistral7b|llama3_8b|qwen2_7b]" >&2
    exit 2
    ;;
esac

if [ ! -f "$PROBE" ]; then
  echo "Missing probe $PROBE. Run scripts/sh/run_probes_spm.sh first." >&2
  exit 2
fi

python -m models.lora_drift --config "$CFG" --probe "$PROBE" --save "$OUT" --neutrals "$NEUTRALS"
