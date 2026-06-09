#!/usr/bin/env bash
set -euo pipefail
CFG=${1:-configs/pythia410m.yml}
PROBE=${2:-probes/pythia410m_layer10.npy}
OUT=${3:-adapters/pythia410m-lora-drift}
SPLIT_DIR=${AAT_SPLIT_DIR:-data/splits/stems_seed9172}
NEUTRALS=${NEUTRALS:-data/neutrals/neutrals_1000.jsonl}
if [[ -f "$SPLIT_DIR/neutrals_train_stems.jsonl" ]]; then
  NEUTRALS="$SPLIT_DIR/neutrals_train_stems.jsonl"
fi
python -m models.lora_drift --config "$CFG" --probe "$PROBE" --save "$OUT" --neutrals "$NEUTRALS"
