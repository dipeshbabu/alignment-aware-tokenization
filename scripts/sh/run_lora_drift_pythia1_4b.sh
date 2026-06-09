#!/usr/bin/env bash
set -euo pipefail
PROBE=${1:-probes/pythia1_4b_layer11.npy}
OUT=${2:-adapters/pythia1_4b-lora-drift}
SPLIT_DIR=${AAT_SPLIT_DIR:-data/splits/stems_seed9172}
ANCHORS=${ANCHORS:-data/anchors/anchors_500.jsonl}
NEUTRALS=${NEUTRALS:-data/neutrals/neutrals_1000.jsonl}
if [[ -f "$SPLIT_DIR/anchors_train_stems.jsonl" ]]; then
  ANCHORS="$SPLIT_DIR/anchors_train_stems.jsonl"
fi
if [[ -f "$SPLIT_DIR/neutrals_train_stems.jsonl" ]]; then
  NEUTRALS="$SPLIT_DIR/neutrals_train_stems.jsonl"
fi

if [ ! -f "$PROBE" ]; then
  python -m models.probe --config configs/pythia1_4b.yml --save "$PROBE" --anchors "$ANCHORS" --neutrals "$NEUTRALS" --batch 4 --max_len 256
fi

python -m models.lora_drift --config configs/pythia1_4b.yml --probe "$PROBE" --save "$OUT" --neutrals "$NEUTRALS"
