#!/usr/bin/env bash
set -euo pipefail

mkdir -p probes

SPLIT_DIR=${AAT_SPLIT_DIR:-data/splits/stems_seed9172}
ANCHORS=${ANCHORS:-data/anchors/anchors_500.jsonl}
NEUTRALS=${NEUTRALS:-data/neutrals/neutrals_1000.jsonl}
if [[ -f "$SPLIT_DIR/anchors_train_stems.jsonl" ]]; then
  ANCHORS="$SPLIT_DIR/anchors_train_stems.jsonl"
fi
if [[ -f "$SPLIT_DIR/neutrals_train_stems.jsonl" ]]; then
  NEUTRALS="$SPLIT_DIR/neutrals_train_stems.jsonl"
fi

python -m models.probe --config configs/mistral7b.yml --save probes/mistral7b_layer16.npy --anchors "$ANCHORS" --neutrals "$NEUTRALS" --batch 2 --max_len 256
python -m models.probe --config configs/llama3_8b.yml --save probes/llama3_8b_layer16.npy --anchors "$ANCHORS" --neutrals "$NEUTRALS" --batch 2 --max_len 256
python -m models.probe --config configs/qwen2_7b.yml --save probes/qwen2_7b_layer16.npy --anchors "$ANCHORS" --neutrals "$NEUTRALS" --batch 2 --max_len 256

echo "Wrote large-backbone probe bases to probes/ using anchors=$ANCHORS neutrals=$NEUTRALS."
