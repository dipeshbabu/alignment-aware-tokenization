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

python -m models.probe \
  --config configs/pythia410m.yml \
  --save probes/pythia410m_layer10.npy \
  --anchors "$ANCHORS" \
  --neutrals "$NEUTRALS" \
  --batch 8 --max_len 256

python -m models.probe \
  --config configs/pythia1_4b.yml \
  --save probes/pythia1_4b_layer11.npy \
  --anchors "$ANCHORS" \
  --neutrals "$NEUTRALS" \
  --batch 4 --max_len 256

echo "Wrote Pythia probe bases to probes/ using anchors=$ANCHORS neutrals=$NEUTRALS."
