#!/usr/bin/env bash
set -euo pipefail

SPLIT_DIR=${1:-data/splits/stems_seed9172}
OUT=${2:-data/tokspill/tokspill_seed9172.jsonl}

if [ ! -f "${SPLIT_DIR}/train_stems.json" ] || [ ! -f "${SPLIT_DIR}/heldout_stems.json" ]; then
  python -m data_tools.make_stem_split \
    --anchors data/anchors/anchors_500.jsonl \
    --out_dir "$SPLIT_DIR" \
    --heldout_frac 0.2 \
    --seed 9172
fi

python -m data_tools.build_tokspill \
  --anchors data/anchors/anchors_500.jsonl \
  --neutrals data/neutrals/neutrals_1000.jsonl \
  --train_stems "${SPLIT_DIR}/train_stems.json" \
  --heldout_stems "${SPLIT_DIR}/heldout_stems.json" \
  --out "$OUT" \
  --seed 9172
