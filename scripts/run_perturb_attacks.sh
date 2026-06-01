#!/usr/bin/env bash
set -euo pipefail

SPLIT_DIR=${1:-data/splits/stems_seed9172}
OUT=${2:-data/eval/attack_perturbed_seed9172.jsonl}

if [ ! -f "${SPLIT_DIR}/heldout_stems.json" ]; then
  python -m data_tools.make_stem_split \
    --anchors data/anchors/anchors_500.jsonl \
    --out_dir "$SPLIT_DIR" \
    --heldout_frac 0.2 \
    --seed 9172
fi

python -m data_tools.perturb_attacks \
  --input data/eval/attack_extra_500.jsonl \
  --output "$OUT" \
  --stems "${SPLIT_DIR}/heldout_stems.json" \
  --variants space_split punct_insert case_flip char_delete char_substitute \
  --per_prompt 3 \
  --seed 9172 \
  --include_original
