#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL=${1:-google/gemma-2b}
U_TRAIN=${2:-data/unlabeled/u_train.jsonl}
OUT_TOK=${3:-tokenizers/gemma_unigram_priors_tok}
OUT_MODEL=${4:-checkpoints/gemma_unigram_remapped}

mkdir -p tokenizers checkpoints

python -m tokenizers.gemma_unigram_priors \
  --base_tokenizer "$BASE_MODEL" \
  --train_jsonl "$U_TRAIN" --key text --limit 500000 \
  --hazard_terms_txt data/anchors/hazard_terms.txt \
  --penalty_substrings_txt data/anchors/hazard_substrings.txt \
  --boost_repeats 30 \
  --out_dir "$OUT_TOK"

python -m models.embed_remap_cli \
  --base_model "$BASE_MODEL" \
  --old_tokenizer "$BASE_MODEL" \
  --new_tokenizer "$OUT_TOK" \
  --out_dir "$OUT_MODEL" \
  --dtype float16

echo "Done. Remapped Gemma model in $OUT_MODEL and tokenizer in $OUT_TOK"