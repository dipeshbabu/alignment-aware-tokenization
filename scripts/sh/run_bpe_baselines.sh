#!/usr/bin/env bash
set -euo pipefail

BASE_TOK=${1:-"EleutherAI/pythia-410m"}
MODEL_ID=${2:-$BASE_TOK}
PROBE=${3:-"probes/pythia410m_layer10.npy"}
ANCHORS=${4:-"data/anchors/anchors_500.jsonl"}

for BASELINE in frequency random shuffled; do
  python -m tokenizers.bpe_search \
    --base_tokenizer "$BASE_TOK" \
    --model_name "$MODEL_ID" \
    --anchors "$ANCHORS" \
    --neutrals data/neutrals/neutrals_1000.jsonl \
    --u_dev_dataset data/unlabeled/u_dev.jsonl \
    --u_dev_size 20000 \
    --probe "$PROBE" \
    --rounds 5 \
    --prune_k 30 \
    --min_benign_hits 5 \
    --alpha 0.7 \
    --beta 0.1 \
    --gamma 0.0 \
    --warmup_steps 150 \
    --baseline "$BASELINE" \
    --out "tokenizers/bpe_${BASELINE}_baseline"
done
