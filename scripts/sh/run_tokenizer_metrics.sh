#!/usr/bin/env bash
set -euo pipefail

TOKSPILL=${1:-data/tokspill/tokspill_seed9172.jsonl}

python -m eval.tokenizer_spillover \
  --tokenizer EleutherAI/pythia-410m \
  --tokspill "$TOKSPILL" \
  --split heldout \
  --out outputs/tokspill_pythia410m_base_heldout.json

if [ -d tokenizers/bpe_searched ]; then
  python -m eval.tokenizer_spillover \
    --tokenizer tokenizers/bpe_searched \
    --tokspill "$TOKSPILL" \
    --split heldout \
    --out outputs/tokspill_pythia410m_bpe_searched_heldout.json
fi

for BASELINE in frequency random shuffled; do
  if [ -d "tokenizers/bpe_${BASELINE}_baseline" ]; then
    python -m eval.tokenizer_spillover \
      --tokenizer "tokenizers/bpe_${BASELINE}_baseline" \
      --tokspill "$TOKSPILL" \
      --split heldout \
      --out "outputs/tokspill_pythia410m_bpe_${BASELINE}_heldout.json"
  fi
done
