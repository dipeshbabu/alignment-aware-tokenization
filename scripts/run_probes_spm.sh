#!/usr/bin/env bash
set -euo pipefail

mkdir -p probes

python -m models.probe --config configs/mistral7b.yml --save probes/mistral7b_layer16.npy --batch 2 --max_len 256
python -m models.probe --config configs/llama3_8b.yml --save probes/llama3_8b_layer16.npy --batch 2 --max_len 256
python -m models.probe --config configs/qwen2_7b.yml --save probes/qwen2_7b_layer16.npy --batch 2 --max_len 256

echo "Wrote SPM-backbone probe vectors to probes/."
