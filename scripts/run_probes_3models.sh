#!/usr/bin/env bash
set -euo pipefail

mkdir -p probes

python -m models.probe \
  --config configs/pythia410m.yml \
  --save probes/pythia410m_layer10.npy \
  --batch 8 --max_len 256

python -m models.probe \
  --config configs/pythia1_4b.yml \
  --save probes/pythia1_4b_layer11.npy \
  --batch 4 --max_len 256

python -m models.probe \
  --config configs/gemma2b.yml \
  --save probes/gemma2b_layer10.npy \
  --batch 2 --max_len 256