#!/usr/bin/env bash
set -euo pipefail
PROBE=${1:-probes/pythia1_4b_layer11.npy}
OUT=${2:-adapters/pythia1_4b-lora-drift}

if [ ! -f "$PROBE" ]; then
  python -m models.probe --config configs/pythia1_4b.yml --save "$PROBE" --batch 4 --max_len 256
fi

python -m models.lora_drift --config configs/pythia1_4b.yml --probe "$PROBE" --save "$OUT"
