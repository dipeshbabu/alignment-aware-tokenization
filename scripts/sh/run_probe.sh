#!/usr/bin/env bash
set -euo pipefail

CFG=${1:-configs/pythia410m.yml}
OUT=${2:-probes/v_layer.npy}

python -m models.probe --config "$CFG" --save "$OUT" --batch 2 --max_len 256
