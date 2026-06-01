#!/usr/bin/env bash
set -euo pipefail
CFG=${1:-configs/pythia410m.yml}
PROBE=${2:-probes/pythia410m_layer10.npy}
OUT=${3:-adapters/pythia410m-lora-drift}
python -m models.lora_drift --config "$CFG" --probe "$PROBE" --save "$OUT"
