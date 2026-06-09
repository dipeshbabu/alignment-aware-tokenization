#!/usr/bin/env bash
set -euo pipefail

# Label-efficiency runs for frozen-feature hazard probes.
# Usage: bash scripts/sh/run_label_efficiency.sh [config] [out_json] [budgets...]

CONFIG=${1:-configs/pythia410m.yml}
OUT=${2:-outputs/label_efficiency_pythia410m.json}
shift 2 || true

if [ "$#" -gt 0 ]; then
  BUDGETS=("$@")
else
  BUDGETS=(50 100 300)
fi

python -m eval.label_efficiency \
  --config "$CONFIG" \
  --budgets "${BUDGETS[@]}" \
  --trials "${LABEL_EFF_TRIALS:-10}" \
  --batch_size "${LABEL_EFF_BS:-16}" \
  --dedup \
  --check_overlap \
  --out_json "$OUT"
