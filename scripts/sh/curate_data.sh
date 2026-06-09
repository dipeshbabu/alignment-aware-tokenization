#!/usr/bin/env bash
set -euo pipefail

# Rebuild the local data needed by the experiment pipeline.
#
# This script writes generated/curated JSONL artifacts under data/. Those files
# are intentionally gitignored because they can be large, safety-sensitive, and
# subject to upstream dataset redistribution constraints.
#
# Useful overrides:
#   U_TRAIN_MB=500 U_DEV_MB=100 ANCHOR_CAP=500 NEUTRAL_CAP=1000 bash scripts/sh/curate_data.sh
#   QUICK=1 bash scripts/sh/curate_data.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

SEED=${SEED:-9172}
QUICK=${QUICK:-0}

if [[ "$QUICK" == "1" ]]; then
  U_TRAIN_MB=${U_TRAIN_MB:-25}
  U_DEV_MB=${U_DEV_MB:-10}
  ANCHOR_CAP=${ANCHOR_CAP:-200}
  NEUTRAL_CAP=${NEUTRAL_CAP:-300}
  ATTACK_N=${ATTACK_N:-100}
  BENIGN_N=${BENIGN_N:-300}
  RTP_N=${RTP_N:-100}
else
  U_TRAIN_MB=${U_TRAIN_MB:-500}
  U_DEV_MB=${U_DEV_MB:-100}
  ANCHOR_CAP=${ANCHOR_CAP:-500}
  NEUTRAL_CAP=${NEUTRAL_CAP:-1000}
  ATTACK_N=${ATTACK_N:-500}
  BENIGN_N=${BENIGN_N:-1500}
  RTP_N=${RTP_N:-500}
fi

mkdir -p data/anchors data/neutrals data/eval data/unlabeled data/splits data/tokspill

echo "[1/8] Curate unlabeled C4 train stream -> data/unlabeled/u_train.jsonl"
uv run python -m data_tools.curate_u \
  --config "${C4_CONFIG:-en}" \
  --split train \
  --target-mb "$U_TRAIN_MB" \
  --seed "$SEED" \
  --out data/unlabeled/u_train.jsonl

echo "[2/8] Curate unlabeled C4 dev stream -> data/unlabeled/u_dev.jsonl"
uv run python -m data_tools.curate_u \
  --config "${C4_CONFIG:-en}" \
  --split validation \
  --target-mb "$U_DEV_MB" \
  --seed "$SEED" \
  --out data/unlabeled/u_dev.jsonl

echo "[3/8] Curate hazard/neutral anchor set -> data/anchors/anchors_500.jsonl"
uv run python -m data_tools.curate_anchors \
  --out data/anchors/anchors_500.jsonl \
  --cap "$ANCHOR_CAP" \
  --hazard-ratio "${ANCHOR_HAZARD_RATIO:-0.75}" \
  --per_source_quota "${ANCHOR_PER_SOURCE_QUOTA:-350}" \
  --use-dolly-neutral

echo "[4/8] Mine matched neutral look-alikes -> data/neutrals/neutrals_1000.jsonl"
uv run python -m data_tools.mine_neutrals \
  --u_path data/unlabeled/u_train.jsonl \
  --anchors data/anchors/anchors_500.jsonl \
  --out data/neutrals/neutrals_1000.jsonl \
  --cap "$NEUTRAL_CAP" \
  --per_stem "${NEUTRAL_PER_STEM:-2}" \
  --min_len "${MIN_STEM_LEN:-3}"

echo "[5/8] Curate attack and benign evaluation pools"
uv run python -m data_tools.curate_attack \
  --run both \
  --n-adv "${ADV_N:-1000}" \
  --n-jbv "$ATTACK_N" \
  --out-adv data/eval/adv_harmful_full.jsonl \
  --out-jbv data/eval/attack_extra_500.jsonl

uv run python -m data_tools.curate_benign \
  --n-dolly "$BENIGN_N" \
  --n-rtp "$RTP_N" \
  --out-dolly data/eval/benign_1500.jsonl \
  --out-rtp data/eval/benign_rtp_extra_500.jsonl \
  --seed "$SEED"

echo "[6/8] Build held-out stem split -> data/splits/stems_seed9172"
uv run python -m data_tools.make_stem_split \
  --anchors data/anchors/anchors_500.jsonl \
  --neutrals data/neutrals/neutrals_1000.jsonl \
  --out_dir data/splits/stems_seed9172 \
  --heldout_frac "${HELDOUT_FRAC:-0.2}" \
  --seed "$SEED"

echo "[7/8] Build TokSpill benchmark -> data/tokspill/tokspill_seed9172.jsonl"
uv run python -m data_tools.build_tokspill \
  --anchors data/anchors/anchors_500.jsonl \
  --neutrals data/neutrals/neutrals_1000.jsonl \
  --train_stems data/splits/stems_seed9172/train_stems.json \
  --heldout_stems data/splits/stems_seed9172/heldout_stems.json \
  --out data/tokspill/tokspill_seed9172.jsonl \
  --seed "$SEED"

echo "[8/8] Build perturbed attack set -> data/eval/attack_perturbed_seed9172.jsonl"
uv run python -m data_tools.perturb_attacks \
  --input data/eval/attack_extra_500.jsonl \
  --output data/eval/attack_perturbed_seed9172.jsonl \
  --stems data/splits/stems_seed9172/heldout_stems.json \
  --seed "$SEED" \
  --include_original

echo "[done] Local curated data is ready under data/."
