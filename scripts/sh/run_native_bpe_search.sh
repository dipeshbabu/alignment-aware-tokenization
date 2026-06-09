#!/usr/bin/env bash
set -euo pipefail

# Family-native BPE/rank-table tokenizer search for HF fast BPE tokenizers.
# Supported families here use tokenizer.json model.type=BPE in Transformers.
#
# Usage:
#   uv run bash scripts/sh/run_native_bpe_search.sh [pythia410m|pythia1_4b|llama3_8b|qwen2_7b]
#
# Set RUN_BASELINES=1 to also run frequency/random/shuffled matched baselines.

FAMILY=${1:-pythia410m}
SPLIT_DIR=${AAT_SPLIT_DIR:-data/splits/stems_seed9172}
ANCHORS_TRAIN="$SPLIT_DIR/anchors_train_stems.jsonl"
ANCHORS_HELDOUT="$SPLIT_DIR/anchors_heldout_stems.jsonl"
NEUTRALS_TRAIN="$SPLIT_DIR/neutrals_train_stems.jsonl"
NEUTRALS_HELDOUT="$SPLIT_DIR/neutrals_heldout_stems.jsonl"

if [[ ! -f "$ANCHORS_TRAIN" || ! -f "$NEUTRALS_TRAIN" ]]; then
  echo "[split] building held-out stem split at $SPLIT_DIR"
  uv run python -m data_tools.make_stem_split \
    --anchors data/anchors/anchors_500.jsonl \
    --neutrals data/neutrals/neutrals_1000.jsonl \
    --out_dir "$SPLIT_DIR" \
    --heldout_frac "${HELDOUT_FRAC:-0.2}" \
    --seed "${SEED:-9172}"
fi

case "$FAMILY" in
  pythia410m)
    BASE_TOK=${BASE_TOK:-EleutherAI/pythia-410m}
    MODEL_ID=${MODEL_ID:-EleutherAI/pythia-410m}
    PROBE=${PROBE:-probes/pythia410m_layer10.npy}
    LAYER=${LAYER:-10}
    ;;
  pythia1_4b)
    BASE_TOK=${BASE_TOK:-EleutherAI/pythia-1.4b}
    MODEL_ID=${MODEL_ID:-EleutherAI/pythia-1.4b}
    PROBE=${PROBE:-probes/pythia1_4b_layer11.npy}
    LAYER=${LAYER:-11}
    ;;
  llama3_8b)
    BASE_TOK=${BASE_TOK:-meta-llama/Meta-Llama-3-8B}
    MODEL_ID=${MODEL_ID:-meta-llama/Meta-Llama-3-8B}
    PROBE=${PROBE:-probes/llama3_8b_layer16.npy}
    LAYER=${LAYER:-16}
    ;;
  qwen2_7b)
    BASE_TOK=${BASE_TOK:-Qwen/Qwen2-7B}
    MODEL_ID=${MODEL_ID:-Qwen/Qwen2-7B}
    PROBE=${PROBE:-probes/qwen2_7b_layer16.npy}
    LAYER=${LAYER:-16}
    ;;
  *)
    echo "Usage: $0 [pythia410m|pythia1_4b|llama3_8b|qwen2_7b]" >&2
    exit 2
    ;;
esac

OUT_BASE=${OUT_BASE:-tokenizers/native_bpe_${FAMILY}}
mkdir -p "$OUT_BASE"

run_one() {
  local baseline=$1
  local out_dir=$2
  local baseline_args=()
  if [[ "$baseline" != "hazard" ]]; then
    baseline_args=(--baseline "$baseline")
  fi

  uv run python -m tokenizers.bpe_search \
    --base_tokenizer "$BASE_TOK" \
    --model_name "$MODEL_ID" \
    --anchors "$ANCHORS_TRAIN" \
    --neutrals "$NEUTRALS_TRAIN" \
    --eval_anchors "$ANCHORS_HELDOUT" \
    --eval_neutrals "$NEUTRALS_HELDOUT" \
    --u_dev_dataset data/unlabeled/u_dev.jsonl \
    --u_dev_size "${U_DEV_SIZE:-20000}" \
    --probe "$PROBE" \
    --rounds "${ROUNDS:-5}" \
    --prune_k "${PRUNE_K:-30}" \
    --min_benign_hits "${MIN_BENIGN_HITS:-5}" \
    --alpha "${ALPHA:-0.7}" \
    --beta "${BETA:-0.1}" \
    --gamma "${GAMMA:-0.0}" \
    --drift_layer "$LAYER" \
    --warmup_steps "${WARMUP_STEPS:-150}" \
    "${baseline_args[@]}" \
    --out "$out_dir"
}

echo "[native-bpe] family=$FAMILY model=$MODEL_ID tokenizer=$BASE_TOK"
run_one hazard "$OUT_BASE/hazard"

if [[ "${RUN_BASELINES:-0}" == "1" ]]; then
  run_one frequency "$OUT_BASE/frequency"
  run_one random "$OUT_BASE/random"
  run_one shuffled "$OUT_BASE/shuffled"
fi

echo "[done] native BPE artifacts under $OUT_BASE"
