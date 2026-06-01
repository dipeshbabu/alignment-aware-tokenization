#!/usr/bin/env bash
set -euo pipefail

MODEL=${1:-EleutherAI/pythia-410m}
PROBE=${2:-probes/pythia410m_layer10.npy}
ATTACK=${3:-data/eval/attack_perturbed_seed9172.jsonl}
OUT=${4:-outputs/jailbreak_pythia410m_judge.json}

if [ -z "${JUDGE_MODEL:-}" ]; then
  echo "Set JUDGE_MODEL to a local or Hugging Face safety judge." >&2
  echo "Classifier example: JUDGE_MODEL=your-org/your-harmfulness-classifier bash scripts/run_judge_eval.sh" >&2
  echo "Causal-LM example: JUDGE_TYPE=causal_lm JUDGE_MODEL=your-org/your-chat-judge bash scripts/run_judge_eval.sh" >&2
  exit 2
fi

python -m eval.eval_jailbreak \
  --config configs/llm_eval.yml \
  --attack_file "$ATTACK" \
  --probe "$PROBE" \
  --model_name "$MODEL" \
  --judge_model "$JUDGE_MODEL" \
  --judge_type "${JUDGE_TYPE:-classifier}" \
  --judge_harm_label "${JUDGE_HARM_LABEL:-HARMFUL}" \
  --judge_threshold "${JUDGE_THRESHOLD:-0.5}" \
  --judge_device "${JUDGE_DEVICE:-cpu}" \
  --judge_bs "${JUDGE_BS:-4}" \
  --judge_max_new_tokens "${JUDGE_MAX_NEW_TOKENS:-24}" \
  --atk_n "${ATK_N:-768}" \
  --benign_n "${BENIGN_N:-256}" \
  --calib_n "${CALIB_N:-256}" \
  --score_mode resp_only \
  --benign_fpr 0.01 \
  --greedy \
  --max_new_tokens "${MAX_NEW_TOKENS:-64}" \
  --gen_bs "${GEN_BS:-8}" \
  --batch_size "${BATCH_SIZE:-16}" \
  --refusal_window 200 \
  --dedup_attacks \
  --out "$OUT"
