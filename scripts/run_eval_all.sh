#!/usr/bin/env bash
set -euo pipefail

PROBE_410M=${1:-probes/pythia410m_layer10.npy}
PROBE_14B=${2:-probes/pythia1_4b_layer11.npy}
PROBE_GEMMA=${3:-probes/gemma2b_layer10.npy}

python -m eval.eval_perplexity --config configs/pythia410m.yml
python -m eval.eval_drift --config configs/pythia410m.yml --probe "$PROBE_410M"
python -m eval.seg_stability --tokenizer EleutherAI/pythia-410m --texts data/unlabeled/u_dev.jsonl --max_texts 400 --out outputs/segstab_pythia410m.json

python -m eval.eval_perplexity --config configs/pythia1_4b.yml
python -m eval.eval_drift --config configs/pythia1_4b.yml --probe "$PROBE_14B"
python -m eval.seg_stability --tokenizer EleutherAI/pythia-1.4b --texts data/unlabeled/u_dev.jsonl --max_texts 400 --out outputs/segstab_pythia1_4b.json

python -m eval.eval_perplexity --config configs/gemma2b.yml
python -m eval.eval_drift --config configs/gemma2b.yml --probe "$PROBE_GEMMA"
python -m eval.seg_stability --tokenizer google/gemma-2b --texts data/unlabeled/u_dev.jsonl --max_texts 400 --out outputs/segstab_gemma2b.json

HF_HUB_ENABLE_HF_TRANSFER=0 python -m eval.eval_jailbreak \
  --config configs/llm_eval.yml \
  --probe "$PROBE_GEMMA" \
  --model_name google/gemma-2b \
  --judge_model google/shieldgemma-2-2b \
  --judge_harm_label HARMFUL \
  --judge_device cpu \
  --atk_n 1000 --benign_n 1500 --calib_n 256 \
  --score_mode resp_only --benign_fpr 0.01 \
  --greedy --max_new_tokens 64 --gen_bs 8 --batch_size 16 \
  --refusal_window 200 --dedup_attacks