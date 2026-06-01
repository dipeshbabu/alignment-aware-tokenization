#!/usr/bin/env bash
set -euo pipefail

python -m scripts.validate_data \
  data/anchors/anchors_500.jsonl \
  data/neutrals/neutrals_1000.jsonl \
  data/eval/attack_extra_500.jsonl \
  data/eval/benign_1500.jsonl

python -m py_compile \
  models/lora_drift.py \
  models/probe.py \
  eval/eval_drift.py \
  eval/eval_jailbreak.py \
  eval/label_efficiency.py \
  eval/eval_perplexity.py \
  eval/seg_stability.py \
  eval/tokenizer_spillover.py \
  tokenizers/bpe_search.py \
  tokenizers/spm_priors.py \
  data_tools/build_tokspill.py \
  data_tools/make_stem_split.py \
  data_tools/perturb_attacks.py \
  scripts/collect_results.py \
  scripts/make_paper_tables.py \
  scripts/validate_data.py

bash -n \
  scripts/run_label_efficiency.sh \
  scripts/run_judge_eval.sh \
  scripts/run_bpe_baselines.sh \
  scripts/run_lora_drift_spm.sh \
  scripts/run_probes_spm.sh \
  scripts/run_spm_priors.sh \
  scripts/run_tokenizer_metrics.sh

bash scripts/run_judge_eval.sh >/tmp/aat_judge_help.log 2>&1 && exit 1 || true
grep -q "Set JUDGE_MODEL" /tmp/aat_judge_help.log
