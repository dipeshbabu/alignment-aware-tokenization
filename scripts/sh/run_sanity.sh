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
  tokenizers/bpe_dropout.py \
  tokenizers/bpe_search.py \
  tokenizers/spm_priors.py \
  data_tools/build_tokspill.py \
  data_tools/make_stem_split.py \
  data_tools/perturb_attacks.py \
  scripts/collect_results.py \
  scripts/make_paper_tables.py \
  scripts/merge_acceptance_summaries.py \
  scripts/run_lora_drift_native_bpe.py \
  scripts/run_xstest_external.py \
  scripts/summarize_acceptance_results.py \
  scripts/validate_data.py

bash -n \
  scripts/sh/run_label_efficiency.sh \
  scripts/sh/run_judge_eval.sh \
  scripts/sh/run_bpe_baselines.sh \
  scripts/sh/run_lora_drift_spm.sh \
  scripts/sh/run_lora_drift_native_bpe.sh \
  scripts/sh/run_probes_spm.sh \
  scripts/sh/run_spm_priors.sh \
  scripts/sh/run_tokenizer_metrics.sh \
  scripts/sh/reproduce_main.sh \
  scripts/sh/run_all.sh \
  scripts/sh/run_acceptance_evals.sh \
  scripts/sh/run_acceptance_matrix.sh \
  scripts/sh/curate_data.sh \
  scripts/sh/run_native_bpe_search.sh

bash scripts/sh/run_judge_eval.sh >/tmp/aat_judge_help.log 2>&1 && exit 1 || true
grep -q "Set JUDGE_MODEL" /tmp/aat_judge_help.log
