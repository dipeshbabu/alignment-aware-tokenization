# Experiment Commands

Run commands from the repository root.

## 0. Setup and Sanity

```bash
uv sync
uv run bash scripts/sh/curate_data.sh
uv run bash scripts/sh/run_all.sh lightweight
```

Generated/curated data is not tracked in git. Use `QUICK=1` for a small local
smoke-test dataset before running full curation.

Use the single shell entry point for the full pipeline:

```bash
uv run bash scripts/sh/run_all.sh lightweight   # checks and stem split
uv run bash scripts/sh/run_all.sh internal      # full internal pipeline
uv run bash scripts/sh/run_all.sh acceptance    # external benchmark pipeline
uv run bash scripts/sh/run_all.sh full          # internal + acceptance
```

The shell entry points live under `scripts/sh/`; Python utilities remain under
`scripts/`. Use the heavy path only when model downloads, GPU training, and full
evaluation are intended:

```bash
AAT_HEAVY=1 uv run bash scripts/sh/reproduce_main.sh
```

Before paper runs, verify that locally curated counts match the paper claims:

```bash
uv run python -m scripts.validate_data \
  data/anchors/anchors_500.jsonl \
  data/neutrals/neutrals_1000.jsonl \
  data/eval/attack_extra_500.jsonl \
  data/eval/benign_1500.jsonl
```

## 1. Held-Out Stem Split

```bash
uv run python -m data_tools.make_stem_split \
  --anchors data/anchors/anchors_500.jsonl \
  --out_dir data/splits/stems_seed9172 \
  --heldout_frac 0.2 \
  --seed 9172
```

Use `data/splits/stems_seed9172/anchors_train_stems.jsonl` for tokenizer/probe
construction and `anchors_heldout_stems.jsonl` only for held-out analysis.

Build TokSpill:

```bash
uv run bash scripts/sh/run_tokspill.sh \
  data/splits/stems_seed9172 \
  data/tokspill/tokspill_seed9172.jsonl
```

Build perturbed attack prompts for optional downstream diagnostics:

```bash
uv run bash scripts/sh/run_perturb_attacks.sh \
  data/splits/stems_seed9172 \
  data/eval/attack_perturbed_seed9172.jsonl
```

## 2. Hazard Probes

Single model:

```bash
uv run bash scripts/sh/run_probe.sh configs/pythia410m.yml probes/pythia410m_layer10.npy
```

Both Pythia models:

```bash
uv run bash scripts/sh/run_probes_pythia.sh
```

## 3. Drift-LoRA Adaptation

Pythia 410M:

```bash
uv run bash scripts/sh/run_lora_drift.sh \
  configs/pythia410m.yml \
  probes/pythia410m_layer10.npy \
  adapters/pythia410m-lora-drift
```

Pythia 1.4B:

```bash
uv run bash scripts/sh/run_lora_drift_pythia1_4b.sh \
  probes/pythia1_4b_layer11.npy \
  adapters/pythia1_4b-lora-drift
```

## 4. BPE Tokenizer Search

Hazard-aware BPE search:

```bash
uv run bash scripts/sh/run_bpe_search.sh \
  EleutherAI/pythia-410m \
  EleutherAI/pythia-410m \
  probes/pythia410m_layer10.npy \
  data/splits/stems_seed9172/anchors_train_stems.jsonl \
  tokenizers/bpe_searched
```

Matched baselines:

```bash
uv run bash scripts/sh/run_bpe_baselines.sh \
  EleutherAI/pythia-410m \
  EleutherAI/pythia-410m \
  probes/pythia410m_layer10.npy \
  data/splits/stems_seed9172/anchors_train_stems.jsonl
```

Tokenizer metrics on held-out stems:

```bash
uv run bash scripts/sh/run_tokenizer_metrics.sh data/tokspill/tokspill_seed9172.jsonl
```

Family-native BPE/rank-table search for BPE-style large backbones:

```bash
RUN_BASELINES=1 uv run bash scripts/sh/run_native_bpe_search.sh llama3_8b
RUN_BASELINES=1 uv run bash scripts/sh/run_native_bpe_search.sh qwen2_7b
uv run bash scripts/sh/run_lora_drift_native_bpe.sh llama3_8b
uv run bash scripts/sh/run_lora_drift_native_bpe.sh qwen2_7b
```

The native BPE search consumes `data/splits/stems_seed9172` when present and
logs held-out drift metrics under `heldout_eval` in each `bpe_search_log.json`.

BPE-dropout baseline tokenizer:

```bash
uv run python -m tokenizers.bpe_dropout \
  --base_tokenizer EleutherAI/pythia-410m \
  --dropout 0.1 \
  --out tokenizers/bpe_dropout_pythia410m
```

Label-efficiency probes:

```bash
uv run bash scripts/sh/run_label_efficiency.sh \
  configs/pythia410m.yml \
  outputs/label_efficiency_pythia410m.json \
  50 100 300
```

## 5. Optional SentencePiece Priors

```bash
uv run bash scripts/sh/run_spm_priors.sh mistral7b
```

Use Mistral as the family-native SPM experiment. LLaMA/Qwen SPM runs are
retokenization stress tests only; use `run_native_bpe_search.sh` for their final
family-native tokenizer edits.

SPM-family probes and drift-LoRA:

```bash
uv run bash scripts/sh/run_probes_spm.sh
uv run bash scripts/sh/run_lora_drift_spm.sh mistral7b
```

SPM stress-test commands:

```bash
uv run bash scripts/sh/run_spm_priors.sh llama3
uv run bash scripts/sh/run_spm_priors.sh qwen2_7b
uv run bash scripts/sh/run_lora_drift_spm.sh llama3_8b
uv run bash scripts/sh/run_lora_drift_spm.sh qwen2_7b
```

## 6. Evaluation

Base model quality, drift, segmentation stability, and proxy jailbreak eval:

```bash
uv run bash scripts/sh/run_eval_all.sh \
  probes/pythia410m_layer10.npy \
  probes/pythia1_4b_layer11.npy
```

Individual commands:

```bash
uv run python -m eval.eval_perplexity --config configs/pythia410m.yml

uv run python -m eval.eval_drift \
  --config configs/pythia410m.yml \
  --probe probes/pythia410m_layer10.npy \
  --out_dir runs/pythia410m_base

uv run python -m eval.seg_stability \
  --tokenizer EleutherAI/pythia-410m \
  --texts data/unlabeled/u_dev.jsonl \
  --max_texts 400 \
  --out outputs/segstab_pythia410m.json

uv run python -m eval.eval_jailbreak \
  --config configs/llm_eval.yml \
  --attack_file data/eval/attack_perturbed_seed9172.jsonl \
  --probe probes/pythia410m_layer10.npy \
  --model_name EleutherAI/pythia-410m \
  --atk_n 256 --benign_n 256 --calib_n 256 \
  --score_mode resp_only --benign_fpr 0.01 \
  --greedy --max_new_tokens 64 --gen_bs 8 --batch_size 16 \
  --refusal_window 200 --dedup_attacks \
  --out outputs/jailbreak_pythia410m_proxy.json
```

SPM combined variants use configs with `model_name` set to the base model and
`tokenizer_name` set to the exported tokenizer folder:

```bash
uv run python -m eval.eval_perplexity --config configs/mistral7b_spm.yml

uv run python -m eval.eval_drift \
  --config configs/mistral7b_spm.yml \
  --probe probes/mistral7b_layer16.npy \
  --out_dir runs/mistral7b_spm_drift

uv run python -m eval.eval_jailbreak \
  --config configs/mistral7b_spm.yml \
  --attack_file data/eval/attack_perturbed_seed9172.jsonl \
  --probe probes/mistral7b_layer16.npy \
  --adapter adapters/mistral7b-spm-lora-drift \
  --atk_n 256 --benign_n 256 --calib_n 256 \
  --score_mode resp_only --benign_fpr 0.01 \
  --greedy --max_new_tokens 64 --gen_bs 4 --batch_size 8 \
  --refusal_window 200 --dedup_attacks \
  --out outputs/jailbreak_mistral7b_spm_proxy.json
```

External judge diagnostic:

```bash
JUDGE_MODEL=your-org/your-harmfulness-judge \
uv run bash scripts/sh/run_judge_eval.sh \
  EleutherAI/pythia-410m \
  probes/pythia410m_layer10.npy \
  data/eval/attack_perturbed_seed9172.jsonl \
  outputs/jailbreak_pythia410m_judge.json
```

Generative/chat-style judge diagnostic:

```bash
JUDGE_TYPE=causal_lm \
JUDGE_MODEL=meta-llama/Llama-Guard-3-8B \
uv run bash scripts/sh/run_judge_eval.sh \
  EleutherAI/pythia-410m \
  probes/pythia410m_layer10.npy \
  data/eval/attack_perturbed_seed9172.jsonl \
  outputs/jailbreak_pythia410m_judge.json
```

Acceptance-target safety evaluation should use external benchmark checkouts, not
only the internal proxy:

```bash
JUDGE_MODEL=your-org/your-harmfulness-judge \
HARMBENCH_DIR=/path/to/HarmBench \
JAILBREAKBENCH_DIR=/path/to/jailbreakbench \
HARMBENCH_CMD='cd /path/to/HarmBench && bash your_eval_command.sh' \
JAILBREAKBENCH_CMD='cd /path/to/jailbreakbench && bash your_eval_command.sh' \
uv run bash scripts/sh/run_acceptance_evals.sh
```

The acceptance script verifies that external benchmark commands write
`outputs/acceptance/harmbench.json`, `jailbreakbench.json`, or `xstest.json`,
then normalizes available metrics into `outputs/acceptance/acceptance_summary.csv`.

## 7. Collect Tables

```bash
uv run python -m scripts.collect_results \
  outputs/*.json runs/*/*summary.json tokenizers/*/bpe_search_log.json \
  --out outputs/results.csv
```

Paper tokenizer table:

```bash
uv run python -m scripts.make_paper_tables \
  --tokenizer_json outputs/tokspill_*.json \
  --out_prefix outputs/table_tokenizer
```
