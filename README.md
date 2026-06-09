# Alignment-Aware Tokenization

Code for experiments on hazard-aware tokenization, subword spillover, drift probes,
BPE merge search, and lightweight drift-regularized LoRA adaptation.

The current repository is organized around BPE-family experiments, with native
BPE/rank-table editing for Pythia, LLaMA 3, and Qwen2 style fast tokenizers, plus
SentencePiece tokenizer-prior experiments for SentencePiece tokenizers such as
Mistral 7B v0.1.

## Setup

Use `uv` from the repository root:

```bash
uv sync
```

Curate the local data files, then run commands through the project environment:

```bash
uv run bash scripts/sh/curate_data.sh
uv run python -m scripts.validate_data \
  data/anchors/anchors_500.jsonl \
  data/neutrals/neutrals_1000.jsonl \
  data/eval/attack_extra_500.jsonl \
  data/eval/benign_1500.jsonl
```

If `uv` is not installed:

```bash
python -m pip install uv
uv sync
```

## Sanity Check

```bash
uv run bash scripts/sh/curate_data.sh
uv run bash scripts/sh/run_all.sh lightweight
```

Generated/curated experiment data is not tracked in git. The curation command
rebuilds local JSONL data under `data/`; use `QUICK=1` for a small smoke-test
dataset. The lightweight run validates the local data snapshot and compiles the
Python entry points.

See `data/MANIFEST.md` for expected generated files and provenance notes.

## Reproducibility Entry Points

One command entry point:

```bash
uv run bash scripts/sh/run_all.sh lightweight   # checks and stem split
uv run bash scripts/sh/run_all.sh internal      # full internal pipeline
uv run bash scripts/sh/run_all.sh acceptance    # external benchmark pipeline
uv run bash scripts/sh/run_all.sh full          # internal + acceptance
```

The shell entry points live under `scripts/sh/`; Python utilities remain under
`scripts/`.

Heavy regeneration path for the current diagnostic pipeline, equivalent to
`run_all.sh internal`:

```bash
AAT_HEAVY=1 uv run bash scripts/sh/reproduce_main.sh
```

Acceptance-target safety evaluation requires external benchmarks and a judge:

```bash
JUDGE_MODEL=your-org/your-harmfulness-judge \
HARMBENCH_DIR=/path/to/HarmBench \
HARMBENCH_CMD='cd /path/to/HarmBench && bash your_eval_command.sh' \
uv run bash scripts/sh/run_acceptance_evals.sh
```

The internal jailbreak proxy is a diagnostic. A final safety claim should be
validated with external safety and over-refusal benchmarks.

Family-native BPE/token-rank interventions for BPE-style tokenizers:

```bash
uv run bash scripts/sh/run_native_bpe_search.sh llama3_8b
uv run bash scripts/sh/run_native_bpe_search.sh qwen2_7b
uv run bash scripts/sh/run_lora_drift_native_bpe.sh llama3_8b
uv run bash scripts/sh/run_lora_drift_native_bpe.sh qwen2_7b
```

Set `RUN_BASELINES=1` on `run_native_bpe_search.sh` to generate frequency,
random, and shuffled-stem matched baselines. Probe and adapter scripts
automatically use `data/splits/stems_seed9172/*_train_stems.jsonl` when present.

## Main Commands

Create a held-out stem split:

```bash
uv run python -m data_tools.make_stem_split \
  --anchors data/anchors/anchors_500.jsonl \
  --out_dir data/splits/stems_seed9172 \
  --heldout_frac 0.2 \
  --seed 9172
```

Build the TokSpill benchmark:

```bash
uv run bash scripts/sh/run_tokspill.sh
```

Build perturbed attack prompts for downstream robustness diagnostics:

```bash
uv run bash scripts/sh/run_perturb_attacks.sh
```

Train hazard probes:

```bash
uv run bash scripts/sh/run_probes_pythia.sh
```

Train drift-LoRA for Pythia 410M:

```bash
uv run bash scripts/sh/run_lora_drift.sh \
  configs/pythia410m.yml \
  probes/pythia410m_layer10.npy \
  adapters/pythia410m-lora-drift
```

Train drift-LoRA for Pythia 1.4B:

```bash
uv run bash scripts/sh/run_lora_drift_pythia1_4b.sh \
  probes/pythia1_4b_layer11.npy \
  adapters/pythia1_4b-lora-drift
```

Run hazard-aware BPE search:

```bash
uv run bash scripts/sh/run_bpe_search.sh \
  EleutherAI/pythia-410m \
  EleutherAI/pythia-410m \
  probes/pythia410m_layer10.npy
```

Run matched BPE baselines:

```bash
uv run bash scripts/sh/run_bpe_baselines.sh \
  EleutherAI/pythia-410m \
  EleutherAI/pythia-410m \
  probes/pythia410m_layer10.npy
```

Run held-out tokenizer spillover metrics:

```bash
uv run bash scripts/sh/run_tokenizer_metrics.sh data/tokspill/tokspill_seed9172.jsonl
```

Run label-efficiency probes:

```bash
uv run bash scripts/sh/run_label_efficiency.sh \
  configs/pythia410m.yml \
  outputs/label_efficiency_pythia410m.json \
  50 100 300
```

Run evaluations:

```bash
uv run bash scripts/sh/run_eval_all.sh \
  probes/pythia410m_layer10.npy \
  probes/pythia1_4b_layer11.npy
```

Optional external-judge safety diagnostic:

```bash
JUDGE_MODEL=your-org/your-harmfulness-judge \
uv run bash scripts/sh/run_judge_eval.sh \
  EleutherAI/pythia-410m \
  probes/pythia410m_layer10.npy \
  data/eval/attack_perturbed_seed9172.jsonl \
  outputs/jailbreak_pythia410m_judge.json
```

For a generative/chat-style local judge instead of a sequence classifier:

```bash
JUDGE_TYPE=causal_lm \
JUDGE_MODEL=meta-llama/Llama-Guard-3-8B \
uv run bash scripts/sh/run_judge_eval.sh \
  EleutherAI/pythia-410m \
  probes/pythia410m_layer10.npy \
  data/eval/attack_perturbed_seed9172.jsonl \
  outputs/jailbreak_pythia410m_judge.json
```

Optional SentencePiece tokenizer-prior experiment. Use Mistral as the
family-native SPM path; LLaMA/Qwen SPM commands are stress tests only.

```bash
uv run bash scripts/sh/run_spm_priors.sh mistral7b
```

Optional SPM stress-test commands:

```bash
uv run bash scripts/sh/run_spm_priors.sh llama3
uv run bash scripts/sh/run_spm_priors.sh qwen2_7b
uv run bash scripts/sh/run_lora_drift_spm.sh llama3_8b
uv run bash scripts/sh/run_lora_drift_spm.sh qwen2_7b
```

Optional SentencePiece-family probes and drift-LoRA:

```bash
uv run bash scripts/sh/run_probes_spm.sh
uv run bash scripts/sh/run_lora_drift_spm.sh mistral7b
```

The SPM configs keep `model_name` as the base model and set `tokenizer_name` to
the exported tokenizer folder, so `eval.eval_perplexity`, `eval.eval_drift`, and
`eval.eval_jailbreak` can evaluate tokenizer-plus-adapter variants directly.

Collect JSON summaries into CSV:

```bash
uv run python -m scripts.collect_results \
  outputs/*.json runs/*/*summary.json tokenizers/*/bpe_search_log.json \
  --out outputs/results.csv
```

Create paper-ready tokenizer metric tables:

```bash
uv run python -m scripts.make_paper_tables \
  --tokenizer_json outputs/tokspill_*.json \
  --out_prefix outputs/table_tokenizer
```

More detailed command variants are in [EXPERIMENTS.md](EXPERIMENTS.md).

## Repository Layout

- `configs/`: model and evaluation configs.
- `data/`: local dataset snapshots.
- `data_tools/`: data curation and stem-split utilities.
- `eval/`: perplexity, drift, segmentation stability, and jailbreak-proxy evals.
- `models/`: probe training, drift-LoRA, and embedding remapping.
- `scripts/sh/`: runnable experiment entry points.
- `tokenizers/`: BPE search and SentencePiece-prior tokenizer code.
- `utils/`: shared seeding and data loading helpers.

Generated outputs are ignored by git: `adapters/`, `outputs/`, `runs/`,
`__pycache__/`, tokenizer artifacts, and package metadata.
