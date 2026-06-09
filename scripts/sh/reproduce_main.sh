#!/usr/bin/env bash
set -euo pipefail

# Reproduce the lightweight artifact checks by default. Set AAT_HEAVY=1 to run
# the GPU/network-heavy experiment pipeline.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

mkdir -p outputs runs probes adapters tokenizers

required_data=(
  data/anchors/anchors_500.jsonl
  data/neutrals/neutrals_1000.jsonl
  data/eval/attack_extra_500.jsonl
  data/eval/benign_1500.jsonl
  data/unlabeled/u_train.jsonl
  data/unlabeled/u_dev.jsonl
)

missing_data=()
for path in "${required_data[@]}"; do
  if [[ ! -f "$path" ]]; then
    missing_data+=("$path")
  fi
done

if [[ "${#missing_data[@]}" -gt 0 ]]; then
  printf '[data] Missing generated data files:\n' >&2
  printf '  %s\n' "${missing_data[@]}" >&2
  cat >&2 <<'MSG'

Generated/curated data is not tracked in git. Rebuild it with:

  uv run bash scripts/sh/curate_data.sh

For a small smoke test:

  QUICK=1 uv run bash scripts/sh/curate_data.sh
MSG
  exit 2
fi

echo "[1/4] Sanity checks and data validation"
uv run bash scripts/sh/run_sanity.sh

echo "[split] Ensure held-out stem split exists"
uv run python -m data_tools.make_stem_split \
  --anchors data/anchors/anchors_500.jsonl \
  --neutrals data/neutrals/neutrals_1000.jsonl \
  --out_dir data/splits/stems_seed9172 \
  --heldout_frac 0.2 \
  --seed 9172

if [[ "${AAT_HEAVY:-0}" != "1" ]]; then
  cat <<'MSG'

[stop] Lightweight checks completed.

Set AAT_HEAVY=1 to run the expensive regeneration path:

  AAT_HEAVY=1 uv run bash scripts/sh/reproduce_main.sh

The heavy path trains probes/adapters and evaluates the current artifact
pipeline. External safety benchmarks are intentionally separate because they
require benchmark checkouts and judge/model configuration; see
scripts/sh/run_acceptance_evals.sh.
MSG
  exit 0
fi

echo "[2/4] Train probes"
uv run bash scripts/sh/run_probes_pythia.sh
uv run bash scripts/sh/run_probes_spm.sh

echo "[3/4] Train tokenizer/adapters"
RUN_BASELINES=1 uv run bash scripts/sh/run_native_bpe_search.sh pythia410m
RUN_BASELINES=1 uv run bash scripts/sh/run_native_bpe_search.sh pythia1_4b
RUN_BASELINES=1 uv run bash scripts/sh/run_native_bpe_search.sh llama3_8b
RUN_BASELINES=1 uv run bash scripts/sh/run_native_bpe_search.sh qwen2_7b

uv run python -m tokenizers.bpe_dropout \
  --base_tokenizer EleutherAI/pythia-410m \
  --dropout 0.1 \
  --out tokenizers/bpe_dropout_pythia410m

uv run bash scripts/sh/run_lora_drift.sh \
  configs/pythia410m.yml \
  probes/pythia410m_layer10.npy \
  adapters/pythia410m-lora-drift

uv run bash scripts/sh/run_lora_drift_pythia1_4b.sh \
  probes/pythia1_4b_layer11.npy \
  adapters/pythia1_4b-lora-drift

uv run bash scripts/sh/run_lora_drift_native_bpe.sh pythia410m
uv run bash scripts/sh/run_lora_drift_native_bpe.sh pythia1_4b
uv run bash scripts/sh/run_lora_drift_native_bpe.sh llama3_8b
uv run bash scripts/sh/run_lora_drift_native_bpe.sh qwen2_7b

uv run bash scripts/sh/run_spm_priors.sh mistral7b
uv run bash scripts/sh/run_lora_drift_spm.sh mistral7b

echo "[4/4] Run current diagnostic evaluations"
uv run bash scripts/sh/run_eval_all.sh \
  probes/pythia410m_layer10.npy \
  probes/pythia1_4b_layer11.npy

uv run python -m scripts.collect_results \
  outputs/*.json runs/*/*summary.json tokenizers/*/bpe_search_log.json \
  --out outputs/results.csv

echo "[done] Current diagnostic artifacts are under outputs/ and runs/."
