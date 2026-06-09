#!/usr/bin/env bash
set -euo pipefail

# Run paper-grade acceptance evaluations over the method comparison matrix.
#
# External benchmark commands receive these exported variables from
# run_acceptance_evals.sh:
#   AAT_SETTING_NAME, AAT_MODEL_NAME, AAT_TOKENIZER_NAME, AAT_ADAPTER,
#   AAT_PROBE, AAT_OUT_DIR, AAT_JUDGE_MODEL
#
# Example:
#   source configs/acceptance.env
#   uv run bash scripts/sh/run_acceptance_matrix.sh pythia410m_base pythia410m_native_bpe_lora

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

SETTINGS=("$@")
if [[ "${#SETTINGS[@]}" -eq 0 ]]; then
  SETTINGS=(
    pythia410m_base
    pythia410m_drift
    pythia410m_native_bpe
    pythia410m_native_bpe_lora
    pythia1_4b_base
    pythia1_4b_drift
    pythia1_4b_native_bpe
    pythia1_4b_native_bpe_lora
    llama3_8b_base
    llama3_8b_native_bpe
    llama3_8b_native_bpe_lora
    qwen2_7b_base
    qwen2_7b_native_bpe
    qwen2_7b_native_bpe_lora
    mistral7b_base
    mistral7b_spm_lora
  )
fi

setting_env() {
  local setting=$1
  ADAPTER=""
  case "$setting" in
    pythia410m_base)
      MODEL_NAME="EleutherAI/pythia-410m"
      TOKENIZER_NAME="EleutherAI/pythia-410m"
      PROBE="probes/pythia410m_layer10.npy"
      ;;
    pythia410m_drift)
      MODEL_NAME="EleutherAI/pythia-410m"
      TOKENIZER_NAME="EleutherAI/pythia-410m"
      ADAPTER="adapters/pythia410m-lora-drift"
      PROBE="probes/pythia410m_layer10.npy"
      ;;
    pythia410m_native_bpe)
      MODEL_NAME="EleutherAI/pythia-410m"
      TOKENIZER_NAME="tokenizers/native_bpe_pythia410m/hazard"
      PROBE="probes/pythia410m_layer10.npy"
      ;;
    pythia410m_native_bpe_lora)
      MODEL_NAME="EleutherAI/pythia-410m"
      TOKENIZER_NAME="tokenizers/native_bpe_pythia410m/hazard"
      ADAPTER="adapters/pythia410m-native-bpe-lora-drift"
      PROBE="probes/pythia410m_layer10.npy"
      ;;
    pythia1_4b_base)
      MODEL_NAME="EleutherAI/pythia-1.4b"
      TOKENIZER_NAME="EleutherAI/pythia-1.4b"
      PROBE="probes/pythia1_4b_layer11.npy"
      ;;
    pythia1_4b_drift)
      MODEL_NAME="EleutherAI/pythia-1.4b"
      TOKENIZER_NAME="EleutherAI/pythia-1.4b"
      ADAPTER="adapters/pythia1_4b-lora-drift"
      PROBE="probes/pythia1_4b_layer11.npy"
      ;;
    pythia1_4b_native_bpe)
      MODEL_NAME="EleutherAI/pythia-1.4b"
      TOKENIZER_NAME="tokenizers/native_bpe_pythia1_4b/hazard"
      PROBE="probes/pythia1_4b_layer11.npy"
      ;;
    pythia1_4b_native_bpe_lora)
      MODEL_NAME="EleutherAI/pythia-1.4b"
      TOKENIZER_NAME="tokenizers/native_bpe_pythia1_4b/hazard"
      ADAPTER="adapters/pythia1_4b-native-bpe-lora-drift"
      PROBE="probes/pythia1_4b_layer11.npy"
      ;;
    llama3_8b_base)
      MODEL_NAME="meta-llama/Meta-Llama-3-8B"
      TOKENIZER_NAME="meta-llama/Meta-Llama-3-8B"
      PROBE="probes/llama3_8b_layer16.npy"
      ;;
    llama3_8b_native_bpe)
      MODEL_NAME="meta-llama/Meta-Llama-3-8B"
      TOKENIZER_NAME="tokenizers/native_bpe_llama3_8b/hazard"
      PROBE="probes/llama3_8b_layer16.npy"
      ;;
    llama3_8b_native_bpe_lora)
      MODEL_NAME="meta-llama/Meta-Llama-3-8B"
      TOKENIZER_NAME="tokenizers/native_bpe_llama3_8b/hazard"
      ADAPTER="adapters/llama3_8b-native-bpe-lora-drift"
      PROBE="probes/llama3_8b_layer16.npy"
      ;;
    qwen2_7b_base)
      MODEL_NAME="Qwen/Qwen2-7B"
      TOKENIZER_NAME="Qwen/Qwen2-7B"
      PROBE="probes/qwen2_7b_layer16.npy"
      ;;
    qwen2_7b_native_bpe)
      MODEL_NAME="Qwen/Qwen2-7B"
      TOKENIZER_NAME="tokenizers/native_bpe_qwen2_7b/hazard"
      PROBE="probes/qwen2_7b_layer16.npy"
      ;;
    qwen2_7b_native_bpe_lora)
      MODEL_NAME="Qwen/Qwen2-7B"
      TOKENIZER_NAME="tokenizers/native_bpe_qwen2_7b/hazard"
      ADAPTER="adapters/qwen2_7b-native-bpe-lora-drift"
      PROBE="probes/qwen2_7b_layer16.npy"
      ;;
    mistral7b_base)
      MODEL_NAME="mistralai/Mistral-7B-v0.1"
      TOKENIZER_NAME="mistralai/Mistral-7B-v0.1"
      PROBE="probes/mistral7b_layer16.npy"
      ;;
    mistral7b_spm_lora)
      MODEL_NAME="mistralai/Mistral-7B-v0.1"
      TOKENIZER_NAME="tokenizers/spm_hazard/spm_mistral7b_hazard_hf"
      ADAPTER="adapters/mistral7b-spm-lora-drift"
      PROBE="probes/mistral7b_layer16.npy"
      ;;
    *)
      echo "Unknown acceptance setting: $setting" >&2
      exit 2
      ;;
  esac
}

for setting in "${SETTINGS[@]}"; do
  setting_env "$setting"
  out_dir="${ACCEPTANCE_ROOT:-outputs/acceptance}/$setting"
  echo "[matrix] $setting -> $out_dir"
  SETTING_NAME="$setting" \
  MODEL_NAME="$MODEL_NAME" \
  TOKENIZER_NAME="$TOKENIZER_NAME" \
  ADAPTER="$ADAPTER" \
  PROBE="$PROBE" \
  OUT_DIR="$out_dir" \
    uv run bash scripts/sh/run_acceptance_evals.sh
done

summary_inputs=()
for setting in "${SETTINGS[@]}"; do
  file="${ACCEPTANCE_ROOT:-outputs/acceptance}/$setting/acceptance_summary.csv"
  if [[ -f "$file" ]]; then
    summary_inputs+=("$file")
  fi
done

if [[ "${#summary_inputs[@]}" -gt 0 ]]; then
  uv run python -m scripts.merge_acceptance_summaries \
    "${summary_inputs[@]}" \
    --out "${ACCEPTANCE_ROOT:-outputs/acceptance}/acceptance_matrix_summary.csv"
fi
