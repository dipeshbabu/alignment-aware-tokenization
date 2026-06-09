#!/usr/bin/env bash
set -euo pipefail

# Acceptance-target safety evaluation entry point.
#
# This script is a guardrail: proxy jailbreak scores are useful diagnostics, but
# they are not enough for a must-accept safety/tokenization paper. Configure
# external benchmark checkouts and judge settings before running.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

: "${HARMBENCH_DIR:=}"
: "${JAILBREAKBENCH_DIR:=}"
: "${XSTEST_FILE:=}"
: "${JUDGE_MODEL:=}"
: "${SETTING_NAME:=adhoc}"
: "${MODEL_NAME:=EleutherAI/pythia-410m}"
: "${TOKENIZER_NAME:=$MODEL_NAME}"
: "${ADAPTER:=}"
: "${PROBE:=probes/pythia410m_layer10.npy}"
: "${OUT_DIR:=outputs/acceptance}"
: "${HARMBENCH_CMD:=}"
: "${JAILBREAKBENCH_CMD:=}"
: "${XSTEST_CMD:=}"

mkdir -p "$OUT_DIR"

if [[ -z "$JUDGE_MODEL" ]]; then
  cat >&2 <<'MSG'
Set JUDGE_MODEL to an external safety judge before running acceptance evals.
Examples: a local harmfulness classifier, LlamaGuard-style guard model, or a
frontier-model judge wrapper.
MSG
  exit 2
fi

if [[ -z "$HARMBENCH_DIR" && -z "$JAILBREAKBENCH_DIR" ]]; then
  cat >&2 <<'MSG'
Set HARMBENCH_DIR or JAILBREAKBENCH_DIR to a local benchmark checkout.
The current paper proxy is not sufficient for a final safety claim.
MSG
  exit 2
fi

echo "[acceptance] setting=$SETTING_NAME model=$MODEL_NAME tokenizer=$TOKENIZER_NAME adapter=${ADAPTER:-none} judge=$JUDGE_MODEL"

export AAT_SETTING_NAME="$SETTING_NAME"
export AAT_MODEL_NAME="$MODEL_NAME"
export AAT_TOKENIZER_NAME="$TOKENIZER_NAME"
export AAT_ADAPTER="$ADAPTER"
export AAT_PROBE="$PROBE"
export AAT_OUT_DIR="$OUT_DIR"
export AAT_JUDGE_MODEL="$JUDGE_MODEL"

if [[ -n "$HARMBENCH_DIR" ]]; then
  echo "[acceptance] HarmBench checkout: $HARMBENCH_DIR"
  if [[ -z "$HARMBENCH_CMD" ]]; then
    echo "Set HARMBENCH_CMD to the exact HarmBench command that writes $OUT_DIR/harmbench.json" >&2
    exit 3
  fi
  bash -lc "$HARMBENCH_CMD"
  test -f "$OUT_DIR/harmbench.json" || {
    echo "HARMBENCH_CMD completed but $OUT_DIR/harmbench.json was not created." >&2
    exit 4
  }
fi

if [[ -n "$JAILBREAKBENCH_DIR" ]]; then
  echo "[acceptance] JailbreakBench checkout: $JAILBREAKBENCH_DIR"
  if [[ -z "$JAILBREAKBENCH_CMD" ]]; then
    echo "Set JAILBREAKBENCH_CMD to the exact JailbreakBench command that writes $OUT_DIR/jailbreakbench.json" >&2
    exit 3
  fi
  bash -lc "$JAILBREAKBENCH_CMD"
  test -f "$OUT_DIR/jailbreakbench.json" || {
    echo "JAILBREAKBENCH_CMD completed but $OUT_DIR/jailbreakbench.json was not created." >&2
    exit 4
  }
fi

if [[ -n "$XSTEST_FILE" ]]; then
  echo "[acceptance] XSTest file: $XSTEST_FILE"
  if [[ -z "$XSTEST_CMD" ]]; then
    echo "Set XSTEST_CMD to the exact over-refusal command that writes $OUT_DIR/xstest.json" >&2
    exit 3
  fi
  bash -lc "$XSTEST_CMD"
  test -f "$OUT_DIR/xstest.json" || {
    echo "XSTEST_CMD completed but $OUT_DIR/xstest.json was not created." >&2
    exit 4
  }
fi

echo "[acceptance] Also run the internal proxy for mediation diagnostics."
adapter_args=()
if [[ -n "$ADAPTER" ]]; then
  adapter_args=(--adapter "$ADAPTER")
fi

uv run python -m eval.eval_jailbreak \
  --config configs/llm_eval.yml \
  --probe "$PROBE" \
  --model_name "$MODEL_NAME" \
  --tokenizer_name "$TOKENIZER_NAME" \
  "${adapter_args[@]}" \
  --judge_model "$JUDGE_MODEL" \
  --judge_type "${JUDGE_TYPE:-classifier}" \
  --judge_harm_label "${JUDGE_HARM_LABEL:-HARMFUL}" \
  --judge_threshold "${JUDGE_THRESHOLD:-0.5}" \
  --judge_device "${JUDGE_DEVICE:-cpu}" \
  --atk_n "${ATK_N:-256}" --benign_n "${BENIGN_N:-256}" --calib_n "${CALIB_N:-256}" \
  --score_mode resp_only --benign_fpr 0.01 \
  --greedy --max_new_tokens "${MAX_NEW_TOKENS:-64}" \
  --gen_bs "${GEN_BS:-8}" --batch_size "${BATCH_SIZE:-16}" \
  --refusal_window 200 --dedup_attacks \
  --out "$OUT_DIR/internal_proxy_with_external_judge.json"

cat > "$OUT_DIR/acceptance_run_metadata.json" <<JSON
{
  "setting_name": "$SETTING_NAME",
  "model_name": "$MODEL_NAME",
  "tokenizer_name": "$TOKENIZER_NAME",
  "adapter": "$ADAPTER",
  "probe": "$PROBE",
  "judge_model": "$JUDGE_MODEL",
  "judge_type": "${JUDGE_TYPE:-classifier}",
  "harmbench_dir": "$HARMBENCH_DIR",
  "jailbreakbench_dir": "$JAILBREAKBENCH_DIR",
  "xstest_file": "$XSTEST_FILE"
}
JSON

json_files=()
for candidate in "$OUT_DIR/harmbench.json" "$OUT_DIR/jailbreakbench.json" "$OUT_DIR/xstest.json" "$OUT_DIR/internal_proxy_with_external_judge.json" "$OUT_DIR/acceptance_run_metadata.json"; do
  if [[ -f "$candidate" ]]; then
    json_files+=("$candidate")
  fi
done

uv run python -m scripts.summarize_acceptance_results "${json_files[@]}" \
  --out "$OUT_DIR/acceptance_summary.csv"
