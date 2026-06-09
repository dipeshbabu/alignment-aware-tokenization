# Config Manifest

Use the base configs for probe training and baseline model evaluation. All model
configs set `drift.rank: 4`, so `models.probe` writes a supervised hazard
subspace by default; set `--rank 1` only for the vector-probe ablation.

- `pythia410m.yml`
- `pythia1_4b.yml`
- `mistral7b.yml`
- `llama3_8b.yml`
- `qwen2_7b.yml`

Use the native-BPE configs for the final family-matched AAT rows after running
`scripts/sh/run_native_bpe_search.sh`:

- `pythia410m_native_bpe.yml`
- `pythia1_4b_native_bpe.yml`
- `llama3_8b_native_bpe.yml`
- `qwen2_7b_native_bpe.yml`

Use the SPM configs only for SentencePiece-family runs and retokenization stress
tests:

- `mistral7b_spm.yml` is the family-native SentencePiece setting.
- `llama3_8b_spm.yml` and `qwen2_7b_spm.yml` are tokenizer-family mismatch
  stress tests, not final evidence for LLaMA/Qwen.

`llm_eval.yml` stores shared internal jailbreak/benign-refusal proxy inputs. The
acceptance-target external benchmark runner is `scripts/sh/run_acceptance_evals.sh`.
