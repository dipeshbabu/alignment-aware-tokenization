setup:
	uv sync

sanity:
	uv run bash scripts/run_sanity.sh

validate:
	uv run python -m scripts.validate_data data/anchors/anchors_500.jsonl data/neutrals/neutrals_1000.jsonl data/eval/attack_extra_500.jsonl data/eval/benign_1500.jsonl

stem_split:
	uv run python -m data_tools.make_stem_split --anchors data/anchors/anchors_500.jsonl --out_dir data/splits/stems_seed9172 --heldout_frac 0.2 --seed 9172

tokspill:
	uv run bash scripts/run_tokspill.sh

perturb_attacks:
	uv run bash scripts/run_perturb_attacks.sh

probe:
	uv run bash scripts/run_probe.sh

probes:
	uv run bash scripts/run_probes_pythia.sh

probes_spm:
	uv run bash scripts/run_probes_spm.sh

train:
	uv run bash scripts/run_lora_drift.sh

train_14b:
	uv run bash scripts/run_lora_drift_pythia1_4b.sh

train_spm:
	uv run bash scripts/run_lora_drift_spm.sh mistral7b

bpe_search:
	uv run bash scripts/run_bpe_search.sh

bpe_baselines:
	uv run bash scripts/run_bpe_baselines.sh

tokenizer_metrics:
	uv run bash scripts/run_tokenizer_metrics.sh

label_efficiency:
	uv run bash scripts/run_label_efficiency.sh

tables:
	uv run python -m scripts.make_paper_tables --tokenizer_json outputs/tokspill_*.json --out_prefix outputs/table_tokenizer

spm_priors:
	uv run bash scripts/run_spm_priors.sh mistral7b

eval:
	uv run bash scripts/run_eval_all.sh

judge_eval:
	uv run bash scripts/run_judge_eval.sh
