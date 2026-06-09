# Data Manifest

The repository does not track generated or curated experiment data. Rebuild the
local data with:

```bash
uv run bash scripts/sh/curate_data.sh
```

For a small smoke-test dataset:

```bash
QUICK=1 uv run bash scripts/sh/curate_data.sh
```

The curation script writes:

- `data/anchors/anchors_500.jsonl`
- `data/neutrals/neutrals_1000.jsonl`
- `data/unlabeled/u_train.jsonl`
- `data/unlabeled/u_dev.jsonl`
- `data/eval/attack_extra_500.jsonl`
- `data/eval/benign_1500.jsonl`
- `data/eval/attack_perturbed_seed9172.jsonl`
- `data/splits/stems_seed9172/*`
- `data/tokspill/tokspill_seed9172.jsonl`

Generated files are ignored because they can be large, safety-sensitive, and
subject to upstream dataset redistribution constraints. The scripts preserve
provenance in each JSONL row through the `source` field where available.
