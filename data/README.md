Generated experiment data is intentionally not tracked in git.

Rebuild the local data needed by the experiments with:

```bash
uv run bash scripts/sh/curate_data.sh
```

For a small smoke-test dataset:

```bash
QUICK=1 uv run bash scripts/sh/curate_data.sh
```

See `data/MANIFEST.md` for the expected generated files and provenance policy.
