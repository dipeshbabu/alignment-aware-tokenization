#!/usr/bin/env bash
set -euo pipefail

# One entry point for the paper experiment pipeline.
#
# Usage:
#   uv run bash scripts/sh/run_all.sh lightweight
#   uv run bash scripts/sh/run_all.sh internal
#   uv run bash scripts/sh/run_all.sh acceptance
#   uv run bash scripts/sh/run_all.sh full
#
# Modes:
#   lightweight  data/syntax checks and held-out split generation
#   internal     full internal GPU/model pipeline, no external benchmarks
#   acceptance   external benchmark/judge pipeline only
#   full         internal + acceptance

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MODE=${1:-lightweight}

case "$MODE" in
  lightweight)
    uv run bash scripts/sh/reproduce_main.sh
    ;;
  internal)
    AAT_HEAVY=1 uv run bash scripts/sh/reproduce_main.sh
    ;;
  acceptance)
    uv run bash scripts/sh/run_acceptance_evals.sh
    ;;
  full)
    AAT_HEAVY=1 uv run bash scripts/sh/reproduce_main.sh
    uv run bash scripts/sh/run_acceptance_evals.sh
    ;;
  *)
    echo "Usage: $0 [lightweight|internal|acceptance|full]" >&2
    exit 2
    ;;
esac
