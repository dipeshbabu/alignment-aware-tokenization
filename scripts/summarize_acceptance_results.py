from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


METRIC_ALIASES = {
    "setting_name": [
        "setting_name",
        "config.setting_name",
    ],
    "model_name": [
        "model_name",
        "config.model_name",
    ],
    "tokenizer_name": [
        "tokenizer_name",
        "config.tokenizer_name",
    ],
    "adapter": [
        "adapter",
        "config.adapter",
    ],
    "judge_model": [
        "judge_model",
        "config.judge_model",
    ],
    "attack_success_rate": [
        "attack_success_rate",
        "asr",
        "harmbench_asr",
        "jailbreak_success",
        "jailbreak_success_rate",
        "metrics.attack_success_rate",
        "metrics.asr",
        "metrics.jailbreak_success",
    ],
    "over_refusal_rate": [
        "over_refusal_rate",
        "benign_refusal",
        "benign_refusal_rate",
        "xstest_refusal_rate",
        "metrics.over_refusal_rate",
        "metrics.benign_refusal",
        "metrics.benign_refusal_rate",
    ],
    "helpfulness": [
        "helpfulness",
        "helpfulness_score",
        "metrics.helpfulness",
        "metrics.helpfulness_score",
    ],
    "ppl": [
        "ppl",
        "perplexity",
        "metrics.ppl",
        "metrics.perplexity",
    ],
}


def flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            flatten(f"{prefix}.{key}" if prefix else str(key), child, out)
    elif isinstance(value, (str, int, float, bool)) or value is None:
        out[prefix] = value


def first_metric(flat: dict[str, Any], names: list[str]) -> Any:
    for name in names:
        if name in flat:
            return flat[name]
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize external acceptance benchmark JSON files into one CSV."
    )
    parser.add_argument("json_files", nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = []
    for file_name in args.json_files:
        path = Path(file_name)
        payload = json.loads(path.read_text(encoding="utf-8"))
        flat: dict[str, Any] = {}
        flatten("", payload, flat)
        row = {
            "file": str(path),
            "benchmark": path.stem,
        }
        for canonical, aliases in METRIC_ALIASES.items():
            row[canonical] = first_metric(flat, aliases)
        rows.append(row)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "benchmark",
                "setting_name",
                "model_name",
                "tokenizer_name",
                "adapter",
                "judge_model",
                "attack_success_rate",
                "over_refusal_rate",
                "helpfulness",
                "ppl",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[done] wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
