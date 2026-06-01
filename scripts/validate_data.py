from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from utils.data_io import extract_text, iter_jsonl_records


def summarize(path: Path) -> dict:
    labels: Counter = Counter()
    sources: Counter = Counter()
    empty_text = 0
    total = 0
    seen: set[str] = set()
    duplicates = 0

    for row in iter_jsonl_records(path):
        total += 1
        labels[row.get("label", "")] += 1
        sources[row.get("source", "")] += 1
        text = extract_text(row)
        if not text:
            empty_text += 1
        elif text in seen:
            duplicates += 1
        else:
            seen.add(text)

    return {
        "path": str(path),
        "total": total,
        "labels": dict(labels),
        "top_sources": sources.most_common(10),
        "empty_text": empty_text,
        "duplicate_text": duplicates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate JSONL dataset counts and labels.")
    parser.add_argument("paths", nargs="+", help="JSONL files to inspect")
    parser.add_argument(
        "--expect",
        action="append",
        default=[],
        metavar="PATH:LABEL:COUNT",
        help="Required count, for example data/anchors/anchors_500.jsonl:hazard:500",
    )
    parser.add_argument("--out", default="", help="Optional JSON summary path")
    args = parser.parse_args()

    summaries = [summarize(Path(path)) for path in args.paths]
    by_path = {item["path"]: item for item in summaries}

    failures: list[str] = []
    for spec in args.expect:
        try:
            path_s, label, count_s = spec.rsplit(":", 2)
            expected = int(count_s)
        except ValueError:
            failures.append(f"Invalid --expect format: {spec}")
            continue

        key = str(Path(path_s))
        summary = by_path.get(key)
        if summary is None:
            failures.append(f"Expected file was not inspected: {path_s}")
            continue
        actual = int(summary["labels"].get(label, 0)) if label else int(summary["total"])
        if actual != expected:
            failures.append(f"{path_s} label={label!r}: expected {expected}, found {actual}")

    payload = {"datasets": summaries, "failures": failures}
    print(json.dumps(payload, indent=2))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
