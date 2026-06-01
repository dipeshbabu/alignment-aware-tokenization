from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def flatten(prefix: str, value, out: dict) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            flatten(f"{prefix}.{key}" if prefix else str(key), child, out)
    elif isinstance(value, (str, int, float, bool)) or value is None:
        out[prefix] = value


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect run JSON summaries into CSV.")
    parser.add_argument("json_files", nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows: list[dict] = []
    keys: set[str] = set()
    for file_name in args.json_files:
        path = Path(file_name)
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        row: dict = {"file": str(path)}
        flatten("", payload, row)
        rows.append(row)
        keys.update(row.keys())

    fieldnames = ["file"] + sorted(k for k in keys if k != "file")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
