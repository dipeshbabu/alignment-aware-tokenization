from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge per-setting acceptance summary CSVs.")
    parser.add_argument("csv_files", nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    for file_name in args.csv_files:
        path = Path(file_name)
        setting = path.parent.name
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                for name in ["setting", *reader.fieldnames]:
                    if name not in fieldnames:
                        fieldnames.append(name)
            for row in reader:
                row = {"setting": setting, **row}
                rows.append(row)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[done] wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
