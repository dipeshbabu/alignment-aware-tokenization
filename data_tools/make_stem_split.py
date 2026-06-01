from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from utils.data_io import extract_text, iter_jsonl_records
from utils.stems import extract_content_stems


def stems_for_text(text: str, min_len: int) -> set[str]:
    return extract_content_stems(text, min_len=min_len)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create train/held-out hazard stem splits and filtered anchor files."
    )
    parser.add_argument("--anchors", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--heldout_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=9172)
    parser.add_argument("--min_stem_len", type=int, default=3)
    args = parser.parse_args()

    rows = []
    all_stems: set[str] = set()
    for row in iter_jsonl_records(args.anchors):
        if row.get("label") not in (None, "", "hazard"):
            continue
        text = extract_text(row)
        if not text:
            continue
        row_stems = stems_for_text(text, args.min_stem_len)
        rows.append((row, row_stems))
        all_stems.update(row_stems)

    if not rows:
        raise SystemExit(f"No hazard anchors found in {args.anchors}")

    rng = random.Random(args.seed)
    stems = sorted(all_stems)
    rng.shuffle(stems)
    n_heldout = max(1, round(len(stems) * args.heldout_frac))
    heldout_stems = set(stems[:n_heldout])
    train_stems = set(stems[n_heldout:])

    train_rows = []
    heldout_rows = []
    mixed_rows = []
    for row, row_stems in rows:
        if row_stems and row_stems <= heldout_stems:
            heldout_rows.append(row)
        elif row_stems and row_stems.isdisjoint(heldout_stems):
            train_rows.append(row)
        else:
            mixed_rows.append((row, row_stems))

    if not heldout_rows and mixed_rows:
        # Anchor descriptions often contain several stems. Preserve a held-out
        # evaluation set by assigning rows with the largest held-out-stem share
        # to heldout and excluding them from training.
        target = max(1, round(len(rows) * args.heldout_frac))
        scored = []
        for row, row_stems in mixed_rows:
            h = len(row_stems & heldout_stems)
            ratio = h / max(1, len(row_stems))
            scored.append((ratio, h, row, row_stems))
        scored.sort(key=lambda item: (-item[0], -item[1]))
        heldout_rows = [row for _, _, row, _ in scored[:target]]
        heldout_ids = {id(row) for row in heldout_rows}
        mixed_rows = [(row, row_stems) for _, _, row, row_stems in scored[target:] if id(row) not in heldout_ids]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def write_jsonl(path: Path, items: list[dict]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    write_jsonl(out_dir / "anchors_train_stems.jsonl", train_rows)
    write_jsonl(out_dir / "anchors_heldout_stems.jsonl", heldout_rows)
    write_jsonl(out_dir / "anchors_mixed_stems.jsonl", [row for row, _ in mixed_rows])
    (out_dir / "train_stems.json").write_text(
        json.dumps(sorted(train_stems), indent=2), encoding="utf-8"
    )
    (out_dir / "heldout_stems.json").write_text(
        json.dumps(sorted(heldout_stems), indent=2), encoding="utf-8"
    )
    (out_dir / "split_summary.json").write_text(
        json.dumps(
            {
                "anchors": args.anchors,
                "seed": args.seed,
                "min_stem_len": args.min_stem_len,
                "heldout_frac": args.heldout_frac,
                "num_stems": len(stems),
                "num_train_stems": len(train_stems),
                "num_heldout_stems": len(heldout_stems),
                "num_train_rows": len(train_rows),
                "num_heldout_rows": len(heldout_rows),
                "num_mixed_rows_excluded": len(mixed_rows),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Wrote stem split to {out_dir}")


if __name__ == "__main__":
    main()
