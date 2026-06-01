from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from utils.data_io import extract_text, iter_jsonl_records
from utils.stems import extract_content_stems


def word_stems(text: str, min_len: int) -> list[str]:
    return sorted(extract_content_stems(text, min_len=min_len))


def load_stem_json(path: str) -> set[str]:
    if not path:
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return {str(x).lower() for x in json.load(f)}


def perturb_word(word: str, op: str, rng: random.Random) -> str:
    if len(word) < 3:
        return word
    idx = rng.randrange(1, len(word) - 1)
    if op == "space_split":
        return word[:idx] + " " + word[idx:]
    if op == "punct_insert":
        return word[:idx] + rng.choice([".", "-", "_", "/"]) + word[idx:]
    if op == "case_flip":
        return "".join(ch.upper() if i % 2 == 0 else ch.lower() for i, ch in enumerate(word))
    if op == "char_delete":
        return word[:idx] + word[idx + 1 :]
    if op == "char_substitute":
        return word[:idx] + rng.choice("abcdefghijklmnopqrstuvwxyz") + word[idx + 1 :]
    return word


def replace_first_stem(text: str, stem: str, replacement: str) -> str:
    return re.sub(re.escape(stem), replacement, text, count=1, flags=re.IGNORECASE)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build TokSpill benchmark JSONL.")
    parser.add_argument("--anchors", required=True)
    parser.add_argument("--neutrals", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--train_stems", default="")
    parser.add_argument("--heldout_stems", default="")
    parser.add_argument("--min_stem_len", type=int, default=3)
    parser.add_argument("--max_neighbors_per_stem", type=int, default=20)
    parser.add_argument("--max_anchor_rows", type=int, default=0)
    parser.add_argument("--perturbations", nargs="+", default=["space_split", "punct_insert"])
    parser.add_argument("--seed", type=int, default=9172)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    train_stems = load_stem_json(args.train_stems)
    heldout_stems = load_stem_json(args.heldout_stems)

    hazard_by_stem: dict[str, list[dict]] = defaultdict(list)
    for row in iter_jsonl_records(args.anchors):
        if row.get("label") not in (None, "", "hazard"):
            continue
        text = extract_text(row)
        if not text:
            continue
        for stem in sorted(set(word_stems(text, args.min_stem_len))):
            hazard_by_stem[stem].append(row)

    if args.max_anchor_rows > 0:
        for stem in list(hazard_by_stem):
            hazard_by_stem[stem] = hazard_by_stem[stem][: args.max_anchor_rows]

    neutral_by_stem: dict[str, list[dict]] = defaultdict(list)
    candidate_stems = set(hazard_by_stem)
    for row in iter_jsonl_records(args.neutrals):
        text = extract_text(row)
        if not text:
            continue
        text_low = text.lower()
        for stem in candidate_stems:
            if stem in text_low and len(neutral_by_stem[stem]) < args.max_neighbors_per_stem:
                neutral_by_stem[stem].append(row)

    def split_for(stem: str) -> str:
        if heldout_stems and stem in heldout_stems:
            return "heldout"
        if train_stems and stem in train_stems:
            return "train"
        return "unsplit"

    rows: list[dict] = []
    for stem in sorted(candidate_stems):
        split = split_for(stem)
        for i, row in enumerate(hazard_by_stem.get(stem, [])):
            text = extract_text(row)
            item = {
                "id": f"{stem}:hazard:{i}",
                "stem": stem,
                "split": split,
                "kind": "hazard_anchor",
                "text": text,
                "source": row.get("source"),
                "original_line": row.get("_line"),
            }
            rows.append(item)
            for op in args.perturbations:
                pert_word = perturb_word(stem, op, rng)
                rows.append(
                    {
                        **item,
                        "id": f"{stem}:hazard:{i}:{op}",
                        "kind": "hazard_perturbed",
                        "perturbation": op,
                        "text": replace_first_stem(text, stem, pert_word),
                    }
                )

        neighbors = neutral_by_stem.get(stem, [])
        for i, row in enumerate(neighbors):
            rows.append(
                {
                    "id": f"{stem}:benign:{i}",
                    "stem": stem,
                    "split": split,
                    "kind": "benign_neighbor",
                    "text": extract_text(row),
                    "source": row.get("source"),
                    "original_line": row.get("_line"),
                }
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "anchors": args.anchors,
        "neutrals": args.neutrals,
        "out": str(out),
        "seed": args.seed,
        "num_rows": len(rows),
        "num_stems": len(candidate_stems),
        "num_train_stems": sum(1 for s in candidate_stems if split_for(s) == "train"),
        "num_heldout_stems": sum(1 for s in candidate_stems if split_for(s) == "heldout"),
        "num_stems_with_neighbors": sum(1 for s in candidate_stems if neutral_by_stem.get(s)),
        "kinds": {k: sum(1 for row in rows if row["kind"] == k) for k in sorted({r["kind"] for r in rows})},
    }
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
