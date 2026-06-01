from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from utils.data_io import extract_text, iter_jsonl_records


DEFAULT_VARIANTS = ("space_split", "punct_insert", "case_flip", "char_delete", "char_substitute")


def load_stems(path: str) -> list[str]:
    if not path:
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [str(x).lower() for x in json.load(f)]


def candidate_words(text: str, stems: list[str]) -> list[tuple[str, str]]:
    text_low = text.lower()
    hits = [(stem, stem) for stem in stems if stem in text_low and len(stem) >= 3]
    if hits:
        return hits
    words = []
    for word in re.findall(r"[A-Za-z0-9]{4,}", text):
        words.append((word.lower(), word))
    return words


def perturb_word(word: str, variant: str, rng: random.Random) -> str:
    if len(word) < 3:
        return word
    idx = rng.randrange(1, len(word) - 1)
    if variant == "space_split":
        return word[:idx] + " " + word[idx:]
    if variant == "punct_insert":
        return word[:idx] + rng.choice([".", "-", "_", "/"]) + word[idx:]
    if variant == "case_flip":
        return "".join(ch.upper() if i % 2 == 0 else ch.lower() for i, ch in enumerate(word))
    if variant == "char_delete":
        return word[:idx] + word[idx + 1 :]
    if variant == "char_substitute":
        return word[:idx] + rng.choice("abcdefghijklmnopqrstuvwxyz") + word[idx + 1 :]
    return word


def replace_once(text: str, needle: str, replacement: str) -> str:
    return re.sub(re.escape(needle), replacement, text, count=1, flags=re.IGNORECASE)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create perturbed attack JSONL for robustness eval.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stems", default="")
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    parser.add_argument("--per_prompt", type=int, default=3)
    parser.add_argument("--seed", type=int, default=9172)
    parser.add_argument("--include_original", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    stems = load_stems(args.stems)
    rows_out: list[dict] = []

    for idx, row in enumerate(iter_jsonl_records(args.input)):
        text = extract_text(row)
        if not text:
            continue
        if args.include_original:
            rows_out.append(
                {
                    "text": text,
                    "label": row.get("label", "attack"),
                    "source": row.get("source"),
                    "original_id": idx,
                    "perturbation": "original",
                }
            )

        candidates = candidate_words(text, stems)
        if not candidates:
            continue
        rng.shuffle(candidates)
        variants = list(args.variants)
        rng.shuffle(variants)
        made = 0
        for variant in variants:
            stem, surface = candidates[made % len(candidates)]
            replacement = perturb_word(surface, variant, rng)
            perturbed = replace_once(text, surface, replacement)
            if perturbed == text:
                continue
            rows_out.append(
                {
                    "text": perturbed,
                    "label": row.get("label", "attack"),
                    "source": row.get("source"),
                    "original_id": idx,
                    "target_stem": stem,
                    "perturbation": variant,
                }
            )
            made += 1
            if made >= args.per_prompt:
                break

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for row in rows_out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input": args.input,
        "output": str(output),
        "stems": args.stems,
        "variants": args.variants,
        "per_prompt": args.per_prompt,
        "seed": args.seed,
        "num_rows": len(rows_out),
    }
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
