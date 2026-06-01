from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


def read_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def find_spans(text: str, stem: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in re.finditer(re.escape(stem), text, flags=re.IGNORECASE)]


def token_offsets(tok, text: str) -> tuple[list[str], list[tuple[int, int]]]:
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    offsets = [tuple(x) for x in enc["offset_mapping"]]
    tokens = tok.convert_ids_to_tokens(ids)
    return tokens, offsets


def overlapping_tokens(tokens: list[str], offsets: list[tuple[int, int]], span: tuple[int, int]) -> list[str]:
    start, end = span
    out: list[str] = []
    for token, (tok_start, tok_end) in zip(tokens, offsets):
        if tok_end <= start or tok_start >= end or tok_end <= tok_start:
            continue
        out.append(token)
    return out


def boundary_starts(offsets: list[tuple[int, int]]) -> set[int]:
    return {int(s) for s, e in offsets if e > s}


def perturb_span(text: str, span: tuple[int, int], op: str) -> tuple[str, int, int]:
    start, end = span
    if end - start < 2:
        return text, start, 0
    mid = start + max(1, (end - start) // 2)
    if op == "insert":
        return text[:mid] + "." + text[mid:], mid, 1
    if op == "delete":
        return text[:mid] + text[mid + 1 :], mid, -1
    if op == "substitute":
        return text[:mid] + "x" + text[mid + 1 :], mid, 0
    return text, start, 0


def align_boundaries(bounds: set[int], edit_pos: int, delta: int) -> set[int]:
    if delta == 0:
        return bounds
    out: set[int] = set()
    for bound in bounds:
        out.add(bound if bound < edit_pos else bound - delta)
    return out


def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TokSpill tokenizer metrics.")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tokspill", required=True)
    parser.add_argument("--split", choices=["all", "train", "heldout", "unsplit"], default="all")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max_rows", type=int, default=0)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    rows = read_jsonl(args.tokspill)
    if args.split != "all":
        rows = [row for row in rows if row.get("split") == args.split]
    if args.max_rows > 0:
        rows = rows[: args.max_rows]

    token_count = 0
    char_count = 0
    span_piece_counts: list[float] = []
    fragmented: list[float] = []
    stability_j: list[float] = []
    stability_flip: list[float] = []
    stability_changed: list[float] = []

    pieces_by_stem_kind: dict[tuple[str, str], list[set[str]]] = defaultdict(list)
    per_kind: dict[str, int] = defaultdict(int)

    for row in rows:
        text = row["text"]
        stem = row["stem"]
        kind = row.get("kind", "")
        per_kind[kind] += 1
        tokens, offsets = token_offsets(tok, text)
        token_count += len(tokens)
        char_count += len(text)

        spans = find_spans(text, stem)
        if not spans:
            continue

        for span in spans:
            stem_tokens = overlapping_tokens(tokens, offsets, span)
            if not stem_tokens:
                continue
            stem_piece_set = set(stem_tokens)
            pieces_by_stem_kind[(stem, kind)].append(stem_piece_set)
            span_piece_counts.append(float(len(stem_tokens)))
            fragmented.append(float(len(stem_tokens) > 1))

            base_bounds = boundary_starts(offsets)
            for op in ("insert", "delete", "substitute"):
                perturbed, edit_pos, delta = perturb_span(text, span, op)
                _, pert_offsets = token_offsets(tok, perturbed)
                pert_bounds = align_boundaries(boundary_starts(pert_offsets), edit_pos, delta)
                score = jaccard(base_bounds, pert_bounds)
                stability_j.append(score)
                stability_flip.append(1.0 - score)
                stability_changed.append(float(base_bounds != pert_bounds))

    overlap_scores: list[float] = []
    for stem in sorted({key[0] for key in pieces_by_stem_kind}):
        hazard_sets = (
            pieces_by_stem_kind.get((stem, "hazard_anchor"), [])
            + pieces_by_stem_kind.get((stem, "hazard_perturbed"), [])
        )
        benign_sets = pieces_by_stem_kind.get((stem, "benign_neighbor"), [])
        if not hazard_sets or not benign_sets:
            continue
        hazard_union = set().union(*hazard_sets)
        for benign in benign_sets:
            overlap_scores.append(jaccard(hazard_union, benign))

    summary = {
        "tokenizer": args.tokenizer,
        "tokspill": args.tokspill,
        "split": args.split,
        "num_rows": len(rows),
        "rows_by_kind": dict(sorted(per_kind.items())),
        "metrics": {
            "tokens_per_char": float(token_count / max(char_count, 1)) if char_count else math.nan,
            "stem_piece_count_mean": mean(span_piece_counts),
            "stem_fragmentation_rate": mean(fragmented),
            "benign_hazard_piece_overlap": mean(overlap_scores),
            "boundary_jaccard": mean(stability_j),
            "boundary_flip_rate": mean(stability_flip),
            "segmentation_changed_rate": mean(stability_changed),
        },
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
