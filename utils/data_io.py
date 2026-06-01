from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence


TEXT_KEYS: tuple[str, ...] = ("text", "prompt", "instruction", "question", "content")


def iter_jsonl_records(path: str | Path) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    row = {"text": str(row)}
            except Exception:
                row = {"text": line.strip()}
            row.setdefault("_line", line_idx)
            yield row


def extract_text(row: dict, keys: Sequence[str] = TEXT_KEYS) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for value in row.values():
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def read_jsonl_texts(
    path: str | Path,
    *,
    label: str | None = None,
    key: str | None = None,
    limit: int | None = None,
) -> list[str]:
    keys = (key,) if key else TEXT_KEYS
    out: list[str] = []
    for row in iter_jsonl_records(path):
        if label is not None and row.get("label") != label:
            continue
        text = extract_text(row, keys)
        if text:
            out.append(text)
            if limit is not None and len(out) >= limit:
                break
    return out


def label_counts(path: str | Path) -> Counter:
    counts: Counter = Counter()
    for row in iter_jsonl_records(path):
        counts[row.get("label", "")] += 1
    return counts


def read_hazard_anchor_texts(path: str | Path, *, limit: int | None = None) -> list[str]:
    counts = label_counts(path)
    if counts.get("hazard", 0) > 0:
        return read_jsonl_texts(path, label="hazard", limit=limit)
    return read_jsonl_texts(path, limit=limit)
