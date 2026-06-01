from __future__ import annotations

import re


STOP_STEMS: set[str] = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "can",
    "could",
    "create",
    "describe",
    "detail",
    "detailed",
    "does",
    "example",
    "examples",
    "explain",
    "for",
    "from",
    "give",
    "guide",
    "help",
    "how",
    "include",
    "including",
    "information",
    "instruction",
    "instructions",
    "into",
    "list",
    "make",
    "method",
    "provide",
    "request",
    "script",
    "show",
    "step",
    "steps",
    "that",
    "the",
    "their",
    "them",
    "this",
    "through",
    "use",
    "used",
    "user",
    "users",
    "using",
    "way",
    "what",
    "when",
    "where",
    "which",
    "with",
    "without",
    "write",
}


def extract_content_stems(text: str, min_len: int = 3, stop_stems: set[str] | None = None) -> set[str]:
    stop = STOP_STEMS if stop_stems is None else stop_stems
    stems: set[str] = set()
    for token in re.split(r"\s+|[^\w]", text.lower()):
        token = "".join(ch for ch in token if ch.isalnum())
        if len(token) >= min_len and token not in stop:
            stems.add(token)
    return stems
