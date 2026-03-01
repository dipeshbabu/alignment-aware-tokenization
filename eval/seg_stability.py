# eval/seg_stability.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Segmentation Stability Evaluation (1-edit protocol)

IMPORTANT FIX:
- For insert/delete edits, character indices shift.
  We must align perturbed boundary indices back into the original string's index space,
  otherwise Jaccard/flip is artificially inflated.

This version:
- OneEditPerturber yields (op, perturbed, edit_pos, delta)
- Evaluator aligns boundaries before comparison.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from utils.seeding import set_global_seed


def read_jsonl_texts(path: str, key: str = "text", limit: Optional[int] = None) -> List[str]:
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            try:
                j = json.loads(line)
            except Exception:
                continue
            t = (j.get(key) or "").strip()
            if t:
                out.append(t)
    return out


def bootstrap_ci(values: np.ndarray, n_boot: int = 800, alpha: float = 0.05, seed: int = 9172) -> Tuple[float, float]:
    if len(values) == 0:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    means = []
    n = len(values)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means.append(values[idx].mean())
    lo, hi = np.percentile(means, [100 * (alpha / 2), 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


@dataclass
class TokenBoundaryExtractor:
    tokenizer: PreTrainedTokenizerBase

    def _fast_offsets(self, text: str) -> Optional[List[Tuple[int, int]]]:
        try:
            enc = self.tokenizer(
                text,
                return_offsets_mapping=True,
                add_special_tokens=False,
                truncation=False,
            )
            offsets = enc.get("offset_mapping")
            if isinstance(offsets, list) and len(offsets) > 0 and isinstance(offsets[0], tuple):
                return offsets
            if isinstance(offsets, list) and len(offsets) > 0 and isinstance(offsets[0], list):
                return [tuple(x) for x in offsets[0]]
        except Exception:
            return None
        return None

    def _heuristic_boundaries(self, text: str) -> Tuple[int, ...]:
        bounds: List[int] = []
        prev_split = True
        for idx, ch in enumerate(text):
            split_here = bool(re.match(r"\s|[^\w]", ch))
            if prev_split and not split_here:
                bounds.append(idx)
            prev_split = split_here
        return tuple(sorted(set(bounds)))

    def boundaries(self, text: str) -> Tuple[int, ...]:
        offsets = self._fast_offsets(text)
        if offsets:
            starts = [s for (s, e) in offsets if (e - s) > 0]
            return tuple(sorted(set(starts)))
        return self._heuristic_boundaries(text)

    def tokens_per_char(self, text: str) -> float:
        if not text:
            return 0.0
        offsets = self._fast_offsets(text)
        if offsets:
            n_tokens = sum(1 for (s, e) in offsets if (e - s) > 0)
        else:
            n_tokens = len([w for w in re.split(r"\s+|[^\w]", text) if w])
        return float(n_tokens) / max(len(text), 1)


def align_boundaries(b1: Set[int], edit_pos: int, delta: int) -> Set[int]:
    """
    Map boundaries from perturbed string back to original index space.
    - insert: delta=+1 => perturbed indices after edit_pos are +1, so subtract 1
    - delete: delta=-1 => perturbed indices after edit_pos are -1, so subtract(-1)=+1
    - swap:   delta=0  => unchanged
    """
    if delta == 0:
        return b1
    out = set()
    for b in b1:
        if b < edit_pos:
            out.add(b)
        else:
            out.add(b - delta)
    return out


@dataclass
class OneEditPerturber:
    ops: Sequence[str] = ("insert", "delete", "swap")
    samples_per_op: int = 3
    seed: int = 9172

    _INSERT_CHARS = list("!?.,:;–—-’'\"()[]{}")

    def __post_init__(self):
        self._rng = random.Random(self.seed)

    def _insert(self, s: str) -> Optional[Tuple[str, int, int]]:
        if not s:
            return None
        pos = self._rng.randrange(0, len(s) + 1)
        ch = self._rng.choice(self._INSERT_CHARS)
        return (s[:pos] + ch + s[pos:], pos, +1)

    def _delete(self, s: str) -> Optional[Tuple[str, int, int]]:
        if len(s) < 2:
            return None
        pos = self._rng.randrange(0, len(s))
        return (s[:pos] + s[pos + 1:], pos, -1)

    def _swap(self, s: str) -> Optional[Tuple[str, int, int]]:
        if len(s) < 2:
            return None
        trials = 0
        while trials < 10:
            pos = self._rng.randrange(0, len(s) - 1)
            if not (s[pos].isspace() or s[pos + 1].isspace()):
                t = s[:pos] + s[pos + 1] + s[pos] + s[pos + 2:]
                return (t, pos, 0)
            trials += 1
        pos = self._rng.randrange(0, len(s) - 1)
        t = s[:pos] + s[pos + 1] + s[pos] + s[pos + 2:]
        return (t, pos, 0)

    def variants(self, s: str) -> Iterable[Tuple[str, str, int, int]]:
        for op in self.ops:
            for _ in range(self.samples_per_op):
                if op == "insert":
                    r = self._insert(s)
                elif op == "delete":
                    r = self._delete(s)
                elif op == "swap":
                    r = self._swap(s)
                else:
                    r = None
                if r is not None:
                    t, pos, delta = r
                    if t != s:
                        yield (op, t, pos, delta)


@dataclass
class StabilityStats:
    jaccards: List[float]
    flips: List[float]
    seg_changed: List[int]
    tpc_original: List[float]


class StabilityEvaluator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, perturber: OneEditPerturber):
        self.tok = tokenizer
        self.boundary_extractor = TokenBoundaryExtractor(tokenizer)
        self.perturber = perturber

    @staticmethod
    def _jaccard(a: Set[int], b: Set[int]) -> float:
        if not a and not b:
            return 1.0
        inter = len(a & b)
        union = len(a | b)
        return float(inter) / max(union, 1)

    def evaluate(
        self,
        texts: Sequence[str],
        bootstrap: int = 0,
        alpha: float = 0.05,
        seed: int = 1234,
    ) -> Tuple[StabilityStats, Optional[dict]]:
        j_list: List[float] = []
        f_list: List[float] = []
        c_list: List[int] = []
        tpc_list: List[float] = []

        for s in texts:
            b0 = set(self.boundary_extractor.boundaries(s))
            tpc0 = self.boundary_extractor.tokens_per_char(s)
            tpc_list.append(tpc0)

            for _, s2, edit_pos, delta in self.perturber.variants(s):
                b1_raw = set(self.boundary_extractor.boundaries(s2))
                b1 = align_boundaries(b1_raw, edit_pos=edit_pos, delta=delta)

                j = self._jaccard(b0, b1)
                j_list.append(j)
                f_list.append(1.0 - j)
                c_list.append(int(b0 != b1))

        stats = StabilityStats(jaccards=j_list, flips=f_list,
                               seg_changed=c_list, tpc_original=tpc_list)

        cis = None
        if bootstrap > 0:
            j_lo, j_hi = bootstrap_ci(np.array(j_list, dtype=np.float32),
                                      n_boot=bootstrap, alpha=alpha, seed=seed)
            f_lo, f_hi = bootstrap_ci(np.array(f_list, dtype=np.float32),
                                      n_boot=bootstrap, alpha=alpha, seed=seed)
            c_lo, c_hi = bootstrap_ci(np.array(c_list, dtype=np.float32),
                                      n_boot=bootstrap, alpha=alpha, seed=seed)
            t_lo, t_hi = bootstrap_ci(np.array(tpc_list, dtype=np.float32),
                                      n_boot=bootstrap, alpha=alpha, seed=seed)
            cis = {
                "jaccard": (j_lo, j_hi),
                "flip_rate": (f_lo, f_hi),
                "seg_change_rate": (c_lo, c_hi),
                "tpc_orig": (t_lo, t_hi),
            }
        return stats, cis


def main():
    ap = argparse.ArgumentParser(description="Segmentation stability (1-edit protocol).")
    ap.add_argument("--tokenizer", required=True, help="HF tokenizer path or model id")
    ap.add_argument("--texts", type=str, default="", help="Optional JSONL file with texts")
    ap.add_argument("--key", type=str, default="text", help="JSONL key to read")
    ap.add_argument("--max_texts", type=int, default=400)
    ap.add_argument("--ops", nargs="+", default=["insert", "delete", "swap"])
    ap.add_argument("--samples_per_op", type=int, default=3)
    ap.add_argument("--bootstrap", type=int, default=800)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=9172)
    args = ap.parse_args()

    set_global_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    if args.texts and os.path.exists(args.texts):
        base_texts = read_jsonl_texts(args.texts, key=args.key, limit=args.max_texts)
    else:
        base_texts = [
            "We booked a photo shoot next weekend.",
            "The keynote had a bombastic tone throughout.",
            "He made bullet points for the meeting.",
            "They celebrated the team's explosive growth.",
            "She posted a photobomb from the party.",
        ] * max(1, args.max_texts // 5)
        base_texts = base_texts[: args.max_texts]

    perturber = OneEditPerturber(
        ops=tuple(args.ops), samples_per_op=args.samples_per_op, seed=args.seed)
    evaluator = StabilityEvaluator(tokenizer=tok, perturber=perturber)

    stats, cis = evaluator.evaluate(
        base_texts, bootstrap=args.bootstrap, alpha=args.alpha, seed=args.seed)

    j_mean = float(np.mean(stats.jaccards)) if stats.jaccards else float("nan")
    f_mean = float(np.mean(stats.flips)) if stats.flips else float("nan")
    c_mean = float(np.mean(stats.seg_changed)) if stats.seg_changed else float("nan")
    tpc_mean = float(np.mean(stats.tpc_original)) if stats.tpc_original else float("nan")

    print("=== Segmentation Stability (1-edit) ===")
    print(f"Texts: {len(base_texts)} | Ops: {','.join(args.ops)} | Samples/op: {args.samples_per_op}")
    print(f"Original TPC (mean):          {tpc_mean:6.4f}" + (
        f"   (95% CI {cis['tpc_orig'][0]:.4f}–{cis['tpc_orig'][1]:.4f})" if cis else ""))
    print(f"Jaccard (mean):               {j_mean:6.4f}" + (
        f"   (95% CI {cis['jaccard'][0]:.4f}–{cis['jaccard'][1]:.4f})" if cis else ""))
    print(f"Boundary flip rate (mean):    {f_mean:6.4f}" + (
        f"   (95% CI {cis['flip_rate'][0]:.4f}–{cis['flip_rate'][1]:.4f})" if cis else ""))
    print(f"Segmentation changed (rate):  {c_mean:6.4f}" + (
        f"   (95% CI {cis['seg_change_rate'][0]:.4f}–{cis['seg_change_rate'][1]:.4f})" if cis else ""))
    print("Notes: Boundary = character index where a token starts; specials ignored; insert/delete boundaries aligned.")
    print("=======================================")


if __name__ == "__main__":
    main()
