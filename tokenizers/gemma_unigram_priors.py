#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

from tokenizers import SentencePieceUnigramTokenizer
from transformers import PreTrainedTokenizerFast, AutoTokenizer


def read_jsonl_texts(path: str, key: str = "text", limit: Optional[int] = None) -> List[str]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                j = json.loads(line)
                t = (j.get(key) or "").strip()
            except Exception:
                t = line
            if t:
                out.append(t)
    return out


def read_list(path: str) -> List[str]:
    xs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t:
                xs.append(t)
    return xs


def build_piece_priors(
    hazard_terms: List[str],
    penalty_substrings: List[str],
    boost: float,
    penalty: float,
) -> Dict[str, float]:
    """
    Tokenizers library supports specifying special tokens, but not direct priors like sentencepiece.
    We emulate priors by augmenting training data with weighted repeats for boosts,
    and by adding negative pressure via a penalty list that we later inspect and report.

    This function returns a dictionary for transparency and logging only.
    """
    priors = {}
    for w in hazard_terms:
        priors[w] = float(boost)
    for s in penalty_substrings:
        priors[s] = float(-abs(penalty))
    return priors


def write_tokenizer_dir(
    out_dir: str,
    tokenizer_obj: SentencePieceUnigramTokenizer,
    base_hf_tokenizer,
    bos_token: str,
    eos_token: str,
    pad_token: str,
    unk_token: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    tokenizer_json = os.path.join(out_dir, "tokenizer.json")
    tokenizer_obj.save(tokenizer_json)

    fast = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_json,
        bos_token=bos_token,
        eos_token=eos_token,
        pad_token=pad_token,
        unk_token=unk_token,
    )

    fast.save_pretrained(out_dir)

    cfg = {
        "model_max_length": getattr(base_hf_tokenizer, "model_max_length", 2048),
        "padding_side": "right",
        "truncation_side": "right",
    }
    with open(os.path.join(out_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_tokenizer", default="google/gemma-2b")
    ap.add_argument("--train_jsonl", required=True, help="unlabeled training jsonl")
    ap.add_argument("--key", default="text")
    ap.add_argument("--limit", type=int, default=500000)
    ap.add_argument("--vocab_size", type=int, default=None,
                    help="defaults to base tokenizer vocab size")
    ap.add_argument("--hazard_terms_txt", required=True, help="one term per line, to be boosted")
    ap.add_argument("--penalty_substrings_txt", required=True,
                    help="one substring per line, logged for reporting")
    ap.add_argument("--boost_repeats", type=int, default=30,
                    help="how many times to repeat hazard lines in training")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    base = AutoTokenizer.from_pretrained(args.base_tokenizer, use_fast=True)
    vocab_size = int(args.vocab_size or getattr(base, "vocab_size", 32000))

    texts = read_jsonl_texts(args.train_jsonl, key=args.key, limit=args.limit)
    hazard_terms = read_list(args.hazard_terms_txt)
    penalty_substrings = read_list(args.penalty_substrings_txt)

    os.makedirs(args.out_dir, exist_ok=True)
    tmp_train = os.path.join(args.out_dir, "unigram_train.txt")

    with open(tmp_train, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(t.replace("\n", " ") + "\n")
        for w in hazard_terms:
            line = f" {w} "
            for _ in range(max(1, int(args.boost_repeats))):
                f.write(line + "\n")

    tokenizer = SentencePieceUnigramTokenizer()
    tokenizer.train(
        files=[tmp_train],
        vocab_size=vocab_size,
        special_tokens=list({base.unk_token or "<unk>", base.bos_token or "<bos>",
                            base.eos_token or "</s>", base.pad_token or "<pad>"}),
    )

    priors = build_piece_priors(hazard_terms, penalty_substrings,
                                boost=float(args.boost_repeats), penalty=1.0)
    with open(os.path.join(args.out_dir, "priors_log.json"), "w", encoding="utf-8") as f:
        json.dump(priors, f, indent=2)

    bos = base.bos_token or "<bos>"
    eos = base.eos_token or "</s>"
    pad = base.pad_token or (base.eos_token or "</s>")
    unk = base.unk_token or "<unk>"

    write_tokenizer_dir(
        out_dir=args.out_dir,
        tokenizer_obj=tokenizer,
        base_hf_tokenizer=base,
        bos_token=bos,
        eos_token=eos,
        pad_token=pad,
        unk_token=unk,
    )

    print(f"Wrote Gemma compatible unigram tokenizer to {args.out_dir}")


if __name__ == "__main__":
    main()
