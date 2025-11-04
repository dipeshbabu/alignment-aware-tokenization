#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert a SentencePiece .model (Unigram) to a Hugging Face tokenizer.json
with correct special tokens. Optionally inherit special tokens from a base HF
model (recommended: use the LM you plan to fine-tune, e.g., LLaMA/Mistral/Qwen).

Typical usage:
  python -m tools.tokenizer_export \
    --spm_model tokenizers/spm_hazard/spm.model \
    --out_dir tokenizers/spm_hazard \
    --base_model meta-llama/Meta-Llama-3-8B

Outputs:
  - tokenizers/spm_hazard/tokenizer.json
  - tokenizers/spm_hazard/tokenizer_config.json
  - (keeps spm.model / spm.vocab from prior step if present)
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional

from tokenizers import normalizers, pre_tokenizers, decoders, Tokenizer
from tokenizers.implementations import SentencePieceUnigramTokenizer
from transformers import PreTrainedTokenizerFast, AutoTokenizer


def _load_specials_from_base(base_model: str) -> dict:
    """Pull special token strings from a base HF tokenizer if available."""
    base = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    specials = {
        "bos_token": getattr(base, "bos_token", None),
        "eos_token": getattr(base, "eos_token", None),
        "unk_token": getattr(base, "unk_token", "<unk>"),
        "pad_token": getattr(base, "pad_token", None),
    }
    # Remove Nones (PreTrainedTokenizerFast handles missing values)
    return {k: v for k, v in specials.items() if v is not None}


def _default_specials() -> dict:
    """Sensible defaults for Unigram/SentencePiece-style tokenizers."""
    # Many SPM LMs use <unk>, and rely on BOS/EOS; pad is optional.
    return {"unk_token": "<unk>", "bos_token": "<s>", "eos_token": "</s>"}


def spm_to_hf(spm_model_path: str, out_dir: str, base_model: Optional[str] = None) -> str:
    """
    Load a SentencePiece Unigram model and export as a HF PreTrainedTokenizerFast.

    Args:
        spm_model_path: path to .model (trained with spm_priors or vanilla SPM)
        out_dir: directory to write tokenizer.json + tokenizer_config.json
        base_model: optional HF model to copy special tokens from

    Returns:
        out_dir (so callers can chain operations)
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1) Load as a tokenizers SentencePieceUnigramTokenizer
    sp_tok = SentencePieceUnigramTokenizer(spm_model_path)

    # 2) Ensure a standard normalizer & pre-tokenizer & decoder (SentencePiece defaults)
    #    (SentencePieceUnigramTokenizer sets these appropriately, but we make it explicit.)
    tok: Tokenizer = sp_tok._tokenizer
    tok.normalizer = normalizers.Sequence([normalizers.NFKC()])
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.Metaspace(replacement="▁", add_prefix_space=True)])
    tok.decoder = decoders.Metaspace(replacement="▁", add_prefix_space=True)

    # 3) Wrap with HF fast tokenizer
    specials = _load_specials_from_base(base_model) if base_model else _default_specials()
    hf_fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        **specials,
    )

    # 4) Save in HF format
    hf_fast.save_pretrained(out_dir)

    # Also write a tiny tokenizer_config supplement (optional, but handy for clarity)
    cfg_path = os.path.join(out_dir, "tokenizer_config.json")
    if os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path, "r", encoding="utf-8"))
    else:
        cfg = {}
    cfg.update({"model_max_length": 4096, "tokenizer_class": "PreTrainedTokenizerFast"})
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print(f"[export] Wrote HF tokenizer to: {out_dir}")
    print(f"         Specials: {specials}")
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spm_model", required=True, help="Path to SentencePiece .model")
    ap.add_argument("--out_dir", required=True, help="Directory to write tokenizer.json")
    ap.add_argument("--base_model", default=None, help="HF model to copy special tokens from")
    args = ap.parse_args()

    spm_to_hf(args.spm_model, args.out_dir, base_model=args.base_model)


if __name__ == "__main__":
    main()
