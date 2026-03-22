#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from models.embed_remap import EmbeddingRemapper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True, help="Original HF model id or local path")
    ap.add_argument("--old_tokenizer", required=True, help="Original tokenizer id/path")
    ap.add_argument("--new_tokenizer", required=True, help="New tokenizer id/path")
    ap.add_argument("--out_dir", required=True, help="Directory to save remapped model + tokenizer")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--average_pool", action="store_true", help="Use average old-subpiece embeddings for unseen pieces")
    ap.add_argument("--no_update_lm_head", action="store_true", help="Do not copy into output embeddings / lm_head")
    args = ap.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    old_tok = AutoTokenizer.from_pretrained(args.old_tokenizer, use_fast=True)
    new_tok = AutoTokenizer.from_pretrained(args.new_tokenizer, use_fast=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch_dtype,
    ).to(device).eval()

    remapper = EmbeddingRemapper()
    model = remapper.remap(
        model=model,
        old_tok=old_tok,
        new_tok=new_tok,
        average_pool=args.average_pool,
        update_lm_head=not args.no_update_lm_head,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    model.save_pretrained(args.out_dir)
    new_tok.save_pretrained(args.out_dir)

    print(f"Saved remapped model and tokenizer to: {args.out_dir}")


if __name__ == "__main__":
    main()