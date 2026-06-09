#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a HF fast BPE tokenizer with tokenizers-model dropout enabled."
    )
    parser.add_argument("--base_tokenizer", required=True)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if not (0.0 < args.dropout < 1.0):
        raise ValueError("--dropout must be in (0, 1)")

    tmp = Path(tempfile.mkdtemp(prefix="aat_bpe_dropout_"))
    tok = AutoTokenizer.from_pretrained(args.base_tokenizer, use_fast=True)
    tok.save_pretrained(tmp)

    tokjson = tmp / "tokenizer.json"
    if not tokjson.exists():
        raise FileNotFoundError(f"tokenizer.json not found for {args.base_tokenizer}")

    payload = json.loads(tokjson.read_text(encoding="utf-8"))
    model = payload.get("model", {})
    if str(model.get("type", "")).lower() != "bpe":
        raise ValueError(f"Expected tokenizer.json model.type=BPE, got {model.get('type')!r}")
    model["dropout"] = float(args.dropout)
    tokjson.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for file in tmp.glob("*"):
        shutil.copy2(file, out_dir / file.name)

    # Verify reloadability.
    _ = AutoTokenizer.from_pretrained(str(out_dir), use_fast=True)
    (out_dir / "aat_bpe_dropout_config.json").write_text(
        json.dumps(
            {
                "base_tokenizer": args.base_tokenizer,
                "dropout": float(args.dropout),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[done] wrote BPE-dropout tokenizer to {out_dir}")


if __name__ == "__main__":
    main()
