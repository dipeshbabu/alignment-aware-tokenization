from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval.eval_jailbreak import gen_batch, refused


TEXT_KEYS = ("text", "prompt", "instruction", "question", "user")
LABEL_KEYS = ("label", "category", "type", "safety_label")


def read_records(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if not line.strip():
                continue
            payload = json.loads(line)
            text = ""
            for key in TEXT_KEYS:
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    text = value.strip()
                    break
            if not text:
                continue
            label = ""
            for key in LABEL_KEYS:
                value = payload.get(key)
                if isinstance(value, str):
                    label = value.lower()
                    break
            rows.append({"id": line_idx, "text": text, "label": label, "raw": payload})
    return rows


def is_safe_record(row: dict[str, Any]) -> bool:
    label = str(row.get("label") or "").lower()
    if not label:
        return True
    unsafe_markers = ("unsafe", "contrast", "harmful", "dangerous", "disallowed")
    return not any(marker in label for marker in unsafe_markers)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute over-refusal on an external XSTest-style JSONL file."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", default="")
    parser.add_argument("--adapter", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--gen_bs", type=int, default=8)
    parser.add_argument("--refusal_window", type=int, default=200)
    parser.add_argument("--include_unsafe", action="store_true")
    args = parser.parse_args()

    rows = read_records(args.input)
    eval_rows = rows if args.include_unsafe else [row for row in rows if is_safe_record(row)]
    if not eval_rows:
        raise ValueError(f"No XSTest rows selected from {args.input}")

    tokenizer_name = args.tokenizer or args.model
    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    tok.padding_side = "right"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter).eval()
    try:
        model.generation_config.pad_token_id = tok.pad_token_id
    except Exception:
        pass

    prompts = [row["text"] for row in eval_rows]
    responses = gen_batch(
        model,
        tok,
        prompts,
        max_new_tokens=args.max_new_tokens,
        greedy=True,
        temperature=0.0,
        top_p=1.0,
        bs=args.gen_bs,
    )
    refused_flags = [refused(response, k=args.refusal_window) for response in responses]
    over_refusal = sum(refused_flags) / max(len(refused_flags), 1)

    out = {
        "benchmark": "xstest",
        "model_name": args.model,
        "tokenizer_name": tokenizer_name,
        "adapter": args.adapter,
        "input": args.input,
        "num_total_rows": len(rows),
        "num_eval_rows": len(eval_rows),
        "over_refusal_rate": over_refusal,
        "metrics": {
            "over_refusal_rate": over_refusal,
            "benign_refusal_rate": over_refusal,
        },
        "examples": [
            {
                "id": row["id"],
                "label": row["label"],
                "text": row["text"],
                "response": response,
                "refused": bool(flag),
            }
            for row, response, flag in zip(eval_rows, responses, refused_flags)
        ],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[done] wrote XSTest over-refusal results to {out_path}")


if __name__ == "__main__":
    main()
