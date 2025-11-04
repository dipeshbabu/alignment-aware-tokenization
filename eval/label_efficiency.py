#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Label-efficiency: train a tiny classifier on frozen mid-layer features for H vs N
with budgets {50,100,300}; report F1 and AUPRC.
"""
from __future__ import annotations
import argparse
import json
import numpy as np
import yaml
import torch
from typing import List
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, average_precision_score
from transformers import AutoTokenizer, AutoModel


def _read_texts(path: str) -> List[str]:
    xs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            t = (j.get("text") or "").strip()
            if t:
                xs.append(t)
    return xs


@torch.no_grad()
def pooled_hidden(model, tok, texts, layer):
    embs = []
    device = model.device
    # ensure tokenizer has a pad token
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    tok.padding_side = "right"

    for t in texts:
        enc = tok(t, return_tensors="pt", truncation=True, max_length=256).to(device)
        out = model(**enc, output_hidden_states=True)
        # [batch=1, seq, hid] -> take [0] then mean over seq
        h = out.hidden_states[layer][0].mean(0).to(torch.float32)  # <-- cast to fp32
        embs.append(h.cpu().numpy())
    hidden_size = getattr(model.config, "hidden_size", embs[0].shape[0] if embs else 0)
    return np.stack(embs) if embs else np.zeros((0, hidden_size), dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--budgets", nargs="+", type=int, default=[50, 100, 300])
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    tok = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=True)
    model = AutoModel.from_pretrained(
        cfg["model_name"],
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    ).to("cuda" if torch.cuda.is_available() else "cpu").eval()

    H = _read_texts(cfg["data"]["anchors"])
    N = _read_texts(cfg["data"]["neutrals"])
    layer = cfg["drift"]["layer"]

    XH = pooled_hidden(model, tok, H, layer)
    XN = pooled_hidden(model, tok, N, layer)
    yH, yN = np.ones(len(XH)), np.zeros(len(XN))
    X = np.vstack([XH, XN])
    y = np.concatenate([yH, yN])

    # simple split: last 20% as test
    n = len(X)
    idx = np.arange(n)
    np.random.default_rng(42).shuffle(idx)
    tr = idx[: int(0.8*n)]
    te = idx[int(0.8*n):]
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]

    for k in args.budgets:
        sub = np.random.default_rng(0).choice(len(Xtr), size=min(k, len(Xtr)), replace=False)
        clf = LogisticRegression(max_iter=500, class_weight="balanced",
                                 solver="lbfgs").fit(Xtr[sub], ytr[sub])
        prob = clf.predict_proba(Xte)[:, 1]
        pred = (prob >= 0.5).astype(int)
        f1 = f1_score(yte, pred)
        auprc = average_precision_score(yte, prob)
        print(f"labels={len(sub)} F1={f1:.4f} AUPRC={auprc:.4f}")


if __name__ == "__main__":
    main()
