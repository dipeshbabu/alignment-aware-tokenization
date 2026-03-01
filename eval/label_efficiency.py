#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Label-efficiency: train a tiny classifier on frozen mid-layer features for H vs N
with budgets {50,100,300}; report F1 and AUPRC as mean±std across repeated trials.
"""
from __future__ import annotations

import argparse
import json
from typing import List, Tuple

import numpy as np
import torch
import yaml
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, average_precision_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel


def _read_texts(path: str) -> List[str]:
    xs: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            t = (j.get("text") or "").strip()
            if t:
                xs.append(t)
    return xs


def _dedup_keep_order(xs: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _overlap_count(a: List[str], b: List[str]) -> int:
    return len(set(a).intersection(set(b)))


@torch.no_grad()
def pooled_hidden(
    model,
    tok,
    texts: List[str],
    layer: int,
    max_len: int = 256,
    batch_size: int = 16,
) -> np.ndarray:
    """
    Mask-aware mean-pool hidden states at a given layer across sequence positions.
    Returns float32 numpy array: [n_texts, hidden_dim].
    """
    device = model.device

    # ensure tokenizer has a pad token
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    tok.padding_side = "right"

    embs: List[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        enc = tok(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
            add_special_tokens=False,
        ).to(device)

        out = model(**enc, output_hidden_states=True)
        h = out.hidden_states[layer]  # [B, T, D]

        mask = enc["attention_mask"].unsqueeze(-1).to(h.dtype)  # [B, T, 1]
        masked = h * mask
        denom = mask.sum(dim=1).clamp_min(1.0)  # [B, 1]
        pooled = (masked.sum(dim=1) / denom).to(torch.float32)  # [B, D]

        embs.append(pooled.cpu().numpy())

    hidden_size = getattr(model.config, "hidden_size", embs[0].shape[1] if embs else 0)
    return np.concatenate(embs, axis=0) if embs else np.zeros((0, hidden_size), dtype=np.float32)


def _fit_eval_once(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """
    Train on a random subset of size k from training set, evaluate on test.
    Returns (F1, AUPRC).
    """
    k_eff = min(k, len(Xtr))
    sub_idx = rng.choice(len(Xtr), size=k_eff, replace=False)

    if len(np.unique(ytr[sub_idx])) < 2:
        return float("nan"), float("nan")

    clf = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
        )
    ).fit(Xtr[sub_idx], ytr[sub_idx])

    prob = clf.predict_proba(Xte)[:, 1]
    pred = (prob >= 0.5).astype(int)
    f1 = f1_score(yte, pred)
    auprc = average_precision_score(yte, prob)
    return float(f1), float(auprc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--budgets", nargs="+", type=int, default=[50, 100, 300])
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--trials", type=int, default=10, help="Subsampling trials per budget")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--dedup", action="store_true")
    ap.add_argument("--check_overlap", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    model_name = cfg["model_name"]
    layer = int(cfg["drift"]["layer"])

    H = _read_texts(cfg["data"]["anchors"])
    N = _read_texts(cfg["data"]["neutrals"])

    if args.dedup:
        H = _dedup_keep_order(H)
        N = _dedup_keep_order(N)

    print(f"[data] |H|={len(H)}  |N|={len(N)}  layer={layer}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device).eval()

    XH = pooled_hidden(model, tok, H, layer=layer, max_len=args.max_len, batch_size=args.batch_size)
    XN = pooled_hidden(model, tok, N, layer=layer, max_len=args.max_len, batch_size=args.batch_size)

    yH = np.ones(len(XH), dtype=np.int64)
    yN = np.zeros(len(XN), dtype=np.int64)
    X = np.vstack([XH, XN])
    y = np.concatenate([yH, yN])

    texts = H + N

    idx = np.arange(len(X))
    tr_idx, te_idx = train_test_split(
        idx,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
        shuffle=True,
    )

    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xte, yte = X[te_idx], y[te_idx]

    if args.check_overlap:
        tr_texts = [texts[i] for i in tr_idx]
        te_texts = [texts[i] for i in te_idx]
        ov = _overlap_count(tr_texts, te_texts)
        print(f"[leakage] exact train/test text overlap: {ov} items")

    print(
        f"[split] train={len(Xtr)} test={len(Xte)}  pos_train={ytr.mean():.3f} pos_test={yte.mean():.3f}")

    rng = np.random.default_rng(args.seed)

    for k in args.budgets:
        f1s, auprcs = [], []
        for _ in range(args.trials):
            f1, auprc = _fit_eval_once(Xtr, ytr, Xte, yte, k=k, rng=rng)
            f1s.append(f1)
            auprcs.append(auprc)

        f1_mean, f1_std = float(np.mean(f1s)), float(np.std(f1s))
        ap_mean, ap_std = float(np.mean(auprcs)), float(np.std(auprcs))

        print(
            f"labels={min(k, len(Xtr)):<4d} "
            f"F1={f1_mean:.4f}±{f1_std:.4f}  "
            f"AUPRC={ap_mean:.4f}±{ap_std:.4f}  "
            f"(trials={args.trials})"
        )


if __name__ == "__main__":
    main()
