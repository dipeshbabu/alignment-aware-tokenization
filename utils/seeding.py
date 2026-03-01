# utils/seeding.py
"""
Unified seeding and run metadata logging.
"""
from __future__ import annotations

import os
import time
import json
import hashlib
import random
from typing import Any

import numpy as np
import torch


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

    # Optional stricter reproducibility (can reduce speed)
    try:
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # torch.use_deterministic_algorithms may raise on some ops
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
        else:
            # safer default for speed
            torch.backends.cudnn.benchmark = True
    except Exception:
        pass


def config_hash(cfg: dict[str, Any]) -> str:
    s = json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode("utf-8")
    # md5 is okay for IDs, but sha256 is safer and just as easy
    return hashlib.sha256(s).hexdigest()[:8]


def log_run_meta(out_dir: str, cfg: dict, extras: dict | None = None) -> None:
    os.makedirs(out_dir, exist_ok=True)
    meta = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": cfg.get("seed", None),
        "config_hash": config_hash(cfg),
        "config": cfg,
    }
    if extras:
        meta.update(extras)
    with open(os.path.join(out_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def tokenizer_hash(tokenizer_dir_or_id: str) -> str:
    """Stable short hash for a tokenizer reference string."""
    return hashlib.sha256(tokenizer_dir_or_id.encode("utf-8")).hexdigest()[:8]
