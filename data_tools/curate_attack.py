# data_tools/curate_attack.py
"""
CURATE_ATTACK — IMPLEMENTATION SPEC (READ FIRST)

Goal
-----
Build a small CLI that harvests “attack” prompts from one or more Hugging Face
datasets (e.g., AdvBench, JailbreakV-28K) and writes them to JSONL for eval.

Deliverable
-----------
A script that can:
  1) Load a dataset (with/without config; possibly gated) and discover a valid split
     when the user hasn’t provided one.
  2) Pick an appropriate text field from each example (auto-detect if needed).
  3) Shuffle (when supported), cap the number of examples, and write JSONL lines:
       {"text": <str>, "label": "attack", "source": "<dataset[:config]/split>"}
  4) Allow running either/both sources via a --run flag: {both|adv|jbv}.
  5) Handle gated datasets gracefully (AdvBench), warning and continuing.

CLI (must support exactly)
--------------------------
--hf-token            Optional token for gated datasets (e.g., AdvBench).
--trust-remote-code   Pass through to datasets.load_dataset (some repos require).
--n-adv               Max examples from AdvBench (default: 1000).
--n-jbv               Max examples from JailbreakV-28K (default: 500).
--out-adv             Output file for AdvBench (default: data/eval/attack_1000.jsonl).
--out-jbv             Output file for JailbreakV-28K (default: data/eval/attack_extra_500.jsonl).
--run                 Which source(s) to run: 'both' (default), 'adv', or 'jbv'.
--adv-ds              Dataset ID for AdvBench (default: walledai/AdvBench).
--adv-config          Optional config for AdvBench (default: None).
--adv-split           Optional split for AdvBench (default: None → discover).
--jbv-ds              Dataset ID for JBV (default: JailbreakV-28K/JailBreakV-28k).
--jbv-config          Config for JBV (default: JailBreakV_28K)  # REQUIRED by that repo
--jbv-split           Optional split for JBV (default: None → discover).

Behavior & Rules
----------------
1) Split discovery:
   - If split is not provided, call get_dataset_split_names(ds, config_name=config, token=token)
     and choose in priority order: "train" > "training" > "default" > "all" > first available.
   - If the call fails, default to "train".

2) Loading:
   - Use load_dataset(ds, [config], split=split, trust_remote_code=flag).
   - If --hf-token is provided, pass it. Prefer `token=...`, fallback to `use_auth_token=...`
     for older datasets. If loading fails (gated/403/404/whatever), warn and return 0 lines.

3) Text extraction:
   - Try fields in order: ["text","prompt","instruction","query","request","Behavior","behavior"].
   - If none found, fall back to the first non-empty string field with length ≥ 10.
   - Skip examples with empty text after stripping.

4) Shuffle & cap:
   - Attempt d.shuffle(seed=9172). If not supported, ignore error and proceed.
   - Write up to N examples per selected dataset (--n-adv / --n-jbv).

5) Output:
   - Each line is JSON: {"text": <str>, "label": "attack", "source": "<ds[:config]/split>"}.
   - Encoding: UTF-8; newline-terminated.

6) Run mode:
   - Use --run to control which dataset(s) actually execute.
   - If --n-adv == 0 or --run != 'adv'/'both', skip AdvBench.
   - If --n-jbv == 0 or --run != 'jbv'/'both', skip JBV.

7) Logging:
   - For each dataset run, print: "[done] {ds} → {out_path} ({written} lines)".
   - If AdvBench fails due to gating/repo issues, print:
       "[diag] AdvBench: skipped or empty (gated or load failed)."

Edge Cases
----------
- AdvBench is gated: require login/token; if not available, skip with a warning.
- JailbreakV-28K requires a config (e.g., JailBreakV_28K). If user changes it
  and it’s invalid, surface the underlying HF error.
- Different schemas: rely on auto text-field detection as described.

Non-Goals
---------
- No need to stream or to support compression.
- No multiprocessing or progress bars.

Acceptance Tests
----------------
1) JBV only:
   $ python scripts/curate_attack.py --run jbv --n-jbv 200 --out-jbv data/eval/jbv.jsonl
   - Produces ~200 lines, prints "[done] JailbreakV-28K/..." message.

2) AdvBench only (with token):
   $ huggingface-cli login
   $ python scripts/curate_attack.py --run adv --n-adv 100 --out-adv data/eval/adv.jsonl
   - Produces ~100 lines, prints "[done] walledai/AdvBench ..." message.

3) AdvBench without token:
   $ python scripts/curate_attack.py --run adv --n-adv 100 --out-adv data/eval/adv.jsonl
   - Prints a warning and writes 0 lines; JBV is not run.

4) Both (default):
   $ python data_tools/curate_attack.py
   - Runs both; honors --n-adv and --n-jbv defaults and out paths.

5) Text-field fallback:
   - Temporarily point --adv-ds to a JSON dataset with a "prompt" field only;
     ensure it is picked and non-empty lines are produced.

Implementation Hints
--------------------
- Use:
    from datasets import load_dataset, get_dataset_split_names
- Keep memory use low: write each line as you iterate.
- Make sure to catch and handle exceptions when loading gated datasets.
"""

import json
import random
import argparse
import sys
from typing import Optional, List, Tuple, Dict, Any
from datasets import load_dataset, get_dataset_split_names
import json
import os
import argparse


SEED = 9172
random.seed(SEED)

TEXT_FIELDS = ["prompt", "Goal"]


import os
import sys
import json
import random
import argparse
from typing import Optional, List, Tuple, Dict, Any

from datasets import load_dataset, get_dataset_split_names

SEED = 9172
random.seed(SEED)

# Prefer Behaviors keys first, but keep general fallbacks
TEXT_FIELDS = ["Goal", "prompt"]


def pick_text(ex: Dict[str, Any]) -> Optional[str]:
    """
    Return the first non-empty string from preferred fields.
    If none, return the first non-empty string (len ≥ 10) in the row.
    """
    # 1) Prefer known fields
    for k in TEXT_FIELDS:
        v = ex.get(k)
        if isinstance(v, str):
            s = v.strip()
            if s:
                return s

    # 2) Fallback: any other non-empty string with length ≥ 10
    for v in ex.values():
        if isinstance(v, str):
            s = v.strip()
            if len(s) >= 10:
                return s

    return None



def discover_split(ds: str, config: Optional[str], token: Optional[str]) -> str:
   """
   Discover a reasonable split to use for a dataset.

   Steps:
   - Try: splits = get_dataset_split_names(ds, config_name=config, token=token)
   - Choose in priority order: 'train' > 'training' > 'default' > 'all' > first.
   - On exception, return 'train'.
   """
   try:
      splits = get_dataset_split_names(ds, config_name=config, token=token)
      if not splits:
         return "train"

      lower_to_orig = {s.lower(): s for s in splits if isinstance(s, str)}

      for cand in ("train", "training", "default", "all"):
         if cand in lower_to_orig:
               return lower_to_orig[cand]

      return splits[0]
   except Exception:
      return "train"


def load_ds(ds: str, config: Optional[str], split: str, token: Optional[str], trust_remote_code: bool):
   """
   Load a dataset split with optional token and trust_remote_code.

   Steps:
   - Build kwargs = { 'split': split, 'trust_remote_code': trust_remote_code }.
   - If token is provided:
         * First try load_dataset(..., token=token, **kwargs).
         * If TypeError, fall back to load_dataset(..., use_auth_token=token, **kwargs).
   - If config is provided, pass it as the second argument to load_dataset.
   - Return the loaded dataset (or let exceptions bubble to caller for warning/skip).
   """
   kwargs = {"split": split, "trust_remote_code": trust_remote_code}

   # Positional args per HF signature: load_dataset(path, [name], **kwargs)
   args = [ds]
   if config is not None:
      args.append(config)

   if token:
      try:
         # Newer datasets use `token=...`
         return load_dataset(*args, token=token, **kwargs)
      except TypeError:
         # Older datasets use `use_auth_token=...`
         return load_dataset(*args, use_auth_token=token, **kwargs)
   else:
      return load_dataset(*args, **kwargs)



def dump_generic(
   ds: str,
   out_path: str,
   n: int,
   *,
   config: Optional[str] = None,
   split: Optional[str] = None,
   token: Optional[str] = None,
   trust_remote_code: bool = False,
   label: str = "attack",
   source_tag: Optional[str] = None,
) -> int:
   """
   Harvest up to `n` attack prompts from a single HF dataset and write JSONL.

   Behavior:
   - Determine split via discover_split if not provided.
   - Load dataset via load_ds (handles token/trust_remote_code).
   - Try to shuffle with seed=SEED (falls back to 9172 if SEED not defined).
   - Iterate examples, extract text via pick_text, skip empties.
   - Write JSONL lines: {"text": ..., "label": <label>, "source": <source_str>}
   - Print completion line and return number of lines written.
   """
   # Resolve split
   if split is None:
      try:
         split = discover_split(ds, config=config, token=token)
      except Exception as e:
         # Fallback per rules if discovery fails
         print(f"[warn] split discovery failed for {ds} (config={config}): {e}. Using 'train'.")
         split = "train"

   # Load dataset
   try:
      d = load_ds(
         ds,
         config=config,
         split=split,
         token=token,
         trust_remote_code=trust_remote_code,
      )
   except Exception as e:
      # Generic warning (caller/CLI can print the special AdvBench diag if desired)
      print(f"[warn] load failed for {ds} (config={config}, split={split}): {e}")
      return 0

   # Shuffle if supported
   try:
      try:
         _ = SEED  # type: ignore[name-defined]
      except NameError:
         SEED = 9172  # fallback
      d = d.shuffle(seed=SEED)  # some datasets may not support shuffle
   except Exception:
      pass  # proceed unshuffled

   # Ensure output directory exists
   out_dir = os.path.dirname(out_path)
   if out_dir:
      os.makedirs(out_dir, exist_ok=True)

   # Build source string
   if source_tag:
      source_str = source_tag
   else:
      cfg_part = f":{config}" if config else ""
      source_str = f"{ds}{cfg_part}/{split}"

   written = 0
   # Write JSONL, newline-terminated, UTF-8
   try:
      with open(out_path, "w", encoding="utf-8") as f:
         # Iterate dataset examples
         for ex in d:
               if written >= n:
                  break
               try:
                  text = pick_text(ex)
               except Exception as e:
                  # If pick_text crashes on a weird row, skip it
                  # (Keep it quiet; continue harvesting.)
                  continue

               if not text:
                  continue
               text = text.strip()
               if not text:
                  continue

               rec = {"text": text, "label": label, "source": source_str}
               f.write(json.dumps(rec, ensure_ascii=False) + "\n")
               written += 1
   except Exception as e:
      print(f"[warn] writing to {out_path} failed: {e}")
      return 0

   print(f"[done] {ds} → {out_path} ({written} lines)")
   return written



def main():
   """
   Wire up CLI and run selected jobs.
   """
   p = argparse.ArgumentParser(description="Harvest attack prompts to JSONL.")
   # Auth / load flags
   p.add_argument("--hf-token", dest="hf_token", default=None)
   p.add_argument("--trust-remote-code", action="store_true", dest="trust_remote_code")

   # Limits & outputs
   p.add_argument("--n-adv", type=int, default=1000)
   p.add_argument("--n-jbv", type=int, default=500)
   p.add_argument("--out-adv", default="data/eval/attack_1000.jsonl")
   p.add_argument("--out-jbv", default="data/eval/attack_extra_500.jsonl")

   # Run mode
   p.add_argument("--run", choices=["both", "adv", "jbv"], default="both")

   # AdvBench dataset knobs
   p.add_argument("--adv-ds", default="walledai/AdvBench")
   p.add_argument("--adv-config", default=None)
   p.add_argument("--adv-split", default=None)

   # JBV dataset knobs
   p.add_argument("--jbv-ds",    default="JailbreakBench/JBB-Behaviors")
   p.add_argument("--jbv-config", default="behaviors")   # Behaviors usually has no config
   p.add_argument("--jbv-split",  default=None)   # let discover_split pick (train)


   args = p.parse_args()

   def should(run_tag: str, n: int) -> bool:
      return (args.run in ("both", run_tag)) and (n > 0)

   # AdvBench
   if should("adv", args.n_adv):
      written = dump_generic(
         ds=args.adv_ds,
         out_path=args.out_adv,
         n=args.n_adv,
         config=args.adv_config,
         split=args.adv_split,
         token=args.hf_token,
         trust_remote_code=args.trust_remote_code,
         label="attack",
         source_tag=None,
      )
      if written == 0:
         print("[diag] AdvBench: skipped or empty (gated or load failed).")

   # JBV
   if should("jbv", args.n_jbv):
      _ = dump_generic(
         ds=args.jbv_ds,
         out_path=args.out_jbv,
         n=args.n_jbv,
         config=args.jbv_config,
         split=args.jbv_split,
         token=args.hf_token,
         trust_remote_code=args.trust_remote_code,
         label="attack",
         source_tag=None,
      )

   # Exit 0 explicitly (nice for shell usage)
   return 0



if __name__ == "__main__":
    main()
