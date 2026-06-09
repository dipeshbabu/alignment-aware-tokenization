from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train drift-LoRA using a native edited BPE tokenizer."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--save", required=True)
    parser.add_argument("--neutrals", default="")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["tokenizer_name"] = args.tokenizer
    cfg.setdefault("out_dir", str(Path(args.save) / "eval_logs"))

    tmp_dir = Path("runs") / "generated_configs"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_cfg = tmp_dir / f"{Path(args.save).name}_native_bpe.yml"
    out_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    cmd = [
        sys.executable,
        "-m",
        "models.lora_drift",
        "--config",
        str(out_cfg),
        "--probe",
        args.probe,
        "--save",
        args.save,
    ]
    if args.neutrals:
        cmd.extend(["--neutrals", args.neutrals])

    print("[native-bpe-lora]", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
