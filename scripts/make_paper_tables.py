from __future__ import annotations

import argparse
import json
from pathlib import Path


TOKENIZER_COLUMNS = [
    ("file", "Run"),
    ("metrics.tokens_per_char", "TPC"),
    ("metrics.stem_fragmentation_rate", "Frag"),
    ("metrics.benign_hazard_piece_overlap", "Overlap"),
    ("metrics.boundary_jaccard", "Jaccard"),
    ("metrics.boundary_flip_rate", "Flip"),
    ("metrics.segmentation_changed_rate", "Changed"),
]


def get_path(obj: dict, dotted: str):
    cur = obj
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def row_name(path: Path) -> str:
    name = path.stem
    for prefix in ("tokspill_", "jailbreak_", "segstab_"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    return name.replace("_", " ")


def make_markdown(paths: list[Path], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, sep]
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["file"] = row_name(path)
        cells = [fmt(get_path(payload, key)) for key, _ in columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def make_latex(paths: list[Path], columns: list[tuple[str, str]]) -> str:
    lines = []
    lines.append("\\begin{tabular}{" + "l" + "r" * (len(columns) - 1) + "}")
    lines.append("\\toprule")
    lines.append(" & ".join(label for _, label in columns) + r" \\")
    lines.append("\\midrule")
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["file"] = row_name(path)
        cells = [fmt(get_path(payload, key)) for key, _ in columns]
        lines.append(" & ".join(cells) + r" \\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper-ready metric tables.")
    parser.add_argument("--tokenizer_json", nargs="+", required=True)
    parser.add_argument("--out_prefix", default="outputs/table_tokenizer")
    parser.add_argument("--format", choices=["both", "markdown", "latex"], default="both")
    args = parser.parse_args()

    paths = [Path(p) for p in args.tokenizer_json]
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    if args.format in ("both", "markdown"):
        out_prefix.with_suffix(".md").write_text(
            make_markdown(paths, TOKENIZER_COLUMNS), encoding="utf-8"
        )
    if args.format in ("both", "latex"):
        out_prefix.with_suffix(".tex").write_text(
            make_latex(paths, TOKENIZER_COLUMNS), encoding="utf-8"
        )

    print(f"Wrote tables with prefix {out_prefix}")


if __name__ == "__main__":
    main()
