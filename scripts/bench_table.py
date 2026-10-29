"""Markdown table of train benchmarks from saved pytest-benchmark runs.

    uv run pytest benchmarks/test_train_adversarial.py --benchmark-only --benchmark-save=NAME
    uv run python scripts/bench_table.py LABEL=.benchmarks/.../0001_NAME.json [LABEL=...]

Rows are grouped by case; each file contributes its "current" rows under its LABEL, and
"naive" rows are taken from the first file only (the naive trainer does not change).
"""

import json
import re
import sys
from pathlib import Path
from typing import Any

CASE = re.compile(r"test_train_(?:adversarial\[(\w+)-(\d+)-(\w+)\]|regular\[(\d+)\])")


def _case(name: str) -> tuple[str, str]:
    match = CASE.fullmatch(name)
    if match is None:
        raise ValueError(f"unexpected benchmark name: {name}")
    kind, length, impl, vocab = match.groups()
    if vocab is not None:
        return f"regular 1MB vocab {vocab}", "current"
    return f"{kind} {int(length) // 1000}k", impl


def _cell(info: dict[str, Any], key: str, fmt: str) -> str:
    return format(info[key], fmt) if key in info else "-"


def main(specs: list[str]) -> None:
    rows: dict[str, list[str]] = {}
    for n, spec in enumerate(specs):
        label, path = spec.split("=", 1)
        for bench in json.loads(Path(path).read_text())["benchmarks"]:
            case, impl = _case(bench["name"])
            if impl == "naive" and n > 0:
                continue
            info = bench["extra_info"]
            rows.setdefault(case, []).append(
                f"| {case} | {label if impl == 'current' else 'naive'} "
                f"| {bench['stats']['median']:.3f} | {info['peak_mib']:.1f} "
                f"| {_cell(info, 'heap_pushes', ',')} | {_cell(info, 'heapify_entries', ',')} "
                f"| {_cell(info, 'max_dead_ratio', '.1%')} "
                f"| {_cell(info, 'final_dead_ratio', '.1%')} |"
            )
    print(
        "| case | impl | median s | peak MiB | heap pushes | heapified entries "
        "| max dead | final dead |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for lines in rows.values():
        print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1:])
