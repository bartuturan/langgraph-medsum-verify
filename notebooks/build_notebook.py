"""Turn kaggle_run.py into kaggle_run.ipynb.

The cells live in a .py file so git can diff them; this script is the only
thing that touches .ipynb JSON.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE / "kaggle_run.py"
DEST = HERE / "kaggle_run.ipynb"

CELL_RE = re.compile(
    r"# -+ CELL (\d+) -+\n(.*?)(?=\n# -+ CELL |\Z)", re.S
)


def build() -> Path:
    text = SRC.read_text(encoding="utf-8")
    cells = []
    for _, body in CELL_RE.findall(text):
        comments, code = [], ""
        m = re.search(r'"""\n(.*?)"""', body, re.S)
        if m:
            code = m.group(1).strip("\n")
            comments = [
                line for line in body[: m.start()].splitlines() if line.strip().startswith("#")
            ]
        if comments:
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [re.sub(r"^#\s?", "", c) + "\n" for c in comments],
            })
        if code:
            cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [l + "\n" for l in code.splitlines()],
            })

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    DEST.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"[write] {DEST}  ({len(cells)} cells)")
    return DEST


if __name__ == "__main__":
    build()
