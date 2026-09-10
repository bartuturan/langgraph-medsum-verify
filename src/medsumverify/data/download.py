"""Fetch the MSLR Cochrane data and the human annotation file.

Deliberately does not use `datasets.load_dataset`: the allenai/mslr2022 repo
contains only a loading script, and script-based datasets stopped working in
`datasets` 3.0. The upstream tarball is five CSVs, so we parse it ourselves.
"""

from __future__ import annotations

import tarfile
from pathlib import Path

import pandas as pd
import requests

from ..config import (
    ANNOTATIONS_URL,
    COCHRANE_FILES,
    MSLR_TARBALL_URL,
    paths,
)

_CHUNK = 1 << 20


def _stream_download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[cache] {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest
    print(f"[get ] {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        step = max(total // 10, 1)  # ~10 lines, not 250 -- logs get read in notebooks
        next_mark = step
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(_CHUNK):
                fh.write(chunk)
                done += len(chunk)
                if total and done >= next_mark:
                    print(f"       {done / 1e6:6.1f} / {total / 1e6:.1f} MB")
                    next_mark += step
    tmp.replace(dest)
    return dest


def _find_cochrane_dir(extract_root: Path) -> Path:
    """The archive layout has moved between releases; locate it by content."""
    for candidate in extract_root.rglob("*"):
        if candidate.is_dir() and candidate.name == "cochrane":
            if (candidate / "dev-inputs.csv").exists():
                return candidate
    raise FileNotFoundError(
        f"no cochrane/ directory with dev-inputs.csv found under {extract_root}"
    )


def ensure_cochrane() -> Path:
    """Download + extract + convert to parquet. Returns the parquet directory."""
    p = paths()
    out = p.cochrane_parquet
    expected = [out / f"{k}.parquet" for k in COCHRANE_FILES]
    if all(f.exists() for f in expected):
        print(f"[cache] cochrane parquet ready ({out})")
        return out

    tarball = _stream_download(MSLR_TARBALL_URL, p.tarball)
    extract_root = p.cache / "mslr_extracted"
    if not extract_root.exists():
        print(f"[tar ] extracting to {extract_root}")
        extract_root.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tarball, "r:gz") as tf:
            # Guard against path traversal in a third-party archive.
            safe = [m for m in tf.getmembers() if not (m.name.startswith("/") or ".." in Path(m.name).parts)]
            tf.extractall(extract_root, members=safe)

    src = _find_cochrane_dir(extract_root)
    out.mkdir(parents=True, exist_ok=True)
    for key, fname in COCHRANE_FILES.items():
        csv = src / fname
        if not csv.exists():
            # test-targets.csv is legitimately absent; anything else is a problem.
            raise FileNotFoundError(f"expected {csv}")
        df = pd.read_csv(csv)
        df.to_parquet(out / f"{key}.parquet", index=False)
        print(f"[conv] {fname:20s} -> {key}.parquet  rows={len(df):>6}  cols={list(df.columns)}")

    if (src / "test-targets.csv").exists():
        print("[note] test-targets.csv exists upstream after all -- revisit the split design")
    return out


def ensure_annotations() -> Path:
    p = paths()
    return _stream_download(ANNOTATIONS_URL, p.annotations_json)


def ensure_all() -> None:
    ensure_cochrane()
    ensure_annotations()


if __name__ == "__main__":
    ensure_all()
