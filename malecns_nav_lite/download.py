"""
download.py — Download MaleCNS v1.0 flat-connectome feather files with
resume support and progress reporting.

Usage (CLI):
    python -m malecns_nav_lite.download --dir data/raw

The three files total ~1.2 GB (CC-BY 4.0).
"""

from __future__ import annotations

import sys
import time
import urllib.request
from pathlib import Path

from .constants import MALECNS_BASE, MALECNS_FILES


def _download_one(url: str, dest: Path, chunk: int = 1 << 20) -> Path:
    """Download *url* to *dest*, resuming if a partial file exists."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    have = tmp.stat().st_size if tmp.exists() else 0
    req = urllib.request.Request(
        url, headers={"Range": f"bytes={have}-"} if have else {}
    )
    try:
        with urllib.request.urlopen(req) as resp:
            total = int(resp.headers.get("Content-Length", 0)) + have
            mode = "ab" if have and resp.status == 206 else "wb"
            if mode == "wb":
                have = 0
            t0 = time.time()
            with open(tmp, mode) as fh:
                while True:
                    block = resp.read(chunk)
                    if not block:
                        break
                    fh.write(block)
                    have += len(block)
                    if total:
                        mb = have / 1e6
                        speed = mb / max(time.time() - t0, 1e-3)
                        sys.stdout.write(
                            f"\r  {dest.name}: {mb:8.1f} / {total / 1e6:.1f} MB"
                            f"  ({speed:.1f} MB/s)"
                        )
                        sys.stdout.flush()
    except Exception as exc:
        sys.stdout.write(f"\n  ERROR downloading {dest.name}: {exc}\n")
        raise
    tmp.replace(dest)
    sys.stdout.write(f"\n  {dest.name}: done ({have / 1e6:.1f} MB)\n")
    return dest


def download_raw(
    data_dir: str | Path = "data/raw",
    skip_existing: bool = True,
    files: list[str] | None = None,
) -> Path:
    """Download MaleCNS v1.0 connectome files to *data_dir*.

    Parameters
    ----------
    data_dir:
        Target directory (created if absent).
    skip_existing:
        If True (default), skip files that already exist at full size.
    files:
        Subset of ``["annotations", "neurotransmitters", "weights"]``.
        If None, all three are downloaded.

    Returns
    -------
    Path
        Resolved *data_dir*.
    """
    data_dir = Path(data_dir).resolve()
    keys = files or list(MALECNS_FILES)
    print(
        "MaleCNS v1.0 — Janelia FlyEM, Univ. of Cambridge, MRC-LMB, Google Research"
        " — CC-BY 4.0\n"
        f"  Target: {data_dir}\n"
        f"  Files : {', '.join(keys)}\n"
    )
    for key in keys:
        name = MALECNS_FILES[key]
        dest = data_dir / name
        if skip_existing and dest.exists() and dest.stat().st_size > 1_000:
            print(f"  {name}: already present, skipping.")
            continue
        url = f"{MALECNS_BASE}/{name}"
        print(f"  Fetching {url}")
        _download_one(url, dest)
    print("Download complete.")
    return data_dir


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Download MaleCNS v1.0 feather files.")
    ap.add_argument("--dir", default="data/raw", help="Destination directory")
    ap.add_argument(
        "--files",
        nargs="+",
        choices=list(MALECNS_FILES),
        default=None,
        help="Subset of files to download",
    )
    args = ap.parse_args()
    download_raw(args.dir, files=args.files)
