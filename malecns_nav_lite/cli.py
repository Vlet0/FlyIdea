"""
cli.py — Command-line interface for malecns_nav_lite.

Commands
--------
  python -m malecns_nav_lite download   — download raw feather files
  python -m malecns_nav_lite build      — build navigation-lite .npz
  python -m malecns_nav_lite inspect    — print summary of a .npz
  python -m malecns_nav_lite query      — query neurons by type regex
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def cmd_download(args):
    from .download import download_raw
    download_raw(data_dir=args.dir, files=args.files or None)


def cmd_build(args):
    from .builder import build_lite
    db = build_lite(
        data_dir=args.data_dir,
        min_synapses=args.min_synapses,
        bfs_hops=args.bfs_hops,
        min_internal_degree=args.min_degree,
        verbose=True,
    )
    out = Path(args.out)
    db.save(out)
    print(f"\nSaved: {out}  ({out.stat().st_size / 1e6:.1f} MB)")


def cmd_inspect(args):
    from .builder import NavLiteDB
    db = NavLiteDB.load(args.db)
    print(db.summary())
    print("\nMeta:")
    for k, v in db.meta.items():
        print(f"  {k}: {v}")


def cmd_query(args):
    from .builder import NavLiteDB
    from .bfs import select_by_regex
    db = NavLiteDB.load(args.db)
    idx = select_by_regex(db.types, [args.pattern])
    if idx.size == 0:
        print("No neurons matched.")
        return
    print(f"Matched {idx.size} neurons:")
    for i in idx[:50]:
        bid = db.body_ids[i] if db.body_ids is not None else "?"
        sc = db.superclass[i] if db.superclass is not None else "?"
        print(f"  [{i:6d}]  type={db.types[i]!r:30s}  side={db.sides[i]}  sc={sc}  body={bid}")
    if idx.size > 50:
        print(f"  … and {idx.size - 50} more")


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="python -m malecns_nav_lite",
        description="MaleCNS navigation-lite connectome toolkit.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # download
    dl = sub.add_parser("download", help="Download raw MaleCNS feather files.")
    dl.add_argument("--dir", default="data/raw", help="Destination directory")
    dl.add_argument(
        "--files", nargs="+",
        choices=["annotations", "neurotransmitters", "weights"],
        help="Subset of files to download",
    )
    dl.set_defaults(func=cmd_download)

    # build
    bd = sub.add_parser("build", help="Build navigation-lite .npz from feather files.")
    bd.add_argument("--data-dir", default="data/raw", help="Directory with .feather files")
    bd.add_argument("--out", default="data/nav_lite.npz", help="Output .npz path")
    bd.add_argument("--min-synapses", type=int, default=3, help="Min edge weight")
    bd.add_argument("--bfs-hops", type=int, default=3, help="BFS expansion hops")
    bd.add_argument("--min-degree", type=int, default=1, help="Min internal degree")
    bd.set_defaults(func=cmd_build)

    # inspect
    ins = sub.add_parser("inspect", help="Print summary of a nav_lite .npz.")
    ins.add_argument("db", help="Path to nav_lite .npz file")
    ins.set_defaults(func=cmd_inspect)

    # query
    qr = sub.add_parser("query", help="Query neurons by type regex.")
    qr.add_argument("db", help="Path to nav_lite .npz file")
    qr.add_argument("pattern", help="Regex pattern for cell type")
    qr.set_defaults(func=cmd_query)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
