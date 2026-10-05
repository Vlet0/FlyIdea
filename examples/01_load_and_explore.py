"""
examples/01_load_and_explore.py — Load the nav_lite database and explore.

Run after building:
    python -m malecns_nav_lite build --data-dir data/raw --out data/nav_lite.npz
    python examples/01_load_and_explore.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

# Add parent to path if running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from malecns_nav_lite import load_lite


def main():
    db_path = Path("data/nav_lite.npz")
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        print("Build first: python -m malecns_nav_lite build --data-dir data/raw --out data/nav_lite.npz")
        return

    print("Loading navigation-lite connectome …")
    db = load_lite(db_path)
    print(db.summary())
    print()

    # --------------------------------------------------------------------- #
    # Group statistics
    # --------------------------------------------------------------------- #
    print("=== Group sizes ===")
    for name, idx in sorted(db.groups.items()):
        L = (db.sides[idx] == "L").sum()
        R = (db.sides[idx] == "R").sum()
        print(f"  {name:12s}: {idx.size:5d} total  ({L}L / {R}R)")

    # --------------------------------------------------------------------- #
    # Connectivity between groups
    # --------------------------------------------------------------------- #
    print("\n=== Connections between groups (synapse count matrix) ===")
    group_names = sorted(db.groups.keys())
    header = f"{'':12s}" + "".join(f"{n:>10s}" for n in group_names)
    print(header)
    for pre_name in group_names:
        pre_idx = db.groups[pre_name]
        row = f"{pre_name:12s}"
        for post_name in group_names:
            post_idx = db.groups[post_name]
            if pre_idx.size and post_idx.size:
                W_sub = db.weights[post_idx][:, pre_idx]
                syn = int(np.abs(W_sub.data).sum())
            else:
                syn = 0
            row += f"{syn:>10,}"
        print(row)

    # --------------------------------------------------------------------- #
    # Top 10 most connected neurons in each major group
    # --------------------------------------------------------------------- #
    print("\n=== Top 10 central-complex neurons by in-degree ===")
    cx_idx = db.groups.get("cx", np.array([], dtype=np.int64))
    if cx_idx.size:
        W_cx = db.weights[cx_idx][:, :]
        in_deg = np.asarray(np.abs(W_cx).sum(axis=1)).ravel()
        top10 = np.argsort(-in_deg)[:10]
        for rank, j in enumerate(top10):
            nidx = cx_idx[j]
            print(
                f"  #{rank+1:2d}  type={db.types[nidx]!r:20s}  "
                f"side={db.sides[nidx]}  in_synapses={int(in_deg[j]):,}"
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
