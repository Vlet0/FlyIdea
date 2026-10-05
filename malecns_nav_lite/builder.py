"""
builder.py — Build the navigation-lite connectome from MaleCNS v1.0 feather files.

Pipeline
--------
1. Load annotations feather → neuron metadata (type, side, superclass).
2. Apply EXCLUDED_SUPERCLASS / EXCLUDED_TYPE_REGEX masks to remove
   olfactory, gustatory, and reproductive neurons.
3. Load neurotransmitters feather → assign synaptic sign.
4. Stream the weights feather batch-by-batch → build sparse weight matrix.
5. Select seed neurons using ALL_SEEDS regex list.
6. Run bidirectional BFS (``bfs_hops`` hops) from seeds.
7. Restrict subgraph to BFS result.
8. Drop neurons with internal degree < min_internal_degree.
9. Save NavLiteDB to output .npz.

Memory usage: the 1.1 GB weights feather is streamed in batches so peak RAM
stays around 3-5 GB rather than loading the full table at once.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
# scipy.sparse imported lazily (inside NavLiteDB.load and build_lite)
# to avoid Python 3.15 beta crash in scipy._ccallback.

from .bfs import bfs_subgraph, exclusion_mask, select_by_regex
from .constants import (
    ALL_SEEDS,
    BFS_HOPS,
    EXCLUDED_SUPERCLASS,
    EXCLUDED_TYPE_REGEX,
    MALECNS_FILES,
    MIN_INTERNAL_DEGREE,
    MIN_SYNAPSES,
    NT_SIGN,
)


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def _pick(cols: list[str], *candidates: str) -> str | None:
    low = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None


def _log(verbose: bool, *args, **kwargs):
    if verbose:
        print(*args, **kwargs)


# --------------------------------------------------------------------------- #
# NavLiteDB container
# --------------------------------------------------------------------------- #

class NavLiteDB:
    """Lightweight navigation-circuit database.

    Attributes
    ----------
    weights:    Signed CSC weight matrix  (n_nav × n_nav),  (post, pre).
    types:      Cell-type string array    (n_nav,).
    sides:      Side label ``"L"``/``"R"``/``""`` (n_nav,).
    superclass: Superclass string array   (n_nav,) or None.
    body_ids:   Original MaleCNS body IDs (n_nav,) or None.
    groups:     Named neuron-index sets (seeds: visual, cx, dn, wing, leg …).
    meta:       Provenance / stats dict.
    """

    def __init__(
        self,
        weights,  # scipy.sparse.csc_matrix or compatible sparse matrix
        types: np.ndarray,
        sides: np.ndarray,
        superclass: np.ndarray | None,
        body_ids: np.ndarray | None,
        groups: dict[str, np.ndarray],
        meta: dict,
    ):
        self.weights = weights
        self.types = types
        self.sides = sides
        self.superclass = superclass
        self.body_ids = body_ids
        self.groups = groups
        self.meta = meta

    @property
    def n(self) -> int:
        return self.weights.shape[0]

    @property
    def n_connections(self) -> int:
        return int(self.weights.nnz)

    @property
    def n_synapses(self) -> int:
        return int(np.abs(self.weights.data).sum())

    def summary(self) -> str:
        lines = [
            "NavLiteDB — navigation-circuit lite connectome",
            f"  Neurons    : {self.n:,}",
            f"  Connections: {self.n_connections:,}",
            f"  Synapses   : {self.n_synapses:,}",
            "  Groups:",
        ]
        for name, idx in sorted(self.groups.items()):
            lines.append(f"    {name:20s}: {idx.size:,} neurons")
        return "\n".join(lines)

    # ------------------------------------------------------------------ I/O #
    def save(self, path: str | Path) -> Path:
        """Save to compressed .npz (loadable without pyarrow)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        W = self.weights.tocsc()
        np.savez_compressed(
            path,
            indptr=W.indptr,
            indices=W.indices,
            data=W.data.astype(np.float32),
            shape=np.array(W.shape),
            types=self.types.astype(str),
            sides=self.sides.astype(str),
            superclass=(
                self.superclass
                if self.superclass is not None
                else np.array([], dtype=str)
            ).astype(str),
            body_ids=(
                self.body_ids
                if self.body_ids is not None
                else np.array([], dtype=np.int64)
            ),
            groups=json.dumps({k: v.tolist() for k, v in self.groups.items()}),
            meta=json.dumps(self.meta),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "NavLiteDB":
        """Load from .npz saved by :meth:`save`."""
        from scipy import sparse  # lazy import
        z = np.load(path, allow_pickle=False)
        W = sparse.csc_matrix(
            (z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"])
        )
        meta = json.loads(str(z["meta"]))
        groups = {
            k: np.asarray(v, dtype=np.int64)
            for k, v in json.loads(str(z["groups"])).items()
        }
        sc = z["superclass"]
        bid = z["body_ids"]
        return cls(
            weights=W,
            types=z["types"],
            sides=z["sides"],
            superclass=sc if sc.size else None,
            body_ids=bid if bid.size else None,
            groups=groups,
            meta=meta,
        )


# --------------------------------------------------------------------------- #
# Main builder
# --------------------------------------------------------------------------- #

def build_lite(
    data_dir: str | Path,
    min_synapses: int = MIN_SYNAPSES,
    bfs_hops: int = BFS_HOPS,
    min_internal_degree: int = MIN_INTERNAL_DEGREE,
    verbose: bool = True,
) -> NavLiteDB:
    """Build the navigation-lite connectome from MaleCNS v1.0 feather files.

    Parameters
    ----------
    data_dir:
        Directory containing the three MaleCNS .feather files.
    min_synapses:
        Edge weight threshold — edges below this are discarded.
    bfs_hops:
        BFS expansion radius from seed neurons (default 3).
    min_internal_degree:
        After BFS, remove neurons with fewer than this many kept connections.
    verbose:
        Print progress messages.

    Returns
    -------
    NavLiteDB
    """
    try:
        import pyarrow.feather as feather
        import pyarrow.ipc as ipc
    except ImportError as exc:
        raise SystemExit(
            "pyarrow is required to build from feather files.\n"
            "Install: pip install pyarrow\n"
            "Or load a pre-built .npz: NavLiteDB.load('nav_lite.npz')"
        ) from exc

    data_dir = Path(data_dir)
    log = _log

    # ------------------------------------------------------------------ #
    # 1. Annotations
    # ------------------------------------------------------------------ #
    ann_path = data_dir / MALECNS_FILES["annotations"]
    _log(verbose, f"Loading annotations: {ann_path.name}")
    ann = feather.read_table(ann_path).to_pandas()

    body_col = _pick(ann.columns.tolist(), "bodyId", "body_id", "body")
    type_col = _pick(ann.columns.tolist(), "type", "cell_type", "celltype")
    side_col = _pick(
        ann.columns.tolist(), "rootSide", "somaSide", "side", "root_side", "soma_side"
    )
    sc_col = _pick(ann.columns.tolist(), "superclass", "super_class")
    status_col = _pick(ann.columns.tolist(), "status", "statusLabel")

    if body_col is None or type_col is None:
        raise ValueError(f"Unexpected annotation columns: {list(ann.columns)[:20]}")

    # Basic quality filter
    keep = ann[body_col].notna()
    if status_col:
        keep &= ann[status_col].astype(str) != "Unimportant"
    ann = ann[keep].reset_index(drop=True)
    _log(verbose, f"  Raw neurons: {len(ann):,}")

    types_arr = ann[type_col].fillna("").astype(str).to_numpy()
    sides_arr = (
        ann[side_col].astype(str).str.upper().str[:1]
        .replace({"N": "", "U": ""}).to_numpy()
        if side_col else np.array([""] * len(ann))
    )
    sc_arr = ann[sc_col].astype(str).to_numpy() if sc_col else None
    body_ids = ann[body_col].to_numpy(np.int64)

    # ------------------------------------------------------------------ #
    # 2. Exclusion mask (olfactory / gustatory / reproductive)
    # ------------------------------------------------------------------ #
    exc = exclusion_mask(types_arr, sc_arr, EXCLUDED_SUPERCLASS, EXCLUDED_TYPE_REGEX)
    keep_mask = ~exc
    _log(verbose, f"  Excluded (olfactory/gustatory/reproductive): {exc.sum():,}")
    _log(verbose, f"  Remaining after exclusion: {keep_mask.sum():,}")

    # Remap to filtered indices
    original_indices = np.flatnonzero(keep_mask)
    types_f = types_arr[original_indices]
    sides_f = sides_arr[original_indices]
    sc_f = sc_arr[original_indices] if sc_arr is not None else None
    body_ids_f = body_ids[original_indices]

    nf = len(types_f)
    body_sorted_order = np.argsort(body_ids_f)
    body_sorted = body_ids_f[body_sorted_order]

    # ------------------------------------------------------------------ #
    # 3. Neurotransmitter signs
    # ------------------------------------------------------------------ #
    signs = np.full(nf, 1.0, dtype=np.float32)
    nt_path = data_dir / MALECNS_FILES["neurotransmitters"]
    if nt_path.exists():
        _log(verbose, f"Loading neurotransmitters: {nt_path.name}")
        nt = feather.read_table(nt_path).to_pandas()
        nb = _pick(nt.columns.tolist(), "body", "bodyId", "body_id")
        nc = _pick(
            nt.columns.tolist(), "consensus_nt", "consensusNt", "predicted_nt", "predictedNt"
        )
        if nb and nc:
            nt_map = dict(
                zip(nt[nb].to_numpy(np.int64), nt[nc].astype(str).str.lower())
            )
            for i, b in enumerate(body_ids_f):
                s = NT_SIGN.get(nt_map.get(int(b), ""), None)
                if s is not None:
                    signs[i] = s
    _log(verbose, f"  Inhibitory neurons: {(signs < 0).sum():,}")

    # ------------------------------------------------------------------ #
    # 4. Stream weights → sparse matrix (filtered neurons only)
    # ------------------------------------------------------------------ #
    _log(verbose, f"Streaming weights: {MALECNS_FILES['weights']}")
    wt_path = data_dir / MALECNS_FILES["weights"]
    src = ipc.open_file(str(wt_path))
    row_list, col_list, val_list = [], [], []
    total_scanned = 0

    for bi in range(src.num_record_batches):
        batch = src.get_batch(bi)
        names = batch.schema.names
        pre_col = _pick(names, "body_pre", "pre", "bodyId_pre")
        post_col = _pick(names, "body_post", "post", "bodyId_post")
        w_col = _pick(names, "weight", "synapses", "count")
        pre = batch.column(names.index(pre_col)).to_numpy()
        post = batch.column(names.index(post_col)).to_numpy()
        w = batch.column(names.index(w_col)).to_numpy()
        total_scanned += len(w)

        ok = w >= min_synapses
        pre, post, w = pre[ok], post[ok], w[ok]

        ip = np.searchsorted(body_sorted, pre)
        iq = np.searchsorted(body_sorted, post)
        ip = np.clip(ip, 0, len(body_sorted) - 1)
        iq = np.clip(iq, 0, len(body_sorted) - 1)
        valid = (body_sorted[ip] == pre) & (body_sorted[iq] == post) & (pre != post)

        pi = body_sorted_order[ip[valid]]
        qi = body_sorted_order[iq[valid]]
        row_list.append(qi)   # post = row
        col_list.append(pi)   # pre  = col
        val_list.append(w[valid].astype(np.float32) * signs[pi])

    rows = np.concatenate(row_list)
    cols = np.concatenate(col_list)
    vals = np.concatenate(val_list)
    from scipy import sparse  # lazy import — pyarrow already loaded, scipy safe here
    W_full = sparse.csc_matrix((vals, (rows, cols)), shape=(nf, nf), dtype=np.float32)
    _log(
        verbose,
        f"  Scanned {total_scanned:,} edges; kept {W_full.nnz:,} (≥{min_synapses} synapses)",
    )

    # ------------------------------------------------------------------ #
    # 5. Seed selection
    # ------------------------------------------------------------------ #
    _log(verbose, f"Selecting seed neurons (BFS seeds from {len(ALL_SEEDS)} patterns)…")
    seed_idx = select_by_regex(types_f, ALL_SEEDS)
    _log(verbose, f"  Seeds found: {seed_idx.size:,}")

    # Build named seed groups for the database
    from .constants import (
        VISUAL_SEED, CX_SEED, DN_SEED, WING_SEED, LEG_SEED, HALTERE_SEED, NECK_SEED
    )
    seed_groups_raw = {
        "visual":   VISUAL_SEED,
        "cx":       CX_SEED,
        "dn":       DN_SEED,
        "wing":     WING_SEED,
        "leg":      LEG_SEED,
        "haltere":  HALTERE_SEED,
        "neck":     NECK_SEED,
    }
    named_seeds: dict[str, np.ndarray] = {}
    for gname, patterns in seed_groups_raw.items():
        named_seeds[gname] = select_by_regex(types_f, patterns)
        _log(verbose, f"    {gname:12s}: {named_seeds[gname].size:,} seeds")

    # ------------------------------------------------------------------ #
    # 6. BFS expansion
    # ------------------------------------------------------------------ #
    # Sensorimotor logic: input = Visual; intermediate = CX; output = DN + Motor
    # Path: Visual (in) -> [Interneurons] -> DN + Motor (out)
    input_seeds = named_seeds["visual"]
    output_seeds = np.concatenate([
        named_seeds["dn"],
        named_seeds["wing"],
        named_seeds["leg"],
        named_seeds["haltere"],
        named_seeds["neck"],
    ])
    cx_seeds = named_seeds["cx"]
    
    _log(verbose, f"BFS expansion (hops={bfs_hops}, sensorimotor intersection)…")
    # For a path from input to output, we intersect forward reach from inputs 
    # with backward reach from outputs. We also ensure CX seeds are kept.
    bfs_keep = bfs_subgraph(W_full, input_seeds, output_seeds, hops=bfs_hops)
    # Also explicitly add CX seeds if they aren't already included
    bfs_keep = np.unique(np.concatenate([bfs_keep, cx_seeds]))
    
    _log(verbose, f"  BFS subgraph: {bfs_keep.size:,} neurons")

    # ------------------------------------------------------------------ #
    # 7. Restrict to BFS subgraph
    # ------------------------------------------------------------------ #
    remap = np.full(nf, -1, dtype=np.int64)
    remap[bfs_keep] = np.arange(bfs_keep.size)

    W_sub = W_full.tocsr()[bfs_keep][:, bfs_keep].tocsc()

    # ------------------------------------------------------------------ #
    # 8. Drop low-degree nodes
    # ------------------------------------------------------------------ #
    if min_internal_degree > 0:
        degree = np.asarray(
            np.abs(W_sub).sum(axis=0) + np.abs(W_sub).sum(axis=1).T
        ).ravel()
        degree_ok = degree >= min_internal_degree
        keep2 = np.flatnonzero(degree_ok)
        remap2 = np.full(bfs_keep.size, -1, dtype=np.int64)
        remap2[keep2] = np.arange(keep2.size)
        W_sub = W_sub.tocsr()[keep2][:, keep2].tocsc()
        bfs_keep = bfs_keep[keep2]
        _log(
            verbose,
            f"  After degree filter (≥{min_internal_degree}): {bfs_keep.size:,} neurons",
        )

    types_nav = types_f[bfs_keep]
    sides_nav = sides_f[bfs_keep]
    sc_nav = sc_f[bfs_keep] if sc_f is not None else None
    body_ids_nav = body_ids_f[bfs_keep]

    # Remap named-seed groups to subgraph indices
    groups_final: dict[str, np.ndarray] = {}
    for gname, fidx in named_seeds.items():
        mapped = remap[fidx]
        if min_internal_degree > 0:
            mapped = remap2[mapped[mapped >= 0]]
        groups_final[gname] = np.sort(mapped[mapped >= 0])

    # ------------------------------------------------------------------ #
    # 9. Assemble NavLiteDB
    # ------------------------------------------------------------------ #
    meta = {
        "name": "malecns-nav-lite",
        "source_version": "MaleCNS v1.0",
        "license": "CC-BY 4.0 — Janelia FlyEM, Univ. of Cambridge, MRC-LMB, Google Research",
        "min_synapses": min_synapses,
        "bfs_hops": bfs_hops,
        "min_internal_degree": min_internal_degree,
        "excluded_superclass": list(EXCLUDED_SUPERCLASS),
        "note": (
            "Navigation-circuit lite: visual → CX → DN → wing/leg motor neurons. "
            "Olfactory, gustatory, and reproductive circuits excluded."
        ),
    }
    db = NavLiteDB(
        weights=W_sub,
        types=types_nav,
        sides=sides_nav,
        superclass=sc_nav,
        body_ids=body_ids_nav,
        groups=groups_final,
        meta=meta,
    )
    _log(verbose, "\n" + db.summary())
    return db


# --------------------------------------------------------------------------- #
# Convenience re-export
# --------------------------------------------------------------------------- #
load_lite = NavLiteDB.load
