"""
bfs.py — BFS-based sensorimotor subgraph extraction.

Algorithm
---------
Mirrors the ``sensorimotor_core`` method in evo_flydrone's connectome.py but is
more flexible: seed neurons are selected by regex, BFS propagates in BOTH
directions (forward = pre→post, backward = post→pre), and the result keeps only
neurons reachable from ANY seed within ``hops`` steps in either direction.

BFS over a sparse (post, pre) adjacency A:
  • forward propagation  : A  @ frontier   (seed fires, follow axon downstream)
  • backward propagation : Aᵀ @ frontier   (find everything that drives seeds)

We keep the UNION of forward and backward reach so that:
  1. Upstream visual neurons that activate the CX are included.
  2. Downstream motor neurons driven by the CX are included.
  3. Interneurons bridging seed groups are included.

Neurons are excluded BEFORE BFS if they match EXCLUDED_SUPERCLASS or
EXCLUDED_TYPE_REGEX (olfactory / gustatory / reproductive filters).
"""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np
# NOTE: scipy.sparse is imported LAZILY inside functions so that this module
# can be imported on Python 3.15 beta where the installed scipy wheel may
# crash on import (access violation in _ccallback.pyd).




# --------------------------------------------------------------------------- #
# Low-level BFS
# --------------------------------------------------------------------------- #

def _bfs_reach_numpy(indptr: np.ndarray, indices: np.ndarray, seeds: np.ndarray, n: int, hops: int) -> np.ndarray:
    """Pure-numpy BFS on a CSR adjacency matrix (fallback for scipy crashes).

    *indptr* and *indices* are the CSR row-pointer and column-index arrays of
    a boolean (n × n) adjacency matrix where entry (i, j) means i is reachable
    from j in one hop (i.e., post=i, pre=j, same as A @ frontier convention).
    """
    seen = np.zeros(n, dtype=bool)
    seen[seeds] = True
    frontier = seeds.copy()
    for _ in range(hops):
        # Expand: for each node in frontier, collect all its row-neighbours
        next_nodes = []
        for node in frontier:
            start, end = int(indptr[node]), int(indptr[node + 1])
            nbrs = indices[start:end]
            next_nodes.append(nbrs)
        if not next_nodes:
            break
        nbrs_all = np.concatenate(next_nodes) if next_nodes else np.array([], dtype=np.intp)
        new_frontier = nbrs_all[~seen[nbrs_all]] if nbrs_all.size else np.array([], dtype=np.intp)
        new_frontier = np.unique(new_frontier)
        if new_frontier.size == 0:
            break
        seen[new_frontier] = True
        frontier = new_frontier
    return seen


def _bfs_reach(A, seeds: np.ndarray, hops: int) -> np.ndarray:
    """Return boolean mask of nodes reachable from *seeds* in ≤ *hops* along A.

    A is a (post, pre) adjacency matrix.  The forward step follows edges from
    pre → post (same convention as evo_flydrone's ``_reach``).

    Tries scipy sparse mat-vec first; falls back to pure-numpy BFS on any error
    (scipy may crash on Python 3.15 beta with certain builds).
    """
    n = A.shape[0]
    try:
        from scipy import sparse as _sp
        A_csr = _sp.csr_matrix(A, dtype=np.float32)
        frontier = np.zeros(n, dtype=np.float32)
        frontier[seeds] = 1.0
        seen = frontier > 0
        for _ in range(hops):
            next_front = (A_csr @ frontier > 0).astype(np.float32)
            next_front[seen] = 0.0
            if not next_front.any():
                break
            seen |= next_front > 0
            frontier = next_front
        return seen
    except Exception:
        # Numpy-only fallback: convert to CSR-like arrays manually
        try:
            csr = A.tocsr()
            indptr = np.asarray(csr.indptr)
            indices = np.asarray(csr.indices)
        except Exception:
            # A might already be ndarray (dense); build from dense
            A_dense = np.asarray(A, dtype=bool)
            indptr = np.zeros(n + 1, dtype=np.intp)
            indices_list = []
            for i in range(n):
                row_nz = np.flatnonzero(A_dense[i])
                indptr[i + 1] = indptr[i] + len(row_nz)
                indices_list.append(row_nz)
            indices = np.concatenate(indices_list).astype(np.intp) if indices_list else np.array([], dtype=np.intp)
        return _bfs_reach_numpy(indptr, indices, seeds, n, hops)


def bfs_subgraph(
    weights,
    input_seeds: np.ndarray,
    output_seeds: np.ndarray,
    hops: int = 3,
) -> np.ndarray:
    """Return indices of neurons on paths from inputs to outputs within `hops`.

    Mirrors ``sensorimotor_core`` from evo_flydrone:
    We keep neurons that are ≤ ``hops`` downstream of inputs AND ≤ ``hops``
    upstream of outputs. Seed neurons themselves are always included.

    Parameters
    ----------
    weights:
        (n, n) sparse signed weight matrix (post, pre).
    input_seeds:
        Integer indices of input neurons (e.g., visual).
    output_seeds:
        Integer indices of output neurons (e.g., DN, motor).
    hops:
        Maximum synaptic hops from inputs/outputs.

    Returns
    -------
    np.ndarray
        Sorted unique integer indices of included neurons.
    """
    if input_seeds.size == 0 or output_seeds.size == 0:
        return np.union1d(input_seeds, output_seeds)

    A_bool = (weights != 0)  # (post, pre)

    # Forward: reachable from inputs (seed → downstream)
    fwd = _bfs_reach(A_bool, input_seeds, hops)
    
    # Backward: reachable from outputs (seed ← upstream)
    # T.tocsc() is needed if it's a scipy sparse matrix
    A_bwd = A_bool.T
    if hasattr(A_bwd, "tocsc"):
        A_bwd = A_bwd.tocsc()
    bwd = _bfs_reach(A_bwd, output_seeds, hops)

    # Intersection of forward and backward reach
    keep_mask = fwd & bwd

    # Convert to indices
    keep_idx = np.flatnonzero(keep_mask)
    
    # Always include all original seeds
    return np.unique(np.concatenate([keep_idx, input_seeds, output_seeds]))


# --------------------------------------------------------------------------- #
# Regex helpers
# --------------------------------------------------------------------------- #

def _compile_patterns(patterns: Sequence[str]) -> list[re.Pattern]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]


def match_any(text: str, patterns: list[re.Pattern]) -> bool:
    return any(rx.search(text) for rx in patterns)


def select_by_regex(
    types: np.ndarray,
    patterns: Sequence[str],
    side: str | None = None,
    sides: np.ndarray | None = None,
) -> np.ndarray:
    """Return indices of neurons whose cell type matches any of *patterns*.

    Parameters
    ----------
    types:    String array of cell-type labels, length n.
    patterns: Iterable of regex strings.
    side:     Optional ``"L"`` / ``"R"`` filter.
    sides:    String array of side labels, same length as *types*.
    """
    compiled = _compile_patterns(patterns)
    mask = np.fromiter(
        (match_any(str(t), compiled) for t in types), dtype=bool, count=len(types)
    )
    if side and sides is not None:
        mask &= np.char.upper(sides.astype(str)) == side.upper()
    return np.flatnonzero(mask)


def exclusion_mask(
    types: np.ndarray,
    superclass: np.ndarray | None,
    excluded_superclass: set[str],
    excluded_type_regex: Sequence[str],
) -> np.ndarray:
    """Return boolean mask True for neurons to EXCLUDE.

    Parameters
    ----------
    types:
        Cell-type string array.
    superclass:
        Superclass string array (same length) or None.
    excluded_superclass:
        Set of superclass strings to drop (case-insensitive).
    excluded_type_regex:
        Regex patterns matching types to drop.
    """
    n = len(types)
    exc = np.zeros(n, dtype=bool)

    # Superclass filter
    if superclass is not None:
        sc_lower = np.char.lower(superclass.astype(str))
        for sc in excluded_superclass:
            exc |= sc_lower == sc.lower()

    # Type-regex filter
    compiled = _compile_patterns(excluded_type_regex)
    exc |= np.fromiter(
        (match_any(str(t), compiled) for t in types), dtype=bool, count=n
    )
    return exc
