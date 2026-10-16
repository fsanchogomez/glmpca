from __future__ import annotations

import anndata as ad
import numpy as np
import torch
from scipy.sparse import issparse
from scipy.stats import spearmanr
from tqdm.auto import tqdm

DEPTH_CORRELATION_CUTOFF = 0.75
"""|Spearman rho| with the depth above which drop_depth drops a component."""

DEPTH_CHUNK_ROWS = 16384
"""Rows of a block when the depth is read from a backed AnnData."""


def cell_depth(X: ad.AnnData | torch.Tensor | np.ndarray) -> np.ndarray:
    """The total count of every cell of `X`, cells in rows, in float64.

    It reads the counts as they are stored, so binarize, which the fits apply when they
    read the matrix, does not change it. A backed AnnData is read by blocks of rows.
    """
    counts = X.X if isinstance(X, ad.AnnData) else X
    if isinstance(counts, torch.Tensor):
        return counts.detach().sum(dim=1, dtype=torch.float64).cpu().numpy()
    if isinstance(counts, np.ndarray) or issparse(counts):
        return np.asarray(counts.sum(axis=1), dtype=np.float64).ravel()
    n = counts.shape[0]
    depth = np.empty(n)
    for start in range(0, n, DEPTH_CHUNK_ROWS):
        block = counts[start : start + DEPTH_CHUNK_ROWS]
        depth[start : start + DEPTH_CHUNK_ROWS] = np.asarray(
            block.sum(axis=1), dtype=np.float64
        ).ravel()
    return depth


def components_to_keep(
    scores: torch.Tensor | np.ndarray, depth: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The components whose scores do not follow `depth`, and every correlation.

    A component goes when the |Spearman rho| of its scores with the depth of the cells
    is above DEPTH_CORRELATION_CUTOFF. It says how many go.
    """
    columns = np.asarray(scores, dtype=np.float64).T
    correlations = np.nan_to_num([
        spearmanr(column, depth).statistic for column in columns
    ])
    dropped = np.abs(correlations) > DEPTH_CORRELATION_CUTOFF
    tqdm.write(
        f"DEPTH: {int(dropped.sum())} of {len(correlations)} components dropped, "
        f"|Spearman rho| with cell depth above {DEPTH_CORRELATION_CUTOFF}"
    )
    return np.flatnonzero(~dropped), correlations
