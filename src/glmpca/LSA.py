"""Latent semantic analysis: the truncated SVD of a TF-IDF matrix, solved directly.

`LSA` computes it with an SVD solver: ARPACK (`scipy.sparse.linalg.svds`) on a matrix in
memory, or a randomized SVD in torch that reads the matrix by blocks of rows, on the CPU
or the GPU, from memory or from a backed AnnData.

The LSA implementation follows gensim's `TfidfModel` with `smartirs='lfu'` followed by
`LsiModel`.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

import anndata as ad
import numpy as np
import torch
from scipy.sparse import csr_matrix, spmatrix
from scipy.sparse.linalg import svds
from tqdm.auto import tqdm

from .GLMPCA import (
    DEFAULT_RANDOM_STATE,
    _announce_device,
    _resolve_device,
    _to_sparse,
)
from .manifolds import largest_entry_signs
from .sparse import BackedRows, SparseRows, binary
from .utils import cell_depth, components_to_keep

if TYPE_CHECKING:
    from collections.abc import Iterator

DEFAULT_CHUNK_ROWS = 16384
SOLVERS = ("auto", "arpack", "randomized")
# Slope of gensim's pivoted unique normalization, `smartirs="lfu"`.
GENSIM_SLOPE = 0.25


def idf(holders: np.ndarray, n_cells: int) -> tuple[np.ndarray, float]:
    """The idf of every feature, and the pivot of the normalization.

    `holders` are the cells that hold every feature. The idf is `log2(cells / holders)`,
    zero for a feature that no cell holds, and the pivot is the mean number of features
    that a cell holds.
    """
    feature_idf = np.where(
        holders > 0, np.log2(n_cells / np.clip(holders, 1.0, None)), 0.0
    )
    return feature_idf, float(holders.sum()) / n_cells


def tfidf(block: csr_matrix, feature_idf: np.ndarray, pivot: float) -> csr_matrix:
    r"""The TF-IDF of the CSR `block`, computed on its stored entries.

    It is gensim's `TfidfModel(normalize=True, smartirs="lfu")`, for a count above
    zero:

        (1 + log2(count)) · idf / ((1 - GENSIM_SLOPE) · pivot + GENSIM_SLOPE · kept)

    with `kept` the features of the cell whose weight is above zero. A zero stays zero,
    so the result has the entries of `block`.
    """
    counts = np.asarray(block.data, dtype=np.float64)
    rows = np.repeat(np.arange(block.shape[0]), np.diff(block.indptr))
    held = counts > 0
    weight = np.where(
        held,
        (1.0 + np.log2(np.where(held, counts, 1.0))) * feature_idf[block.indices],
        0.0,
    )
    kept = np.bincount(rows[weight > 0], minlength=block.shape[0])
    norm = (1.0 - GENSIM_SLOPE) * pivot + GENSIM_SLOPE * kept
    values = weight / norm[rows]
    return csr_matrix(
        (values.astype(np.float32), block.indices, block.indptr), shape=block.shape
    )


def _torch_csr(block: csr_matrix, device: torch.device) -> torch.Tensor:
    """`block` as a torch CSR tensor on `device`."""
    index = np.promote_types(block.indptr.dtype, block.indices.dtype)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Sparse CSR tensor support")
        return torch.sparse_csr_tensor(
            torch.from_numpy(block.indptr.astype(index, copy=False)),
            torch.from_numpy(block.indices.astype(index, copy=False)),
            torch.from_numpy(np.asarray(block.data, dtype=np.float32)),
            size=block.shape,
            check_invariants=False,
        ).to(device)


class _TfidfRows(SparseRows):
    """The TF-IDF of `source`, made block by block as it is read."""

    def __init__(
        self, source: SparseRows, feature_idf: np.ndarray, pivot: float
    ) -> None:
        self.source = source
        self.feature_idf = feature_idf
        self.pivot = pivot

    @property
    def shape(self) -> torch.Size:
        return self.source.shape

    def _block(self, rows: slice | torch.Tensor | np.ndarray) -> csr_matrix:
        block = csr_matrix(self.source._block(rows), dtype=np.float32)
        return tfidf(block, self.feature_idf, self.pivot)


class LSA:
    r"""Latent semantic analysis of a cell-by-feature count matrix.

    The counts are turned into the TF-IDF of gensim's `TfidfModel(smartirs="lfu")`
    (`tfidf`), and the components are the top `n_pc` right singular vectors of the
    TF-IDF matrix `W`, the optimum of `||W - W V V.T||` over orthonormal `V`. The scores
    of a cell are `w V`.

    Parameters
    ----------
    n_pc : int
        Number of components. Defaults to 20.
    solver : str
        "arpack" for `scipy.sparse.linalg.svds` on the TF-IDF matrix in memory, on
        the CPU, which gives the exact optimum. "randomized" for the randomized SVD of
        Halko et al. in torch, by blocks of rows: 3 + 2 n_iter passes over the matrix,
        on `device`, from memory or from a backed AnnData. "auto" takes "arpack" for a
        matrix in memory on the CPU, else "randomized". Defaults to "auto".
    n_iter : int
        Power iterations of the randomized solver. Defaults to 8.
    oversample : int or None
        Columns of the randomized solver beyond n_pc. None takes n_pc. Defaults to None.
    drop_depth : bool
        Whether to drop every component whose scores follow the depth of the cells,
        the total of their counts as given, |Spearman rho| above
        DEPTH_CORRELATION_CUTOFF (`components_to_keep`), so fewer than n_pc components
        can stay. Defaults to False.
    binarize : bool
        Whether to read every value that is not zero as 1. Defaults to False.
    chunk_size : int or None
        Rows of a block. None takes DEFAULT_CHUNK_ROWS. Defaults to None.
    device : str, torch.device or None
        Device of the randomized solver and of transform. None selects "cuda" if it is
        available, else "cpu". Defaults to None.
    random_state : int or None
        Seed of the start of the solvers. Defaults to DEFAULT_RANDOM_STATE (42).

    Attributes
    ----------
    components_ : torch.Tensor
        `V`, features by components, ordered by singular value. The entry of largest
        absolute value of every component is positive.
    singular_values_ : torch.Tensor
        The singular value of every component.
    solver_ : str
        The solver that the fit used.
    idf_ : np.ndarray
        The idf of every feature, which transform uses for new cells.
    pivot_ : float or None
        The pivot of the normalization.
    depth_correlations_, n_dropped_components_
        With drop_depth, the Spearman rho of every component with the depth, and the
        number of components dropped.
    """

    def __init__(
        self,
        n_pc: int = 20,
        solver: Literal["auto", "arpack", "randomized"] = "auto",
        n_iter: int = 8,
        oversample: int | None = None,
        drop_depth: bool = False,
        binarize: bool = False,
        chunk_size: int | None = None,
        device: str | torch.device | None = None,
        random_state: int | None = DEFAULT_RANDOM_STATE,
    ) -> None:
        if solver not in SOLVERS:
            choices = ", ".join(repr(name) for name in SOLVERS)
            msg = f"solver={solver!r} is not valid. Use one of {choices}."
            raise ValueError(msg)
        if oversample is None:
            oversample = n_pc
        if n_pc < 1 or n_iter < 0 or oversample < 0:
            msg = (
                f"n_pc={n_pc}, n_iter={n_iter} and oversample={oversample} are not "
                f"valid. Use n_pc of 1 or more, and n_iter and oversample of 0 or more."
            )
            raise ValueError(msg)
        self.n_pc = n_pc
        self.solver = solver
        self.n_iter = n_iter
        self.oversample = oversample
        self.drop_depth = drop_depth
        self.binarize = binarize
        self.chunk_size = chunk_size
        self.device = device
        self.random_state = random_state

        self.components_: torch.Tensor | None = None
        self.singular_values_: torch.Tensor | None = None
        self.solver_: str | None = None
        self.idf_: np.ndarray | None = None
        self.pivot_: float | None = None
        self.depth_correlations_: np.ndarray | None = None
        self.n_dropped_components_: int | None = None

    def fit(self, X: torch.Tensor | np.ndarray | ad.AnnData | spmatrix) -> LSA:
        """Fits the components to `X`, cells in rows and features in columns."""
        self.fit_transform(X)
        return self

    def fit_transform(
        self, X: torch.Tensor | np.ndarray | ad.AnnData | spmatrix
    ) -> torch.Tensor:
        """Fits the components to `X` and returns the scores of its cells, on the CPU.

        The scores come from the solver, so they cost no pass after the fit.
        """
        source = self._source(X)
        n, p = source.shape
        if self.n_pc >= min(n, p):
            msg = (
                f"n_pc={self.n_pc} components need more cells and features than that, "
                f"but the input has {n} cells and {p} features. Use n_pc="
                f"{min(n, p) - 1} or fewer."
            )
            raise ValueError(msg)
        backed = isinstance(source, BackedRows)
        if self.solver == "arpack" and backed:
            msg = (
                "solver='arpack' needs the matrix in memory. Use solver='randomized' "
                "for a backed AnnData."
            )
            raise ValueError(msg)
        device = _resolve_device(self.device)
        solver = self.solver
        if solver == "auto":
            solver = "arpack" if not backed and device.type == "cpu" else "randomized"
        self.solver_ = solver
        tqdm.write(f"SOLVER: {solver}")

        self._fit_idf(source)
        weighted = self._weighted(source)
        if solver == "arpack":
            components, singular, scores = self._arpack(weighted)
        else:
            _announce_device(device)
            components, singular, scores = self._randomized(weighted, device)

        signs = largest_entry_signs(components)
        components, scores = components * signs, scores * signs

        self.depth_correlations_ = self.n_dropped_components_ = None
        if self.drop_depth:
            keep, self.depth_correlations_ = components_to_keep(scores, cell_depth(X))
            self.n_dropped_components_ = components.shape[1] - len(keep)
            kept = torch.from_numpy(keep)
            components, singular, scores = (
                components[:, kept],
                singular[kept],
                scores[:, kept],
            )
        self.components_ = components.contiguous()
        self.singular_values_ = singular
        return scores

    def transform(
        self, X: torch.Tensor | np.ndarray | ad.AnnData | spmatrix
    ) -> torch.Tensor:
        """The scores of the cells of `X`, with the idf of the fit."""
        if self.components_ is None:
            msg = "LSA is not fitted. Call fit() before transform()."
            raise RuntimeError(msg)
        assert self.idf_ is not None
        assert self.pivot_ is not None
        device = _resolve_device(self.device)
        weighted = _TfidfRows(self._source(X), self.idf_, self.pivot_)
        components = self.components_.to(device)
        scores = [
            (_torch_csr(block, device) @ components).cpu()
            for _, block in self._raw_blocks(weighted)
        ]
        return torch.cat(scores)

    def _source(
        self, X: torch.Tensor | np.ndarray | ad.AnnData | spmatrix
    ) -> SparseRows:
        """`X` as CSR rows: read from the file when backed, else held in memory."""
        if isinstance(X, ad.AnnData) and X.isbacked:
            return BackedRows(X.X, binarize=self.binarize)
        rows = _to_sparse(X)
        return SparseRows(binary(rows.matrix)) if self.binarize else rows

    def _raw_blocks(self, source: SparseRows) -> Iterator[tuple[slice, csr_matrix]]:
        """The rows of `source` by blocks of `chunk_size`, as float32 CSR."""
        n, chunk = source.shape[0], self.chunk_size or DEFAULT_CHUNK_ROWS
        for start in range(0, n, chunk):
            rows = slice(start, min(start + chunk, n))
            yield rows, csr_matrix(source._block(rows), dtype=np.float32)

    def _fit_idf(self, source: SparseRows) -> None:
        """Fits the idf in one pass over the counts."""
        n, p = source.shape
        holders = np.zeros(p)
        for _, block in self._raw_blocks(source):
            if bool((block.data < 0).any()):
                msg = "LSA takes counts, but the input has negative values."
                raise ValueError(msg)
            holders += np.asarray((block > 0).sum(axis=0), dtype=np.float64).ravel()
        self.idf_, self.pivot_ = idf(holders, n)

    def _weighted(self, source: SparseRows) -> SparseRows:
        """The TF-IDF of `source`: whole for a matrix in memory, else made as read."""
        assert self.idf_ is not None
        assert self.pivot_ is not None
        if isinstance(source, BackedRows):
            return _TfidfRows(source, self.idf_, self.pivot_)
        return SparseRows(tfidf(source.matrix, self.idf_, self.pivot_))

    def _arpack(
        self, weighted: SparseRows
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        u, s, vt = svds(weighted.matrix, k=self.n_pc, rng=self.random_state)
        order = np.argsort(s)[::-1]
        return (
            torch.tensor(vt[order].T, dtype=torch.float32),
            torch.tensor(s[order], dtype=torch.float32),
            torch.tensor(u[:, order] * s[order], dtype=torch.float32),
        )

    def _randomized(
        self, weighted: SparseRows, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The randomized SVD of Halko et al., with a QR step after every product.

        The scores are `W V`, one pass more than `Q U S` costs, so that transform gives
        the cells of the fit the scores that the fit gives them.
        """
        n, p = weighted.shape
        width = min(self.n_pc + self.oversample, n, p)
        generator = torch.Generator()
        if self.random_state is not None:
            generator.manual_seed(self.random_state)
        start = torch.randn(p, width, generator=generator).to(device)

        with tqdm(
            total=3 + 2 * self.n_iter, desc="LSA", unit="pass", dynamic_ncols=True
        ) as passes:

            def times(right: torch.Tensor) -> torch.Tensor:
                product = torch.empty(n, right.shape[1], device=device)
                for rows, block in self._raw_blocks(weighted):
                    product[rows] = _torch_csr(block, device) @ right
                passes.update(1)
                return product

            def transposed_times(left: torch.Tensor) -> torch.Tensor:
                product = torch.zeros(p, left.shape[1], device=device)
                for rows, block in self._raw_blocks(weighted):
                    product += _torch_csr(block.T.tocsr(), device) @ left[rows]
                passes.update(1)
                return product

            basis = torch.linalg.qr(times(start)).Q
            for _ in range(self.n_iter):
                features = torch.linalg.qr(transposed_times(basis)).Q
                basis = torch.linalg.qr(times(features)).Q
            projected = transposed_times(basis).T
            _, singular, right = torch.linalg.svd(projected, full_matrices=False)
            components = right[: self.n_pc].T.contiguous()
            scores = times(components)
        return components.cpu(), singular[: self.n_pc].cpu(), scores.cpu()
