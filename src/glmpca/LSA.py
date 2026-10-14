"""Latent semantic analysis: the truncated SVD of a TF-IDF matrix, solved directly.

The LSI families of GLMPCA reach the same subspace by gradient steps. `LSA` computes it
with an SVD solver instead: ARPACK (`scipy.sparse.linalg.svds`) on a matrix in memory,
or a randomized SVD in torch that reads the matrix by blocks of rows, on the CPU or the
GPU, from memory or from a backed AnnData. The TF-IDF of Signac or of gensim (`tfidf`)
is computed on the stored entries of every CSR block, so nothing is made dense.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

import anndata as ad
import numpy as np
import torch
from scipy.sparse import csr_matrix, issparse, spmatrix
from scipy.sparse.linalg import svds
from scipy.stats import spearmanr
from tqdm.auto import tqdm

from .GLMPCA import (
    DEFAULT_RANDOM_STATE,
    _announce_device,
    _resolve_device,
    _to_sparse,
)
from .manifolds import largest_entry_signs
from .sparse import BackedRows, SparseRows, binary

if TYPE_CHECKING:
    from collections.abc import Iterator

DEFAULT_CHUNK_ROWS = 16384
FLAVORS = ("gensim", "signac")
SOLVERS = ("auto", "arpack", "randomized")
DEPTH_CORRELATION_CUTOFF = 0.75
"""|Spearman rho| with the depth above which drop_depth drops a component."""
# Scale factor of Signac's `RunTFIDF(method = 1)`, applied before `log1p`.
TFIDF_SCALE = 1e4
# Slope of gensim's pivoted unique normalization, `smartirs="lfu"`.
GENSIM_SLOPE = 0.25


def idf(
    flavor: str, totals: np.ndarray, holders: np.ndarray, n_cells: int
) -> tuple[np.ndarray, float | None]:
    """The idf of every feature, and for "gensim" the pivot of the normalization.

    `totals` are the total counts of the features and `holders` the cells that hold
    them. "signac" divides the number of cells by the total counts, as Signac's
    `rowSums` does, with a total of at least 1. "gensim" takes `log2(cells / holders)`,
    zero for a feature that no cell holds, and the pivot is the mean number of features
    that a cell holds.
    """
    if flavor == "signac":
        return n_cells / np.clip(totals, 1.0, None), None
    feature_idf = np.where(
        holders > 0, np.log2(n_cells / np.clip(holders, 1.0, None)), 0.0
    )
    return feature_idf, float(holders.sum()) / n_cells


def tfidf(
    block: csr_matrix, flavor: str, feature_idf: np.ndarray, pivot: float | None
) -> csr_matrix:
    r"""The TF-IDF of the CSR `block`, computed on its stored entries.

    "signac" is Signac's `RunTFIDF(method = 1)`:

        log1p(count / depth of the cell · idf · TFIDF_SCALE)

    "gensim" is gensim's `TfidfModel(normalize=True, smartirs="lfu")`, for a count
    above zero:

        (1 + log2(count)) · idf / ((1 - GENSIM_SLOPE) · pivot + GENSIM_SLOPE · kept)

    with `kept` the features of the cell whose weight is above zero. Both keep a zero at
    zero, so the result has the entries of `block`.
    """
    counts = np.asarray(block.data, dtype=np.float64)
    rows = np.repeat(np.arange(block.shape[0]), np.diff(block.indptr))
    if flavor == "signac":
        depth = np.bincount(rows, weights=counts, minlength=block.shape[0])
        values = np.log1p(
            counts
            / np.clip(depth, 1.0, None)[rows]
            * feature_idf[block.indices]
            * TFIDF_SCALE
        )
    else:
        assert pivot is not None
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


class LSA:
    r"""Latent semantic analysis of a cell-by-feature count matrix.

    The counts are turned into the TF-IDF of Signac's `RunTFIDF(method = 1)` or of
    gensim's `TfidfModel(smartirs="lfu")` (`tfidf`), and the components are the top
    `n_pc` right singular vectors of the TF-IDF matrix `W`, the optimum of
    `||W - W V V.T||` over orthonormal `V`. The scores of a cell are `w V`,
    standardized for "signac" as `RunSVD(scale.embeddings = TRUE)` does.

    Parameters
    ----------
    n_pc : int
        Number of components. Defaults to 20.
    flavor : str
        "gensim" or "signac", the TF-IDF. Defaults to "signac".
    solver : str
        "arpack" for `scipy.sparse.linalg.svds` on the TF-IDF matrix in memory, on
        the CPU, which gives the exact optimum. "randomized" for the randomized SVD of
        Halko et al. in torch, by blocks of rows: 3 + 2 n_iter passes over the matrix,
        on `device`, from memory or from a backed AnnData. "auto" takes "arpack" for a
        matrix in memory on the CPU, else "randomized". Defaults to "auto".
    n_iter : int
        Power iterations of the randomized solver. Defaults to 4.
    oversample : int
        Columns of the randomized solver beyond n_pc. Defaults to 10.
    drop_depth : bool
        Whether to drop every component whose scores follow the depth of the cells,
        |Spearman rho| above DEPTH_CORRELATION_CUTOFF, so fewer than n_pc components
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
        For "gensim", the pivot of the normalization.
    depth_correlations_, n_dropped_components_
        With drop_depth, the Spearman rho of every component with the depth, and the
        number of components dropped.
    score_mean_, score_sd_
        For "signac", the mean and the standard deviation of every score.
    """

    def __init__(
        self,
        n_pc: int = 20,
        flavor: Literal["gensim", "signac"] = "gensim",
        solver: Literal["auto", "arpack", "randomized"] = "auto",
        n_iter: int = 4,
        oversample: int = 10,
        drop_depth: bool = False,
        binarize: bool = False,
        chunk_size: int | None = None,
        device: str | torch.device | None = None,
        random_state: int | None = DEFAULT_RANDOM_STATE,
    ) -> None:
        if flavor not in FLAVORS:
            msg = f"flavor={flavor!r} is not valid. Use 'gensim' or 'signac'."
            raise ValueError(msg)
        if solver not in SOLVERS:
            choices = ", ".join(repr(name) for name in SOLVERS)
            msg = f"solver={solver!r} is not valid. Use one of {choices}."
            raise ValueError(msg)
        if n_pc < 1 or n_iter < 0 or oversample < 0:
            msg = (
                f"n_pc={n_pc}, n_iter={n_iter} and oversample={oversample} are not "
                f"valid. Use n_pc of 1 or more, and n_iter and oversample of 0 or more."
            )
            raise ValueError(msg)
        self.n_pc = n_pc
        self.flavor = flavor
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
        self.score_mean_: torch.Tensor | None = None
        self.score_sd_: torch.Tensor | None = None

    def fit(self, X: torch.Tensor | np.ndarray | ad.AnnData | spmatrix) -> LSA:
        """Fits the components to `X`, cells in rows and features in columns."""
        self.fit_transform(X)
        return self

    def fit_transform(self, X: torch.Tensor | np.ndarray | ad.AnnData) -> torch.Tensor:
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

        depth = self._fit_idf(source)
        if solver == "arpack":
            components, singular, scores = self._arpack(source)
        else:
            _announce_device(device)
            components, singular, scores = self._randomized(source, device)

        signs = largest_entry_signs(components)
        components, scores = components * signs, scores * signs

        self.depth_correlations_ = self.n_dropped_components_ = None
        if self.drop_depth:
            correlations = np.nan_to_num([
                spearmanr(score, depth).statistic for score in scores.double().T
            ])
            dropped = np.abs(correlations) > DEPTH_CORRELATION_CUTOFF
            kept = torch.from_numpy(np.flatnonzero(~dropped))
            self.depth_correlations_ = correlations
            self.n_dropped_components_ = int(dropped.sum())
            tqdm.write(
                f"DEPTH: {self.n_dropped_components_} of {components.shape[1]} "
                f"components dropped, |Spearman rho| with cell depth above "
                f"{DEPTH_CORRELATION_CUTOFF}"
            )
            components, singular, scores = (
                components[:, kept],
                singular[kept],
                scores[:, kept],
            )
        self.components_ = components.contiguous()
        self.singular_values_ = singular

        self.score_mean_ = self.score_sd_ = None
        if self.flavor == "signac":
            self.score_mean_ = scores.mean(dim=0)
            self.score_sd_ = scores.std(dim=0).clip(min=1e-12)
            scores = (scores - self.score_mean_) / self.score_sd_
        return scores

    def transform(self, X: torch.Tensor | np.ndarray | ad.AnnData) -> torch.Tensor:
        """The scores of the cells of `X`, with the idf of the fit."""
        if self.components_ is None:
            msg = "LSA is not fitted. Call fit() before transform()."
            raise RuntimeError(msg)
        device = _resolve_device(self.device)
        source = self._source(X)
        components = self.components_.to(device)
        scores = [
            (_torch_csr(block, device) @ components).cpu()
            for _, block in self._tfidf_blocks(source, None)
        ]
        scores_all = torch.cat(scores)
        if self.score_mean_ is not None and self.score_sd_ is not None:
            scores_all = (scores_all - self.score_mean_) / self.score_sd_
        return scores_all

    def _source(self, X: torch.Tensor | np.ndarray | ad.AnnData) -> SparseRows:
        """`X` as CSR rows: read from the file when backed, else held in memory."""
        if isinstance(X, ad.AnnData) and X.isbacked:
            return BackedRows(X.X, binarize=self.binarize)
        rows = (
            SparseRows(csr_matrix(X, dtype=np.float32))
            if issparse(X)
            else _to_sparse(X)
        )
        return SparseRows(binary(rows.matrix)) if self.binarize else rows

    def _chunk(self) -> int:
        return self.chunk_size or DEFAULT_CHUNK_ROWS

    def _raw_blocks(self, source: SparseRows) -> Iterator[tuple[slice, csr_matrix]]:
        """The rows of `source` by blocks, as they are stored, as float32 CSR."""
        n, chunk = source.shape[0], self._chunk()
        for start in range(0, n, chunk):
            rows = slice(start, min(start + chunk, n))
            block = source._block(rows)
            yield rows, csr_matrix(block if issparse(block) else np.asarray(block))

    def _tfidf(self, block: csr_matrix) -> csr_matrix:
        """`tfidf` of `block` with the idf of the fit."""
        assert self.idf_ is not None
        return tfidf(block, self.flavor, self.idf_, self.pivot_)

    def _tfidf_blocks(
        self, source: SparseRows, tfidf_matrix: csr_matrix | None
    ) -> Iterator[tuple[slice, csr_matrix]]:
        """The TF-IDF by blocks of rows: slices of `tfidf_matrix`, or made as read."""
        if tfidf_matrix is not None:
            n, chunk = tfidf_matrix.shape[0], self._chunk()
            for start in range(0, n, chunk):
                rows = slice(start, min(start + chunk, n))
                yield rows, tfidf_matrix[rows]
            return
        for rows, block in self._raw_blocks(source):
            yield rows, self._tfidf(block)

    def _fit_idf(self, source: SparseRows) -> np.ndarray:
        """Fits the idf in one pass over the counts, and returns the depth of a cell."""
        n, p = source.shape
        totals = np.zeros(p)
        holders = np.zeros(p)
        depth = np.empty(n)
        for rows, block in self._raw_blocks(source):
            if bool((block.data < 0).any()):
                msg = "LSA takes counts, but the input has negative values."
                raise ValueError(msg)
            totals += np.asarray(block.sum(axis=0), dtype=np.float64).ravel()
            holders += np.asarray((block > 0).sum(axis=0), dtype=np.float64).ravel()
            depth[rows] = np.asarray(block.sum(axis=1), dtype=np.float64).ravel()
        self.idf_, self.pivot_ = idf(self.flavor, totals, holders, n)
        return depth

    def _tfidf_in_memory(self, source: SparseRows) -> csr_matrix | None:
        """The whole TF-IDF matrix for a matrix in memory, else None."""
        if isinstance(source, BackedRows):
            return None
        return self._tfidf(source.matrix)

    def _arpack(
        self, source: SparseRows
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tfidf_matrix = self._tfidf_in_memory(source)
        assert tfidf_matrix is not None
        u, s, vt = svds(tfidf_matrix, k=self.n_pc, rng=self.random_state)
        order = np.argsort(s)[::-1]
        components = torch.from_numpy(np.ascontiguousarray(vt[order].T))
        singular = torch.from_numpy(np.ascontiguousarray(s[order]))
        scores = torch.from_numpy(np.ascontiguousarray(u[:, order] * s[order]))
        return components.float(), singular.float(), scores.float()

    def _randomized(
        self, source: SparseRows, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The randomized SVD of Halko et al., with a QR step after every product.

        The scores are `W V`, one pass more than `Q U S` costs, so that transform gives
        the cells of the fit the scores that the fit gives them.
        """
        n, p = source.shape
        width = min(self.n_pc + self.oversample, n, p)
        tfidf_matrix = self._tfidf_in_memory(source)
        generator = torch.Generator()
        if self.random_state is not None:
            generator.manual_seed(self.random_state)
        start = torch.randn(p, width, generator=generator).to(device)

        with tqdm(
            total=3 + 2 * self.n_iter, desc="LSA", unit="pass", dynamic_ncols=True
        ) as passes:

            def times(right: torch.Tensor) -> torch.Tensor:
                product = torch.empty(n, right.shape[1], device=device)
                for rows, block in self._tfidf_blocks(source, tfidf_matrix):
                    product[rows] = _torch_csr(block, device) @ right
                passes.update(1)
                return product

            def transposed_times(left: torch.Tensor) -> torch.Tensor:
                product = torch.zeros(p, left.shape[1], device=device)
                for rows, block in self._tfidf_blocks(source, tfidf_matrix):
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
