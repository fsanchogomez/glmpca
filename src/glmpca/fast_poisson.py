r"""Poisson GLM-PCA by Alternating Poisson Regression (APR).

A reproduction of the method of Weine, Carbonetto and Stephens (2024),
"Accelerated dimensionality reduction of single-cell RNA sequencing data with
fastglmpca", Bioinformatics 40(8), btae494.

The model is the Poisson GLM-PCA of Townes et al. (2019), fitted directly on the counts
rather than on saturated parameters (the rest of this package):

    Y[i, j] ~ Poisson(exp(H[i, j])),    H = U Vᵀ

with cells in the rows of `Y` and features in its columns. Two structural columns hold
the parts that are not of interest, as in `fastglmpca`:

    H[i, j] = size_factor[i] + intercept[j] + sum_k U[i, k] V[j, k]

The size factor of a cell and the intercept of a feature are columns of `U` and `V`
whose partner column is fixed to one.

The fit alternates two blocks (Algorithm 1 of the paper):

1. With `V` fixed, every row of `U` is the maximum likelihood estimate of a Poisson GLM
   with design `V`. The rows are independent.
2. With `U` fixed, every row of `V` is the same problem with design `U`.

Each block runs one pass of cyclic coordinate descent with a 1-D Newton step per
coordinate, as in the paper. For coordinate `k` of row `i`, with `rate = exp(H)`:

    gradient  = sum_j rate[i, j] * D[j, k] - sum_j Y[i, j] * D[j, k]
    curvature = sum_j rate[i, j] * D[j, k]^2
    step      = gradient / curvature

Every row shares the same design, so a coordinate is updated for all rows at once, which
is what makes this practical in torch.

After the fit, the "PCA-like" decomposition comes from the SVD of the free part of
`U Vᵀ`, computed through the QR factors so that no `n` by `p` matrix is formed.
"""

from __future__ import annotations

import warnings

import torch
from tqdm.auto import tqdm

from .ExponentialFamily import Poisson
from .GLMPCA import _resolve_device, _to_tensor

MAX_LOG_RATE = 30.0
"""Upper bound of `H`, so that `exp(H)` stays finite in float32."""

MAX_STEP = 1.0
"""Largest Newton step of one coordinate, in log space. It keeps a flat curvature from
sending a coordinate far away, where the reference implementation uses a line search."""


class FastPoissonPCA:
    r"""Poisson GLM-PCA fitted by Alternating Poisson Regression.

    Parameters
    ----------
    n_pc : int
        Number of components, `K` in the paper.

    max_iter : int
        Maximum number of alternating passes over `U` and `V`. Defaults to 100.

    tol : float
        Stop when one pass improves the log-likelihood by less than `tol` times the
        absolute log-likelihood. Defaults to 1e-6.

    device : str, torch.device or None
        Device to fit on. None selects "cuda" if it is available, else "cpu". The "mps"
        device is not supported. Fitted attributes are stored on the CPU.

    Attributes
    ----------
    loadings_ : torch.Tensor
        Orthonormal feature loadings, shape `(p, K)`.

    scores_ : torch.Tensor
        Cell coordinates, shape `(n, K)`: the left factors times the singular values.

    singular_values_ : torch.Tensor
        The `K` singular values of the fitted `U Vᵀ`.

    intercept_ : torch.Tensor
        Intercept of every feature, shape `(p,)`.

    size_factors_ : torch.Tensor
        Size factor of every cell, shape `(n,)`.

    log_likelihoods_ : list[float]
        Poisson log-likelihood after each pass, with the `-log(y!)` term, so that it is
        the real log-likelihood and not the optimisation objective alone.

    """

    def __init__(
        self,
        n_pc: int,
        max_iter: int = 100,
        tol: float = 1e-6,
        device: str | torch.device | None = None,
    ) -> None:
        self.n_pc = n_pc
        self.max_iter = max_iter
        self.tol = tol
        self.device = device

        self.loadings_: torch.Tensor | None = None
        self.scores_: torch.Tensor | None = None
        self.singular_values_: torch.Tensor | None = None
        self.intercept_: torch.Tensor | None = None
        self.size_factors_: torch.Tensor | None = None
        self.log_likelihoods_: list[float] = []

    def fit(self, X: object) -> bool:
        r"""Fits the model to counts with cells in rows and features in columns."""
        device = _resolve_device(self.device)
        Y = _to_tensor(X).to(device)  # ty: ignore[invalid-argument-type]
        if Y.shape[0] < 2:
            msg = (
                f"A fit needs at least 2 rows (cells), but the input has {Y.shape[0]}. "
                f"Check that the data has cells in rows and features in columns."
            )
            raise ValueError(msg)
        if torch.any(Y < 0):
            msg = "Poisson GLM-PCA needs counts, but the input has negative values."
            raise ValueError(msg)

        U, V = self._initialize(Y)
        # Column 0 is the size factor, free in U and fixed in V. Column 1 is the
        # intercept of a feature, fixed in U and free in V.
        free_in_u = [0, *range(2, self.n_pc + 2)]
        free_in_v = [1, *range(2, self.n_pc + 2)]

        rate = torch.exp((U @ V.T).clip(max=MAX_LOG_RATE))
        buffer = torch.empty_like(rate)
        # log h(y) = -log(y!) does not move with U or V, so one pass is enough. It is
        # what separates the objective of the fit from a real log-likelihood.
        log_base_measure = float(Poisson().log_base_measure(Y).sum())
        self.log_likelihoods_ = []
        previous = -torch.inf
        with tqdm(total=self.max_iter, unit="pass", dynamic_ncols=True) as passes:
            for _ in range(self.max_iter):
                _descend(Y, U, V, rate, free_in_u, buffer, over_rows=True)
                _descend(Y.T, V, U, rate, free_in_v, buffer, over_rows=False)

                log_likelihood = (
                    float((U * (Y @ V)).sum() - rate.sum()) + log_base_measure
                )
                self.log_likelihoods_.append(log_likelihood)
                # The postfix waits for the update, so the bar is drawn one time a pass.
                passes.set_postfix(log_lik=f"{log_likelihood:.4E}", refresh=False)
                passes.update(1)
                if abs(log_likelihood - previous) <= self.tol * abs(log_likelihood):
                    break
                previous = log_likelihood
            else:
                msg = (
                    f"The log-likelihood still moved after {self.max_iter} passes. "
                    f"Raise max_iter for a tighter fit."
                )
                warnings.warn(msg, UserWarning, stacklevel=2)

        self._store(U, V)
        return True

    def transform(self, X: object) -> torch.Tensor:
        r"""Coordinates of new cells, with the fitted loadings held fixed.

        Every new cell is one Poisson GLM with the fitted loadings as its design, which
        is the same step that `fit` runs over the rows of `U`.
        """
        if self.loadings_ is None or self.intercept_ is None:
            msg = "FastPoissonPCA is not fitted. Call fit() before transform()."
            raise RuntimeError(msg)

        Y = _to_tensor(X)  # ty: ignore[invalid-argument-type]
        loadings = self.loadings_.to(Y.device)
        V = torch.cat(
            [
                torch.ones(loadings.shape[0], 1, device=Y.device),
                self.intercept_.to(Y.device).unsqueeze(1),
                loadings,
            ],
            dim=1,
        )
        U = torch.zeros(Y.shape[0], V.shape[1], device=Y.device)
        depth = Y.sum(dim=1).clip(min=1.0)
        U[:, 0] = torch.log(depth / depth.mean())
        U[:, 1] = 1.0

        rate = torch.exp((U @ V.T).clip(max=MAX_LOG_RATE))
        buffer = torch.empty_like(rate)
        free = [0, *range(2, V.shape[1])]
        previous = -torch.inf
        for _ in range(self.max_iter):
            _descend(Y, U, V, rate, free, buffer, over_rows=True)
            # The same test as fit, so that transform stops as soon as it has landed.
            likelihood = float((U * (Y @ V)).sum() - rate.sum())
            if abs(likelihood - previous) <= self.tol * abs(likelihood):
                break
            previous = likelihood
        return U[:, 2:]

    def _initialize(self, Y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Small random factors, as in fastglmpca, with the structural columns set."""
        n, p = Y.shape
        U = torch.randn(n, self.n_pc + 2, device=Y.device) * 1e-4
        V = torch.randn(p, self.n_pc + 2, device=Y.device) * 1e-4

        depth = Y.sum(dim=1).clip(min=1.0)
        U[:, 0] = torch.log(depth / depth.mean())
        U[:, 1] = 1.0
        V[:, 0] = 1.0
        V[:, 1] = torch.log(Y.mean(dim=0).clip(min=1e-4))
        return U, V

    def _store(self, U: torch.Tensor, V: torch.Tensor) -> None:
        """Keeps the structural columns, and the SVD of the free part."""
        self.size_factors_ = U[:, 0].detach().cpu()
        self.intercept_ = V[:, 1].detach().cpu()

        # U_free V_freeᵀ = Q_u (R_u R_vᵀ) Q_vᵀ, so the SVD of the K x K middle matrix
        # gives the decomposition without ever forming an n x p matrix.
        q_u, r_u = torch.linalg.qr(U[:, 2:])
        q_v, r_v = torch.linalg.qr(V[:, 2:])
        left, singular_values, right = torch.linalg.svd(r_u @ r_v.T)

        self.scores_ = (q_u @ left * singular_values).detach().cpu()
        self.loadings_ = (q_v @ right.T).detach().cpu()
        self.singular_values_ = singular_values.detach().cpu()


def _descend(
    Y: torch.Tensor,
    W: torch.Tensor,
    D: torch.Tensor,
    rate: torch.Tensor,
    free: list[int],
    buffer: torch.Tensor,
    *,
    over_rows: bool,
) -> None:
    r"""One cyclic coordinate descent pass over the rows of `W`, in place.

    `Y` holds the counts with the rows of `W` in its rows, `D` is the design (the other
    factor matrix), and `rate` is `exp(U Vᵀ)` in its `(cells, features)` layout, updated
    as the coordinates move. `over_rows` says whether a row of `W` is a row of `rate`
    (the `U` block) or one of its columns (the `V` block); the maths is the same, and
    keeping `rate` in one layout keeps every pass over it contiguous.

    `buffer` is an `n` by `p` workspace, so that a pass allocates nothing.
    """
    statistics = Y @ D
    for column in free:
        design = D[:, column]
        if over_rows:
            gradient = rate @ design - statistics[:, column]
            curvature = rate @ design.square()
        else:
            gradient = design @ rate - statistics[:, column]
            curvature = design.square() @ rate
        step = (gradient / curvature.clip(min=1e-12)).clip(-MAX_STEP, MAX_STEP)

        W[:, column] -= step
        # exp of the outer product, in place, so the pass costs one read and one write
        # of `rate` instead of three of each.
        if over_rows:
            torch.outer(-step, design, out=buffer)
        else:
            torch.outer(-design, step, out=buffer)
        buffer.clamp_(max=MAX_LOG_RATE).exp_()
        rate.mul_(buffer)
