r"""The Grassmann manifold `Gr(p, k)`, and the canonical basis of one of its points.

The cost of GLM-PCA sees the loadings only through the projector `V Vᵀ`:

    Θ̂ = (Θ - offset) V Vᵀ + offset

so any rotation `V Q` of the loadings gives the same fit. The problem therefore lives
on the Grassmann manifold of the `k`-dimensional subspaces of `R^p`, and the Stiefel
manifold of orthonormal bases holds it with a redundant rotation on top.

Two consequences follow, one for each part of this module:

- **The optimisation.** For a cost of the span alone, the Stiefel gradient is
  already horizontal, so plain gradient steps are the same on both manifolds. They
  part when an optimiser rescales the gradient element by element, as Adagrad and Adam
  do: the Stiefel projection then keeps a vertical part, a pure rotation of the basis
  that the cost cannot see, which the Grassmann projection drops.
- **The output.** Nothing in the cost fixes the basis within the span, so a fit ends
  on an arbitrary one: its components are neither ordered by variance nor reproducible
  between runs. `canonical_basis` picks one representative of the span, the basis
  whose scores are uncorrelated and ordered by variance, which is what PCA reports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .stiefel import EuclideanStiefel

if TYPE_CHECKING:
    from collections.abc import Iterable


class Grassmann(EuclideanStiefel):
    r"""The subspaces of dimension `k` in `R^p`, each held by an orthonormal basis.

    A point is stored as a `p x k` matrix with orthonormal columns, as on the Stiefel
    manifold, but two bases of the same span are the same point. The tangent space
    at `V` is the horizontal space `{u : Vᵀ u = 0}`, so the projection is
    `(I - V Vᵀ) u`. It drops the vertical part `V skew(Vᵀ u)` that the Stiefel
    projection keeps.

    The retraction and the random point are those of the Stiefel manifold: the QR step
    returns a basis of the moved span, and that is all a point here needs.
    """

    name = "Grassmann"

    def proju(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Projects `u` on the horizontal space at `x`: `u - x (xᵀ u)`."""
        return u - x @ (x.transpose(-1, -2) @ u)

    egrad2rgrad = proju


def canonical_basis(
    loadings: torch.Tensor, centered_chunks: Iterable[torch.Tensor]
) -> torch.Tensor:
    r"""The basis of `span(loadings)` whose scores are uncorrelated, by variance.

    `centered_chunks` yields the rows of `Θ - offset` in blocks, so that the `n x p`
    matrix is never formed. The scores `(Θ - offset) V` of every block add to their
    `k x k` covariance, whose eigenvectors `B` rotate the basis: the scores of `V B` are
    then
    uncorrelated, and the columns are sorted by decreasing variance. The scores and
    their sums run in float64: two components of nearly equal variance make the
    eigenvectors sensitive, and float32 would let the chunk size move the basis.

    The sign of an eigenvector is arbitrary, so every column is turned to make its
    entry of largest magnitude positive. With that, the same span gives the same basis
    whatever basis of it came in, which is what makes a component reproducible between
    runs.

    The span, and with it `V Vᵀ` and every fitted value, does not change.
    """
    size = loadings.shape[1]
    gram = torch.zeros(size, size, dtype=torch.float64)
    total = torch.zeros(size, dtype=torch.float64)
    rows = 0
    for chunk in centered_chunks:
        scores = (chunk.double() @ loadings.double()).cpu()
        gram += scores.T @ scores
        total += scores.sum(dim=0)
        rows += scores.shape[0]
    mean = total / max(rows, 1)
    covariance = gram / max(rows, 1) - torch.outer(mean, mean)

    variances, rotation = torch.linalg.eigh(covariance)
    order = torch.argsort(variances, descending=True)
    rotation = rotation[:, order].to(dtype=loadings.dtype, device=loadings.device)
    basis = loadings @ rotation

    largest = basis.abs().argmax(dim=0)
    signs = torch.sign(basis[largest, torch.arange(size, device=basis.device)])
    signs[signs == 0] = 1
    return basis * signs
