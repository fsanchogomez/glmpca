from __future__ import annotations

import torch


class EuclideanStiefel:
    r"""The Stiefel manifold `{ X : XᵀX = I }` with the Euclidean inner product.

    The loadings `V` of GLM-PCA live here, which keeps `VᵀV = I` at every step
    without a penalty term.
    """

    name = "Stiefel(euclidean)"

    def proju(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Projects `u` on the tangent space at `x`: `u - x sym(x.T u)`."""
        inner = x.transpose(-1, -2) @ u
        return u - x @ (0.5 * (inner.transpose(-1, -2) + inner))

    egrad2rgrad = proju

    def retr(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Moves from `x` along `u` and returns to the manifold with a QR step.

        `torch.linalg.qr` does not fix the signs, so a negative value on the diagonal
        of `R` flips the related column of `Q`. The correction keeps the retraction
        continuous at `u = 0`. A zero on the diagonal counts as positive, which is
        what `sign(sign(d) + 0.5)` gives.
        """
        q, r = torch.linalg.qr(x + u)
        unflip = torch.diagonal(r, 0, -1, -2).sign().add(0.5).sign()
        return q * unflip[..., None, :]

    def random(
        self,
        n: int,
        k: int,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """A random point of `St(n, k)`: the QR factor of a normal sample."""
        if k > n:
            msg = f"The Stiefel manifold needs k <= n, but k={k} and n={n}."
            raise ValueError(msg)
        return torch.linalg.qr(torch.randn(n, k, dtype=dtype, device=device))[0]
