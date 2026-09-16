from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .euclidean import Euclidean

if TYPE_CHECKING:
    from typing_extensions import Self

    from .stiefel import EuclideanStiefel


class ManifoldParameter(torch.nn.Parameter):
    """A parameter that knows the manifold it lives on.

    `RiemannianAdagrad` reads `manifold` to convert the gradient, to project the
    direction and to take the step. Without a manifold, the parameter is Euclidean,
    and the optimiser then behaves like `torch.optim.Adagrad`.
    """

    manifold: Euclidean | EuclideanStiefel

    def __new__(
        cls,
        data: torch.Tensor | None = None,
        manifold: Euclidean | EuclideanStiefel | None = None,
        requires_grad: bool = True,
    ) -> Self:
        if data is None:
            data = torch.empty(0)
        return super().__new__(cls, data, requires_grad)

    def __init__(
        self,
        data: torch.Tensor | None = None,
        manifold: Euclidean | EuclideanStiefel | None = None,
        requires_grad: bool = True,
    ) -> None:
        self.manifold = manifold if manifold is not None else Euclidean()

    def __repr__(self, *, tensor_contents: object = None) -> str:
        return (
            f"Parameter on {self.manifold.name} containing:\n"
            + torch.Tensor.__repr__(self)
        )
