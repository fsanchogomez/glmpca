from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class Euclidean:
    """Unconstrained parameters, such as the intercept.

    The gradient needs no change, every direction is a tangent direction, and a step
    is an addition.
    """

    name = "Euclidean"

    def proju(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return u

    egrad2rgrad = proju

    def retr(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return x + u
