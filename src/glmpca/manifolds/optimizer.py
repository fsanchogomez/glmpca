from __future__ import annotations

from typing import TYPE_CHECKING, overload

import torch
import torch.optim

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.optim.optimizer import ParamsT


class RiemannianAdagrad(torch.optim.Optimizer):
    """Adagrad on a manifold.

    Each step converts the Euclidean gradient to the Riemannian one, accumulates the
    squares as Adagrad does, projects the scaled direction on the tangent space, and
    then retracts back to the manifold.
    """

    def __init__(self, params: ParamsT, lr: float = 1e-2, eps: float = 1e-10) -> None:
        super().__init__(params, {"lr": lr, "eps": eps})

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for point in group["params"]:
                if point.grad is None:
                    continue
                manifold = point.manifold
                state = self.state[point]
                if not state:
                    state["sum"] = torch.zeros_like(point)
                rgrad = manifold.egrad2rgrad(point, point.grad)
                state["sum"].add_(rgrad.square())
                std = state["sum"].sqrt().add_(group["eps"])
                direction = manifold.proju(point, rgrad / std)
                point.copy_(manifold.retr(point, -group["lr"] * direction))
        return loss
