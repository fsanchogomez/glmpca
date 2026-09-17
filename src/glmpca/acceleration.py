r"""Damped Anderson Acceleration with Restarts and Epsilon Monotonicity (DAAREM).

A reproduction of Henderson and Varadhan (2019), "Damped Anderson Acceleration With
Restarts and Monotonicity Control for Accelerating EM and EM-like Algorithms", Journal
of Computational and Graphical Statistics 28(4), 834-846, following their R package
`daarem` (`daarem_base_objfn.R` and `DampingFind.R`).

DAAREM accelerates a fixed-point iteration `x -> F(x)` that raises an objective at every
step. It fits a quasi-Newton model of the residual `f = F(x) - x` from the last few
iterates and jumps along it. Three parts keep the jump safe:

- **Damping.** The least-squares coefficients are shrunk by a ridge, chosen so that the
  damped coefficients keep a share `delta_k` of the norm of the undamped ones. The share
  starts small and grows as jumps succeed.
- **Restarts.** The memory is cleared every `order` iterations. A restart that lost
  ground damps harder.
- **Epsilon monotonicity.** A jump is used only when it leaves the objective no more
  than `mon_tol` below where the iteration started. Otherwise the plain fixed-point step
  is taken instead.

The accelerator holds no knowledge of the problem. The caller runs one step of `F`,
hands over the iterate and the residual, evaluates its own objective at the proposal,
and says whether it took it:

    proposal = accelerator.propose(theta, residual)
    if proposal is None:                      # the first call has no history yet
        ...
    elif candidate >= current - accelerator.mon_tol:
        accelerator.accept(candidate)
    else:
        accelerator.reject(plain_objective)
"""

from __future__ import annotations

import math

import torch

DAMPING_ITERATIONS = 10
"""Newton steps allowed in the search for the ridge, as in `DampingFind`."""


class DaaremAccelerator:
    r"""The DAAREM scheme over a flat parameter vector.

    Parameters
    ----------
    size : int
        Number of parameters, the length of the vectors handed to `propose`.

    order : int
        Number of iterates kept, `m` in the paper. The memory costs `2 * order * size`
        numbers. Defaults to 5, the default of the R package.

    alpha : float
        Base of the damping schedule. Defaults to 1.2.

    kappa : float
        Damping exponent at the start, so the first jumps keep a share
        `1 / sqrt(1 + alpha ** kappa)` of the undamped step. Defaults to 25.

    mon_tol : float
        How far below the objective of the current iterate a jump may leave it and still
        be taken. Defaults to 0.05, the value that fastglmpca uses.

    cycl_mon_tol : float
        How far the objective may fall between two restarts before the damping grows.
        Defaults to 0.

    device : str, torch.device or None
        Device of the memory. It should be the device of the parameters.

    dtype : torch.dtype
        Dtype of the memory. It should be the dtype of the parameters.

    Attributes
    ----------
    accepted : int
        Jumps taken so far.

    proposed : int
        Jumps offered so far.

    """

    def __init__(
        self,
        size: int,
        *,
        order: int = 5,
        alpha: float = 1.2,
        kappa: float = 25.0,
        mon_tol: float = 0.05,
        cycl_mon_tol: float = 0.0,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if order < 1:
            msg = f"order={order} is not valid. Use 1 or more."
            raise ValueError(msg)
        self.order = min(order, max(size // 2, 1))
        self.alpha = alpha
        self.kappa = kappa
        self.mon_tol = mon_tol
        self.cycl_mon_tol = cycl_mon_tol

        self.iterates = torch.zeros(size, self.order, device=device, dtype=dtype)
        self.residuals = torch.zeros(size, self.order, device=device, dtype=dtype)
        self.previous_iterate: torch.Tensor | None = None
        self.previous_residual: torch.Tensor | None = None

        self.column = 0
        self.shrink = 0
        self.best = -math.inf
        self.lambda_ridge = 1e5
        self.penalty = 0.0
        self.accepted = 0
        self.proposed = 0

    def propose(
        self, theta: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor | None:
        r"""The accelerated iterate, or None while there is no history to build it from.

        `theta` is the iterate this step started from and `residual` is `F(theta) -
        theta`. Both are read, not kept, so the caller may reuse its buffers.
        """
        if self.previous_iterate is None or self.previous_residual is None:
            self.previous_iterate = theta.clone()
            self.previous_residual = residual.clone()
            return None

        self.iterates[:, self.column] = theta - self.previous_iterate
        self.residuals[:, self.column] = residual - self.previous_residual
        self.previous_iterate.copy_(theta)
        self.previous_residual.copy_(residual)

        held = self.column + 1
        iterates = self.iterates[:, :held]
        residuals = self.residuals[:, :held]
        left, values, right = torch.linalg.svd(residuals, full_matrices=False)
        projected = left.T @ residual
        ridge = self._damping(projected * projected, values)
        gamma = right.T @ ((values * projected) / (values * values + ridge))

        self.proposed += 1
        return (theta - iterates @ gamma) + (residual - residuals @ gamma)

    def accept(self, objective: float) -> None:
        """Records that the caller took the jump, and lowers the damping."""
        self.accepted += 1
        self.shrink += 1
        self._advance(objective)

    def reject(self, objective: float) -> None:
        """Records that the caller kept the plain step, and holds the damping."""
        self._advance(objective)

    def _advance(self, objective: float) -> None:
        """Moves to the next column, and restarts the memory when it is full."""
        self.column += 1
        if self.column == self.order:
            self.column = 0
            if objective < self.best - self.cycl_mon_tol:
                self.shrink = max(self.shrink - self.order, -2 * int(self.kappa))
            self.best = objective

    def _damping(self, projected: torch.Tensor, values: torch.Tensor) -> float:
        r"""The ridge that leaves the damped coefficients at a share `delta_k`.

        `projected` holds the squared entries of `Uᵀ f` and `values` the singular values
        of the residual memory. This is `DampingFind` of the R package: a safeguarded
        Newton search on the norm of the ridge solution, in float64 because the two
        terms of the derivative cancel.
        """
        keep = values > 0
        values = values[keep].double()
        squares = values * values
        projected = projected[keep].double()

        power = self.kappa - self.shrink
        target = math.exp(-0.5 * math.log1p(self.alpha**power))
        undamped = float((projected / squares).sum().sqrt())
        goal = target * undamped
        if goal == 0.0:
            return self.lambda_ridge

        ridge = self.lambda_ridge - self.penalty / goal
        lower = (undamped * (undamped - goal)) / float(
            (projected / (squares * squares)).sum()
        )
        upper = math.sqrt(float((projected * squares).sum())) / goal
        stop_low = math.exp(-0.5 * math.log1p(self.alpha ** (power + 0.5)))
        stop_high = math.exp(-0.5 * math.log1p(self.alpha ** (power - 0.5)))

        norm = step = 0.0
        for _ in range(DAMPING_ITERATIONS):
            if ridge <= lower or ridge >= upper:
                ridge = max(1e-4 * upper, math.sqrt(max(lower, 0.0) * upper))
            damped = (values / (squares + ridge)).square()
            norm = math.sqrt(float((projected * damped).sum()))
            if norm == 0.0:
                break
            derivative = -float((projected * damped / (squares + ridge)).sum()) / norm
            step = (norm - goal) / derivative
            if stop_low * undamped <= norm <= stop_high * undamped:
                break
            upper = upper if norm >= goal else ridge
            lower = max(lower, ridge - step)
            ridge = ridge - (norm * step) / goal

        self.lambda_ridge = ridge
        self.penalty = norm * step
        return ridge
