from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, overload

import torch
import torch.optim

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.optim.optimizer import ParamsT

    from .parameter import ManifoldParameter


class RiemannianAdagrad(torch.optim.Optimizer):
    """Adagrad on a manifold.

    Each step converts the Euclidean gradient to the Riemannian one, accumulates
    the squares as Adagrad does, projects the scaled direction on the tangent space,
    and then retracts back to the manifold.
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


class RiemannianAdam(torch.optim.Optimizer):
    """Adam on a manifold.

    Both moments follow the Riemannian gradient. After a step, the first moment
    is carried to the tangent space of the new point, which is a projection on
    `EuclideanStiefel`.

    The second moment is element-wise, as in `torch.optim.Adam` and in
    `RiemannianAdagrad`.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        super().__init__(params, {"lr": lr, "betas": betas, "eps": eps})

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
            first_decay, second_decay = group["betas"]
            for point in group["params"]:
                if point.grad is None:
                    continue
                manifold = point.manifold
                state = self.state[point]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(point)
                    state["exp_avg_sq"] = torch.zeros_like(point)
                state["step"] += 1

                rgrad = manifold.egrad2rgrad(point, point.grad)
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(first_decay).add_(rgrad, alpha=1 - first_decay)
                exp_avg_sq.mul_(second_decay).addcmul_(
                    rgrad, rgrad, value=1 - second_decay
                )

                first_bias = 1 - first_decay ** state["step"]
                second_bias = 1 - second_decay ** state["step"]
                std = (exp_avg_sq / second_bias).sqrt().add_(group["eps"])
                direction = manifold.proju(point, exp_avg / first_bias / std)

                moved = manifold.retr(point, -group["lr"] * direction)
                # Carry the momentum to the tangent space of the new point.
                state["exp_avg"] = manifold.proju(moved, exp_avg)
                point.copy_(moved)
        return loss


def _interpolate(
    low: tuple[float, float, float], high: tuple[float, float, float]
) -> float:
    """The next trial step between two brackets, by a safeguarded quadratic fit.

    The quadratic through the value and the slope at `low` and the value at `high` has
    its least point at the step below. Bisection is the fallback, and the result is kept
    away from both ends, as Nocedal and Wright advise: a pure bisection from a bracket
    that is orders of magnitude too wide cannot reach the right scale in few trials.
    """
    span = high[0] - low[0]
    curvature = high[1] - low[1] - low[2] * span
    step = low[0] + 0.5 * span
    if math.isfinite(curvature) and curvature > 0:
        candidate = low[0] - low[2] * span * span / (2 * curvature)
        if math.isfinite(candidate):
            step = candidate
    near, far = low[0] + 0.1 * span, low[0] + 0.9 * span
    return min(max(step, min(near, far)), max(near, far))


def _as_float(value: object) -> float:
    """The number a closure returned, detached when it is a tensor with a graph."""
    return float(value.detach()) if isinstance(value, torch.Tensor) else float(value)  # ty: ignore[invalid-argument-type]


def _gradient(point: ManifoldParameter) -> torch.Tensor:
    """The Euclidean gradient of a parameter, or zeros when the closure left none."""
    return point.grad if point.grad is not None else torch.zeros_like(point)


class RiemannianConjugateGradient(torch.optim.Optimizer):
    r"""Conjugate gradients on a manifold, with Polak-Ribière+ and a Wolfe line search.

    The direction of a step is the gradient corrected by the direction before it:

        d_k = -g_k + beta_k * T(d_{k-1}),
        beta_k = max(0, <g_k, g_k - T(g_{k-1})> / <g_{k-1}, g_{k-1}>)

    where `T` carries a tangent vector of the previous point to the tangent space of the
    current one. On both manifolds here that transport is `proju`, the same projection
    that `RiemannianAdam` uses for its momentum.

    **There is no learning rate.** The step along `d_k` is chosen by a line search that
    satisfies the strong Wolfe conditions,

        f(x_a) <= f(x) + c1 * a * <g, d>          (enough decrease)
        |<g(x_a), T(d)>| <= c2 * |<g, d>|         (enough flattening)

    with `x_a = retr(x, a * d)`. This is what conjugate gradients needs: the second
    condition is what keeps successive directions conjugate, and a method without it has
    to borrow a step size whose meaning does not carry from one problem to the next.

    The search needs the objective at trial points, so `step` takes a closure, as
    `torch.optim.LBFGS` does, and calls it several times per step. **The closure must
    zero the gradients, evaluate the objective at the current parameters, call
    `backward()` and return the value.** It must be the whole objective, not a
    mini-batch: a search along a mini-batch is a search on a different random function
    at every step, and it tunes the step to that one batch.

    Two guards keep the direction usable, both standard for Polak-Ribière: `beta` is
    clipped at zero, and a direction that does not descend is replaced by `-g`.

    Attributes
    ----------
    last_step : float
        The step that the last line search accepted. Zero means it found nothing better
        than the point it started from, and the next step restarts from `-g`.

    restarts : int
        Steps that fell back to steepest descent.

    evaluations : int
        Calls to the closure so far, the real cost of the line search.

    """

    def __init__(
        self,
        params: ParamsT,
        c1: float = 1e-4,
        c2: float = 0.1,
        max_evaluations: int = 20,
        eps: float = 1e-10,
    ) -> None:
        if not 0 < c1 < c2 < 1:
            msg = f"The Wolfe constants need 0 < c1 < c2 < 1, but c1={c1} and c2={c2}."
            raise ValueError(msg)
        super().__init__(
            params,
            {"c1": c1, "c2": c2, "max_evaluations": max_evaluations, "eps": eps},
        )
        self.last_step = 0.0
        self.previous_slope = 0.0
        self.restarts = 0
        self.evaluations = 0

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], torch.Tensor | float]) -> float: ...

    @torch.no_grad()
    def step(
        self, closure: Callable[[], torch.Tensor | float] | None = None
    ) -> float | None:
        if closure is None:
            msg = (
                "RiemannianConjugateGradient needs a closure. It has no learning rate, "
                "so it chooses its step by a line search, which has to evaluate the "
                "objective at trial points."
            )
            raise RuntimeError(msg)
        settings = self.param_groups[0]
        with torch.enable_grad():
            value = _as_float(closure())
        self.evaluations += 1

        points = [
            point
            for group in self.param_groups
            for point in group["params"]
            if point.grad is not None
        ]
        if not points:
            return value

        directions, slope = self._directions(points, settings["eps"])
        origin = [point.detach().clone() for point in points]
        self.last_step = self._search(
            closure, points, origin, directions, value, slope, settings
        )
        self._place(points, origin, directions, self.last_step)
        if self.last_step == 0.0:
            for point in points:
                self.state[point].clear()
        return value

    def _directions(
        self, points: list[ManifoldParameter], eps: float
    ) -> tuple[list[torch.Tensor], float]:
        """The Polak-Ribière+ direction of every parameter, and the slope along them."""
        directions = []
        slope = 0.0
        for point in points:
            manifold = point.manifold
            state = self.state[point]
            rgrad = manifold.egrad2rgrad(point, _gradient(point))

            if not state:
                direction = -rgrad
                self.restarts += 1
            else:
                carried_grad = manifold.proju(point, state["grad"])
                carried_direction = manifold.proju(point, state["direction"])
                beta = float(
                    (rgrad * (rgrad - carried_grad)).sum() / (state["grad_norm"] + eps)
                )
                direction = -rgrad + max(beta, 0.0) * carried_direction
                if float((direction * rgrad).sum()) >= 0.0:
                    direction = -rgrad
                    self.restarts += 1

            state["grad"] = rgrad
            state["grad_norm"] = float(rgrad.square().sum())
            state["direction"] = direction
            directions.append(direction)
            slope += float((rgrad * direction).sum())
        return directions, slope

    def _place(
        self,
        points: list[ManifoldParameter],
        origin: list[torch.Tensor],
        directions: list[torch.Tensor],
        step: float,
    ) -> None:
        """Moves every parameter to `retr(origin, step * direction)`."""
        for point, start, direction in zip(points, origin, directions, strict=True):
            point.copy_(point.manifold.retr(start, step * direction))

    def _search(
        self,
        closure: Callable[[], torch.Tensor | float],
        points: list[ManifoldParameter],
        origin: list[torch.Tensor],
        directions: list[torch.Tensor],
        value: float,
        slope: float,
        settings: dict[str, Any],
    ) -> float:
        """The strong Wolfe search of Nocedal and Wright, bracketing then bisection."""
        c1, c2 = settings["c1"], settings["c2"]
        budget = settings["max_evaluations"]

        def probe(step: float) -> tuple[float, float]:
            """The objective and the slope along the direction, at `step`."""
            self._place(points, origin, directions, step)
            with torch.enable_grad():
                trial = _as_float(closure())
            self.evaluations += 1
            derivative = 0.0
            for point, direction in zip(points, directions, strict=True):
                manifold = point.manifold
                rgrad = manifold.egrad2rgrad(point, _gradient(point))
                derivative += float((rgrad * manifold.proju(point, direction)).sum())
            return trial, derivative

        if slope >= 0.0:
            return 0.0
        if self.last_step > 0.0 and self.previous_slope < 0.0:
            # Nocedal and Wright 3.60: the first guess assumes the same first-order
            # change as the step before, which usually lands inside the bracket at once.
            step = self.last_step * self.previous_slope / slope
        else:
            largest = max(float(direction.abs().max()) for direction in directions)
            if largest == 0.0:
                return 0.0
            step = min(1.0, 1.0 / largest)
        self.previous_slope = slope

        previous = (0.0, value, slope)
        used = 0
        low = high = None
        while used < budget:
            trial, derivative = probe(step)
            used += 1
            descended = math.isfinite(trial) and trial <= value + c1 * step * slope
            if not descended or (used > 1 and trial >= previous[1]):
                low, high = previous, (step, trial, derivative)
                break
            if abs(derivative) <= -c2 * slope:
                return step
            if derivative >= 0.0:
                low, high = (step, trial, derivative), previous
                break
            previous = (step, trial, derivative)
            step = min(step * 2.0, 1e10)
        if low is None or high is None:
            return previous[0]

        while used < budget:
            step = _interpolate(low, high)
            trial, derivative = probe(step)
            used += 1
            if (
                not math.isfinite(trial)
                or trial > value + c1 * step * slope
                or trial >= low[1]
            ):
                high = (step, trial, derivative)
            else:
                if abs(derivative) <= -c2 * slope:
                    return step
                if derivative * (high[0] - low[0]) >= 0.0:
                    high = low
                low = (step, trial, derivative)
        return low[0]
