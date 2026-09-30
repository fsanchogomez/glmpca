"""Tests for the manifold primitives."""

from __future__ import annotations

import itertools
import pickle
from typing import TYPE_CHECKING

import pytest
import torch
from glmpca.manifolds import (
    Euclidean,
    EuclideanStiefel,
    ManifoldParameter,
    RiemannianAdagrad,
    RiemannianAdam,
    RiemannianConjugateGradient,
)

if TYPE_CHECKING:
    from collections.abc import Callable

N_ROWS = 40
N_COLUMNS = 4


@pytest.fixture(autouse=True)
def seed() -> None:
    torch.manual_seed(0)


def point_and_direction() -> tuple[torch.Tensor, torch.Tensor]:
    x = EuclideanStiefel().random(N_ROWS, N_COLUMNS)
    return x, torch.randn(N_ROWS, N_COLUMNS)


def test_random_gives_an_orthonormal_point() -> None:
    x = EuclideanStiefel().random(N_ROWS, N_COLUMNS)

    assert x.shape == (N_ROWS, N_COLUMNS)
    torch.testing.assert_close(x.T @ x, torch.eye(N_COLUMNS), atol=1e-6, rtol=0)


def test_random_rejects_more_columns_than_rows() -> None:
    with pytest.raises(ValueError, match="k <= n"):
        EuclideanStiefel().random(3, 5)


def test_proju_gives_a_tangent_direction() -> None:
    x, u = point_and_direction()

    tangent = EuclideanStiefel().proju(x, u)

    # A direction is tangent to the Stiefel manifold when x.T @ direction is skew.
    inner = x.T @ tangent
    torch.testing.assert_close(inner, -inner.T, atol=1e-6, rtol=0)


def test_egrad2rgrad_is_the_projection() -> None:
    x, u = point_and_direction()
    manifold = EuclideanStiefel()

    torch.testing.assert_close(manifold.egrad2rgrad(x, u), manifold.proju(x, u))


def test_retr_returns_to_the_manifold() -> None:
    x, u = point_and_direction()

    moved = EuclideanStiefel().retr(x, 0.1 * u)

    torch.testing.assert_close(moved.T @ moved, torch.eye(N_COLUMNS), atol=1e-6, rtol=0)


def test_retr_does_not_move_a_point_without_a_direction() -> None:
    x, _ = point_and_direction()

    torch.testing.assert_close(
        EuclideanStiefel().retr(x, torch.zeros_like(x)), x, atol=1e-6, rtol=0
    )


def test_retr_keeps_the_column_signs_when_r_has_a_zero_diagonal() -> None:
    # A rank-deficient input gives a zero on the diagonal of R. sign(0) is 0, so a
    # plain sign correction would zero that column of Q.
    x = torch.zeros(3, 2)
    x[0, 0] = 1.0

    moved = EuclideanStiefel().retr(x, torch.zeros_like(x))

    assert torch.isfinite(moved).all()
    assert not torch.allclose(moved[:, 1], torch.zeros(3))


def test_euclidean_leaves_the_gradient_and_adds_the_step() -> None:
    manifold = Euclidean()
    x, u = torch.randn(5), torch.randn(5)

    assert manifold.proju(x, u) is u
    assert manifold.egrad2rgrad(x, u) is u
    torch.testing.assert_close(manifold.retr(x, u), x + u)


def test_a_parameter_without_a_manifold_is_euclidean() -> None:
    parameter = ManifoldParameter(torch.randn(5))

    assert isinstance(parameter.manifold, Euclidean)
    assert parameter.requires_grad


def test_a_parameter_keeps_the_manifold_it_is_given() -> None:
    x = EuclideanStiefel().random(N_ROWS, N_COLUMNS)

    parameter = ManifoldParameter(x, manifold=EuclideanStiefel())

    assert isinstance(parameter.manifold, EuclideanStiefel)
    assert "Stiefel(euclidean)" in repr(parameter)


def test_a_parameter_survives_a_pickle_round_trip() -> None:
    parameter = ManifoldParameter(
        EuclideanStiefel().random(N_ROWS, N_COLUMNS), manifold=EuclideanStiefel()
    )

    restored = pickle.loads(pickle.dumps(parameter))

    torch.testing.assert_close(restored.data, parameter.data)
    assert isinstance(restored.manifold, EuclideanStiefel)


def test_the_optimiser_keeps_the_loadings_orthonormal() -> None:
    parameter = ManifoldParameter(
        EuclideanStiefel().random(N_ROWS, N_COLUMNS), manifold=EuclideanStiefel()
    )
    target = torch.randn(N_ROWS, N_COLUMNS)
    optimizer = RiemannianAdagrad([parameter], lr=0.1)

    for _ in range(20):
        optimizer.zero_grad()
        ((parameter - target) ** 2).sum().backward()
        optimizer.step()

    torch.testing.assert_close(
        parameter.data.T @ parameter.data, torch.eye(N_COLUMNS), atol=1e-5, rtol=0
    )


def test_riemannian_adam_on_euclidean_matches_torch_adam() -> None:
    start = torch.randn(6, 3)
    target = torch.randn(6, 3)
    ours = ManifoldParameter(start.clone())
    theirs = torch.nn.Parameter(start.clone())
    our_optimizer = RiemannianAdam([ours], lr=0.05)
    their_optimizer = torch.optim.Adam([theirs], lr=0.05)

    for _ in range(50):
        for parameter, optimizer in ((ours, our_optimizer), (theirs, their_optimizer)):
            optimizer.zero_grad()
            ((parameter - target) ** 2).sum().backward()
            optimizer.step()

    torch.testing.assert_close(ours.data, theirs.data, atol=1e-6, rtol=0)


def test_riemannian_adam_keeps_the_loadings_orthonormal() -> None:
    parameter = ManifoldParameter(
        EuclideanStiefel().random(N_ROWS, N_COLUMNS), manifold=EuclideanStiefel()
    )
    target = torch.randn(N_ROWS, N_COLUMNS)
    optimizer = RiemannianAdam([parameter], lr=0.05)

    for _ in range(30):
        optimizer.zero_grad()
        ((parameter - target) ** 2).sum().backward()
        optimizer.step()

    torch.testing.assert_close(
        parameter.data.T @ parameter.data, torch.eye(N_COLUMNS), atol=1e-5, rtol=0
    )


def quadratic(
    parameter: torch.Tensor, matrix: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """An ill-conditioned quadratic, where conjugacy is what buys the speed."""
    difference = parameter - target
    return (difference * (matrix @ difference)).sum()


def quadratic_problem(size: int = 30) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    return torch.diag(torch.logspace(0, 3, size)), torch.randn(size, 1)


def closure_of(
    optimizer: torch.optim.Optimizer,
    parameter: ManifoldParameter,
    matrix: torch.Tensor,
    target: torch.Tensor,
) -> Callable[[], torch.Tensor]:
    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = quadratic(parameter, matrix, target)
        loss.backward()
        return loss

    return closure


def test_conjugate_gradient_beats_the_scaled_optimisers_on_a_quadratic() -> None:
    matrix, target = quadratic_problem()
    results = {}

    parameter = ManifoldParameter(torch.zeros(matrix.shape[0], 1))
    optimizer = RiemannianConjugateGradient([parameter])
    closure = closure_of(optimizer, parameter, matrix, target)
    for _ in range(20):
        optimizer.step(closure)
    results["cg"] = float(quadratic(parameter.data, matrix, target))

    parameter = ManifoldParameter(torch.zeros(matrix.shape[0], 1))
    plain = RiemannianAdagrad([parameter], lr=0.1)
    for _ in range(20):
        plain.zero_grad()
        quadratic(parameter, matrix, target).backward()
        plain.step()
    results["adagrad"] = float(quadratic(parameter.data, matrix, target))

    assert results["cg"] < results["adagrad"], results


def test_the_line_search_never_lets_the_cost_rise() -> None:
    matrix, target = quadratic_problem()
    parameter = ManifoldParameter(torch.zeros(matrix.shape[0], 1))
    optimizer = RiemannianConjugateGradient([parameter])
    closure = closure_of(optimizer, parameter, matrix, target)

    costs = [float(optimizer.step(closure)) for _ in range(20)]

    assert all(later <= earlier for earlier, later in itertools.pairwise(costs)), costs


def test_the_accepted_step_satisfies_the_wolfe_conditions() -> None:
    matrix, target = quadratic_problem()
    parameter = ManifoldParameter(torch.zeros(matrix.shape[0], 1))
    optimizer = RiemannianConjugateGradient([parameter], c1=1e-4, c2=0.1)
    closure = closure_of(optimizer, parameter, matrix, target)
    optimizer.step(closure)

    start = parameter.data.clone()
    value = float(optimizer.step(closure))
    state = optimizer.state[parameter]
    direction = state["direction"].clone()
    slope = float((state["grad"] * direction).sum())
    step = optimizer.last_step

    # Enough decrease, then enough flattening, at the point the search accepted.
    landed = float(quadratic(parameter.data, matrix, target))
    assert landed <= value + 1e-4 * step * slope
    optimizer.zero_grad()
    quadratic(parameter, matrix, target).backward()
    assert parameter.grad is not None
    assert abs(float((parameter.grad * direction).sum())) <= 0.1 * abs(slope)
    assert not torch.equal(parameter.data, start)


def test_conjugate_gradient_keeps_the_loadings_orthonormal() -> None:
    parameter = ManifoldParameter(
        EuclideanStiefel().random(N_ROWS, N_COLUMNS), manifold=EuclideanStiefel()
    )
    target = torch.randn(N_ROWS, N_COLUMNS)
    optimizer = RiemannianConjugateGradient([parameter])

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = ((parameter - target) ** 2).sum()
        loss.backward()
        return loss

    for _ in range(20):
        optimizer.step(closure)

    torch.testing.assert_close(
        parameter.data.T @ parameter.data, torch.eye(N_COLUMNS), atol=1e-5, rtol=0
    )


def test_the_first_direction_is_steepest_descent() -> None:
    matrix, target = quadratic_problem(size=4)
    parameter = ManifoldParameter(torch.zeros(4, 1))
    optimizer = RiemannianConjugateGradient([parameter])
    closure = closure_of(optimizer, parameter, matrix, target)

    optimizer.step(closure)

    assert optimizer.restarts == 1
    state = optimizer.state[parameter]
    torch.testing.assert_close(state["direction"], -state["grad"], atol=0, rtol=0)


def test_every_direction_descends() -> None:
    matrix, target = quadratic_problem(size=20)
    parameter = ManifoldParameter(torch.zeros(20, 1))
    optimizer = RiemannianConjugateGradient([parameter])
    closure = closure_of(optimizer, parameter, matrix, target)

    products = []
    for _ in range(20):
        optimizer.step(closure)
        state = optimizer.state[parameter]
        products.append(float((state["direction"] * state["grad"]).sum()))

    assert all(product < 0 for product in products), max(products)


def test_reusing_the_last_evaluation_changes_nothing_but_the_count() -> None:
    matrix, target = quadratic_problem()
    runs = {}
    for reuse in (True, False):
        parameter = ManifoldParameter(torch.zeros(matrix.shape[0], 1))
        optimizer = RiemannianConjugateGradient([parameter])
        closure = closure_of(optimizer, parameter, matrix, target)
        costs = []
        for _ in range(20):
            if not reuse:
                with torch.no_grad():
                    parameter.add_(0.0)
            costs.append(float(optimizer.step(closure)))
        runs[reuse] = (parameter.data.clone(), costs, optimizer.evaluations)

    torch.testing.assert_close(runs[True][0], runs[False][0], atol=0, rtol=0)
    assert runs[True][1] == runs[False][1]
    assert runs[True][2] < runs[False][2], (runs[True][2], runs[False][2])


def test_evaluations_count_every_call_of_the_closure() -> None:
    matrix, target = quadratic_problem()
    parameter = ManifoldParameter(torch.zeros(matrix.shape[0], 1))
    optimizer = RiemannianConjugateGradient([parameter])
    inner = closure_of(optimizer, parameter, matrix, target)
    calls = 0

    def closure() -> torch.Tensor:
        nonlocal calls
        calls += 1
        return inner()

    for _ in range(20):
        optimizer.step(closure)

    assert optimizer.evaluations == calls


def test_a_step_without_a_closure_is_refused() -> None:
    optimizer = RiemannianConjugateGradient([ManifoldParameter(torch.zeros(3))])

    with pytest.raises(RuntimeError, match="needs a closure"):
        optimizer.step()


def test_wolfe_constants_out_of_order_are_rejected() -> None:
    with pytest.raises(ValueError, match="0 < c1 < c2 < 1"):
        RiemannianConjugateGradient([ManifoldParameter(torch.zeros(3))], c1=0.5, c2=0.1)
