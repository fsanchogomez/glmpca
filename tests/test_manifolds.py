"""Tests for the manifold primitives."""

from __future__ import annotations

import pickle

import pytest
import torch
from glmpca.manifolds import (
    Euclidean,
    EuclideanStiefel,
    ManifoldParameter,
    RiemannianAdagrad,
)

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
