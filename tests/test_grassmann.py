"""Tests for the Grassmann manifold and for `canonical_basis`."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from glmpca.GLMPCA import GLMPCA
from glmpca.manifolds import (
    EuclideanStiefel,
    Grassmann,
    ManifoldParameter,
    RiemannianAdagrad,
    canonical_basis,
)

N_ROWS = 40
N_COLUMNS = 4


@pytest.fixture(autouse=True)
def seed() -> None:
    torch.manual_seed(0)
    np.random.seed(0)


def point() -> torch.Tensor:
    return Grassmann().random(N_ROWS, N_COLUMNS)


def rotation() -> torch.Tensor:
    return torch.linalg.qr(torch.randn(N_COLUMNS, N_COLUMNS))[0]


def span_cost(loadings: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
    """A cost of the span alone: the residual of `data` outside it."""
    return (data - data @ loadings @ loadings.T).square().sum()


def test_the_projection_is_horizontal() -> None:
    x = point()

    tangent = Grassmann().proju(x, torch.randn(N_ROWS, N_COLUMNS))

    torch.testing.assert_close(
        x.T @ tangent, torch.zeros(N_COLUMNS, N_COLUMNS), atol=1e-5, rtol=0
    )


def test_the_projection_drops_a_pure_rotation_of_the_basis() -> None:
    x = point()
    skew = torch.randn(N_COLUMNS, N_COLUMNS)
    vertical = x @ (skew - skew.T)

    torch.testing.assert_close(
        Grassmann().proju(x, vertical), torch.zeros_like(x), atol=1e-5, rtol=0
    )
    assert float(EuclideanStiefel().proju(x, vertical).norm()) > 0.1


def test_egrad2rgrad_is_the_projection() -> None:
    x, u = point(), torch.randn(N_ROWS, N_COLUMNS)

    torch.testing.assert_close(Grassmann().egrad2rgrad(x, u), Grassmann().proju(x, u))


def test_the_two_manifolds_agree_on_the_gradient_of_a_cost_of_the_span() -> None:
    data = torch.randn(30, N_ROWS)
    x = point().requires_grad_(True)
    (gradient,) = torch.autograd.grad(span_cost(x, data), x)
    x = x.detach()

    torch.testing.assert_close(
        Grassmann().egrad2rgrad(x, gradient),
        EuclideanStiefel().egrad2rgrad(x, gradient),
        atol=1e-4,
        rtol=1e-4,
    )


def test_adagrad_on_the_grassmann_manifold_keeps_the_basis_orthonormal() -> None:
    data = torch.randn(30, N_ROWS)
    parameter = ManifoldParameter(point(), manifold=Grassmann())
    optimizer = RiemannianAdagrad([parameter], lr=0.1)

    for _ in range(20):
        optimizer.zero_grad()
        span_cost(parameter, data).backward()
        optimizer.step()

    torch.testing.assert_close(
        parameter.data.T @ parameter.data, torch.eye(N_COLUMNS), atol=1e-5, rtol=0
    )


def test_the_canonical_basis_keeps_the_span() -> None:
    x, data = point(), torch.randn(60, N_ROWS)

    basis = canonical_basis(x, [data])

    torch.testing.assert_close(basis @ basis.T, x @ x.T, atol=1e-5, rtol=0)
    torch.testing.assert_close(basis.T @ basis, torch.eye(N_COLUMNS), atol=1e-5, rtol=0)


def test_the_canonical_scores_are_uncorrelated_and_ordered_by_variance() -> None:
    x, data = point(), torch.randn(200, N_ROWS) * torch.linspace(0.5, 3.0, N_ROWS)

    scores = data @ canonical_basis(x, [data])

    covariance = torch.cov(scores.T)
    variances = torch.diagonal(covariance)
    assert torch.all(variances[:-1] >= variances[1:])
    off_diagonal = covariance - torch.diag(variances)
    assert float(off_diagonal.abs().max()) < 1e-4 * float(variances.max())


def test_every_basis_of_a_span_gives_the_same_canonical_basis() -> None:
    x, data = point(), torch.randn(60, N_ROWS)

    torch.testing.assert_close(
        canonical_basis(x @ rotation(), [data]),
        canonical_basis(x, [data]),
        atol=1e-5,
        rtol=0,
    )


def test_the_chunks_do_not_move_the_canonical_basis() -> None:
    x, data = point(), torch.randn(60, N_ROWS)

    torch.testing.assert_close(
        canonical_basis(x, list(data.split(7))),
        canonical_basis(x, [data]),
        atol=1e-6,
        rtol=0,
    )


def test_the_fitted_loadings_live_on_the_grassmann_manifold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifolds: list[object] = []
    original = GLMPCA._create_saturated_loading_optim

    def recording(self: GLMPCA, *arguments: object, **keywords: object) -> object:
        result = original(self, *arguments, **keywords)  # ty: ignore[invalid-argument-type]
        loadings = result[1]
        assert isinstance(loadings, ManifoldParameter)
        manifolds.append(loadings.manifold)
        return result

    monkeypatch.setattr(GLMPCA, "_create_saturated_loading_optim", recording)
    counts = torch.poisson(torch.full((40, 12), 3.0))

    for init in ("spectral", "lsi", "random"):
        GLMPCA(2, family="poisson", init=init, max_iter=1, batch_size=16).fit(counts)

    assert all(isinstance(manifold, Grassmann) for manifold in manifolds)


def test_a_fit_reports_its_components_ordered_by_variance() -> None:
    counts = torch.poisson(torch.full((200, 30), 3.0))
    model = GLMPCA(4, family="poisson", max_iter=5, batch_size=32)

    model.fit(counts)
    variances = model.transform(counts).var(dim=0)

    assert torch.all(variances[:-1] >= variances[1:]), variances
