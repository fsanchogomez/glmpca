"""Tests for ``GLMPCA``: family selection, the fit/transform contract, AnnData input."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal

import anndata as ad
import numpy as np
import pytest
import torch
from geoopt import ManifoldParameter
from glmpca.ExponentialFamily import GLMFamily, Poisson
from glmpca.GLMPCA import GLMPCA, _RiemannianAdagrad
from scipy import sparse

if TYPE_CHECKING:
    from collections.abc import Callable

N_CELLS = 40
N_FEATURES = 12
N_PC = 2


@pytest.fixture(autouse=True)
def seed() -> None:
    torch.manual_seed(0)
    np.random.seed(0)


def sample(family: GLMFamily) -> torch.Tensor:
    """Data inside the support of ``family``."""
    shape = (N_CELLS, N_FEATURES)
    if family is GLMFamily.gaussian:
        return torch.randn(shape)
    if family is GLMFamily.poisson:
        return torch.poisson(torch.full(shape, 3.0))
    if family is GLMFamily.bernoulli:
        return torch.bernoulli(torch.full(shape, 0.4))
    if family in {GLMFamily.beta, GLMFamily.sigmoid_beta}:
        return torch.rand(shape) * 0.8 + 0.1
    return torch.rand(shape) + 0.5


@pytest.mark.parametrize("family", list(GLMFamily))
def test_fit_and_transform_project_onto_n_pc_components(family: GLMFamily) -> None:
    X = sample(family)
    model = GLMPCA(N_PC, family=family, max_iter=2, batch_size=16)

    assert model.fit(X)
    assert model.saturated_loadings_ is not None
    assert model.saturated_loadings_.shape == (N_FEATURES, N_PC)
    assert model.transform(X).shape == (N_CELLS, N_PC)


@pytest.mark.parametrize("family", list(GLMFamily))
def test_a_family_name_and_its_member_select_the_same_distribution(
    family: GLMFamily,
) -> None:
    assert type(GLMPCA(N_PC, family=family.value).exponential_family) is (
        family.distribution()
    )
    assert type(GLMPCA(N_PC, family=family).exponential_family) is (
        family.distribution()
    )


def test_an_exponential_family_instance_is_used_as_given() -> None:
    family = Poisson({"min_val": 1e-10})
    assert GLMPCA(N_PC, family=family).exponential_family is family


def test_an_unknown_family_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="nope"):
        GLMPCA(N_PC, family="nope")


def test_transform_before_fit_fails_with_advice() -> None:
    with pytest.raises(RuntimeError, match=r"Call fit\(\)"):
        GLMPCA(N_PC, family="poisson").transform(sample(GLMFamily.poisson))


def test_a_batch_size_larger_than_the_row_count_is_reduced_with_a_warning() -> None:
    X = sample(GLMFamily.poisson)[:10]
    model = GLMPCA(N_PC, family="poisson", max_iter=3, batch_size=256)

    with pytest.warns(UserWarning, match=r"batch_size=256 .* \(10\)"):
        assert model.fit(X)

    assert model.batch_size == 256
    assert [len(scores) for scores in model.loadings_learning_scores_] == [3]


def test_a_batch_size_equal_to_the_row_count_is_used_without_a_warning() -> None:
    X = sample(GLMFamily.poisson)[:16]
    model = GLMPCA(N_PC, family="poisson", max_iter=3, batch_size=16)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        assert model.fit(X)


def fitted_loadings(X: torch.Tensor | ad.AnnData) -> torch.Tensor:
    torch.manual_seed(0)
    np.random.seed(0)
    model = GLMPCA(N_PC, family="poisson", max_iter=2, batch_size=8)
    model.fit(X)
    assert model.saturated_loadings_ is not None
    return model.saturated_loadings_.detach()


@pytest.mark.parametrize(
    "matrix_type",
    [
        np.asarray,
        sparse.csr_matrix,
        sparse.csc_matrix,
        sparse.csr_array,
        sparse.csc_array,
    ],
    ids=lambda matrix_type: matrix_type.__name__,
)
def test_anndata_input_is_fitted_with_cells_as_features(
    matrix_type: Callable[[np.ndarray], Any],
) -> None:
    counts = (
        np.random
        .default_rng(0)
        .poisson(3.0, size=(N_CELLS, N_FEATURES))
        .astype(np.float32)
    )
    adata = ad.AnnData(matrix_type(counts))

    torch.testing.assert_close(
        fitted_loadings(adata), fitted_loadings(torch.Tensor(counts.T))
    )


@pytest.mark.parametrize("init", ["spectral", "random"])
@pytest.mark.parametrize("family", list(GLMFamily))
def test_fitted_loadings_are_orthonormal(
    family: GLMFamily, init: Literal["spectral", "random"]
) -> None:
    model = GLMPCA(N_PC, family=family, init=init, max_iter=5, batch_size=16)
    model.fit(sample(family))

    assert model.saturated_loadings_ is not None
    loadings = model.saturated_loadings_.detach()
    torch.testing.assert_close(
        loadings.T @ loadings, torch.eye(N_PC), rtol=0, atol=1e-5
    )


def test_riemannian_adagrad_on_euclidean_matches_torch_adagrad() -> None:
    start, target = torch.randn(7), torch.randn(7)
    ours = ManifoldParameter(start.clone())
    theirs = torch.nn.Parameter(start.clone())
    optimizers = {
        ours: _RiemannianAdagrad([ours], lr=0.1),
        theirs: torch.optim.Adagrad([theirs], lr=0.1, eps=1e-10),
    }
    for _ in range(50):
        for point, optimizer in optimizers.items():
            optimizer.zero_grad()
            (point - target).square().sum().backward()
            optimizer.step()

    torch.testing.assert_close(ours.detach(), theirs.detach(), rtol=0, atol=1e-6)
