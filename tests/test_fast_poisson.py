"""Tests for the Alternating Poisson Regression fit of `fast_poisson`."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest
import scipy.special
import torch
from glmpca.fast_poisson import FastPoissonPCA

N_CELLS = 400
N_FEATURES = 120
N_PC = 2


@pytest.fixture(autouse=True)
def seed() -> None:
    """The random start of the fit reads the global torch generator."""
    torch.manual_seed(0)


def simulate() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Counts from the model itself: a size factor, an intercept and two components."""
    rng = np.random.default_rng(0)
    truth = {
        "scores": rng.normal(0.0, 0.7, size=(N_CELLS, N_PC)),
        "loadings": rng.normal(0.0, 0.7, size=(N_FEATURES, N_PC)),
        "size_factors": rng.normal(0.0, 0.4, size=N_CELLS),
        "intercept": rng.normal(0.5, 0.5, size=N_FEATURES),
    }
    truth["log_rate"] = (
        truth["size_factors"][:, None]
        + truth["intercept"][None, :]
        + truth["scores"] @ truth["loadings"].T
    )
    return rng.poisson(np.exp(truth["log_rate"])).astype(np.float32), truth


def subspace_cosines(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    q_first, _ = np.linalg.qr(first)
    q_second, _ = np.linalg.qr(second)
    return np.linalg.svd(q_first.T @ q_second, compute_uv=False)


def test_the_log_likelihood_never_falls() -> None:
    counts, _ = simulate()
    model = FastPoissonPCA(N_PC, max_iter=50, tol=1e-12)

    model.fit(torch.tensor(counts))

    # The block updates of APR improve the log-likelihood at every pass.
    steps = np.diff(model.log_likelihoods_)
    assert np.all(steps >= -1e-4), float(np.min(steps))


def test_a_fit_that_did_not_converge_warns() -> None:
    counts, _ = simulate()
    model = FastPoissonPCA(N_PC, max_iter=2, tol=1e-12)

    with pytest.warns(UserWarning, match="still moved"):
        model.fit(torch.tensor(counts))


def test_the_fit_recovers_the_simulated_structure() -> None:
    counts, truth = simulate()
    model = FastPoissonPCA(N_PC, max_iter=100, tol=1e-8)

    model.fit(torch.tensor(counts))

    assert model.scores_ is not None
    assert model.size_factors_ is not None
    assert model.intercept_ is not None
    # The factors are identifiable only up to rotation, and a component can trade off
    # against the size factor, so these bounds sit below what one start reaches (over
    # five starts: cosines 0.89 to 0.94, size factor 0.89 to 0.91, intercept 0.91 to
    # 0.95), while the log-likelihood is the same for every start.
    assert subspace_cosines(model.scores_.numpy(), truth["scores"]).min() > 0.85
    assert np.corrcoef(model.size_factors_.numpy(), truth["size_factors"])[0, 1] > 0.85
    assert np.corrcoef(model.intercept_.numpy(), truth["intercept"])[0, 1] > 0.9


def test_the_fit_reaches_the_log_likelihood_of_the_true_parameters() -> None:
    counts, truth = simulate()
    log_rate = truth["log_rate"]
    oracle = float(
        (counts * log_rate).sum()
        - np.exp(log_rate).sum()
        - scipy.special.gammaln(counts + 1).sum()
    )
    model = FastPoissonPCA(N_PC, max_iter=100, tol=1e-8)

    model.fit(torch.tensor(counts))

    # The maximum likelihood fit of a sample scores at least as well as the parameters
    # that generated it.
    assert model.log_likelihoods_[-1] >= oracle


def test_the_loadings_are_orthonormal() -> None:
    counts, _ = simulate()
    model = FastPoissonPCA(N_PC, max_iter=50, tol=1e-8)

    model.fit(torch.tensor(counts))

    assert model.loadings_ is not None
    torch.testing.assert_close(
        model.loadings_.T @ model.loadings_, torch.eye(N_PC), atol=1e-4, rtol=0
    )
    assert model.singular_values_ is not None
    assert torch.all(model.singular_values_[:-1] >= model.singular_values_[1:])


def test_transform_reproduces_the_fitted_coordinates() -> None:
    counts, _ = simulate()
    model = FastPoissonPCA(N_PC, max_iter=100, tol=1e-8)
    model.fit(torch.tensor(counts))
    assert model.scores_ is not None

    coordinates = model.transform(torch.tensor(counts))

    for component in range(N_PC):
        correlation = np.corrcoef(
            coordinates[:, component].numpy(), model.scores_[:, component].numpy()
        )[0, 1]
        assert abs(correlation) > 0.95


def test_an_anndata_input_is_accepted() -> None:
    counts, _ = simulate()
    model = FastPoissonPCA(N_PC, max_iter=20, tol=1e-6)

    model.fit(ad.AnnData(counts))

    assert model.scores_ is not None
    assert model.scores_.shape == (N_CELLS, N_PC)
    assert model.loadings_ is not None
    assert model.loadings_.shape == (N_FEATURES, N_PC)


def test_negative_values_are_rejected() -> None:
    model = FastPoissonPCA(N_PC)

    with pytest.raises(ValueError, match="needs counts"):
        model.fit(torch.full((4, 3), -1.0))


def test_an_input_with_one_row_is_rejected() -> None:
    model = FastPoissonPCA(N_PC)

    with pytest.raises(ValueError, match="at least 2 rows"):
        model.fit(torch.ones(1, 3))


def test_transform_before_fit_fails_with_advice() -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        FastPoissonPCA(N_PC).transform(torch.ones(4, 3))
