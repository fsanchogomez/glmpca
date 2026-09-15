"""Check each exponential family against the scipy distribution it encodes.

Each test sweeps a grid of parameters and compares the density, or its log, over
a grid of observations.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pytest
import scipy
import torch
from glmpca.ExponentialFamily import (
    Bernoulli,
    Beta,
    Gamma,
    Gaussian,
    Poisson,
    SigmoidBeta,
)

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture(autouse=True)
def seed() -> None:
    torch.manual_seed(42)


def test_gaussian_pdf_matches_scipy() -> None:
    X = torch.linspace(-5, 5, 500)
    for theta in torch.normal(0.0, 1.0, (25,)):
        pdf = Gaussian().distribution(X, theta.expand_as(X))
        np.testing.assert_array_almost_equal(
            pdf.numpy(), scipy.stats.norm.pdf(X.numpy(), loc=float(theta)), decimal=3
        )


def test_bernoulli_success_probability_is_the_sigmoid_of_theta() -> None:
    theta = torch.linspace(-30, 30, 1000)
    success = Bernoulli().distribution(torch.ones_like(theta), theta)
    torch.testing.assert_close(success, torch.sigmoid(theta), rtol=0, atol=1e-6)


def test_poisson_log_pmf_matches_scipy() -> None:
    X = torch.linspace(0, 100, 101)
    for theta in torch.linspace(-50, 5, 10):
        log_pmf = Poisson().log_distribution(X, theta.expand_as(X))
        np.testing.assert_array_almost_equal(
            log_pmf.numpy(),
            scipy.stats.poisson.logpmf(X.numpy(), np.exp(float(theta))),
            decimal=3,
        )


@pytest.mark.parametrize("m", [1.0, 2.5])
def test_poisson_saturated_parameters_map_zero_counts_to_minus_m(m: float) -> None:
    X = torch.tensor([0.0, 1.0, 3.0, 0.0])
    torch.testing.assert_close(
        Poisson({"m": m}).invert_g(X), torch.tensor([-m, 0.0, math.log(3.0), -m])
    )


@pytest.mark.parametrize(
    "family_params", [None, {"n_jobs": 2}], ids=["none", "partial"]
)
def test_poisson_m_defaults_to_one(family_params: dict[str, int] | None) -> None:
    assert Poisson(family_params).invert_g(torch.zeros(2)).tolist() == [-1.0, -1.0]


@pytest.mark.parametrize(
    ("family", "thetas", "mean"),
    [
        (Beta, torch.linspace(0.01, 0.99, 10), lambda theta: theta),
        (SigmoidBeta, torch.logit(torch.linspace(0.01, 0.99, 10)), torch.sigmoid),
    ],
    ids=["beta", "sigmoid_beta"],
)
def test_beta_log_pdf_matches_scipy(
    family: type[Beta],
    thetas: torch.Tensor,
    mean: Callable[[torch.Tensor], torch.Tensor],
) -> None:
    X = torch.linspace(1e-4, 1 - 1e-4, 100)
    for nu in torch.rand(20) * 10:
        for theta in thetas:
            distribution = family()
            distribution.family_params["nu"] = nu.expand_as(X)
            log_pdf = distribution.log_distribution(X, theta.expand_as(X))
            p = float(mean(theta))
            np.testing.assert_array_almost_equal(
                log_pdf.numpy(),
                scipy.stats.beta.logpdf(X.numpy(), p * float(nu), (1 - p) * float(nu)),
                decimal=2,
            )


def test_gamma_pdf_matches_scipy() -> None:
    X = torch.logspace(-8, 5, 1000)
    for nu in torch.rand(20) * 10:
        for theta in torch.logspace(-5, 3, 50):
            distribution = Gamma()
            distribution.family_params["nu"] = nu.expand_as(X)
            pdf = distribution.distribution(X, theta.expand_as(X))
            np.testing.assert_array_almost_equal(
                pdf.numpy(),
                scipy.stats.gamma.pdf(
                    X.numpy(), a=float(theta) + 1, loc=0, scale=1 / float(nu)
                ),
                decimal=2,
            )


@pytest.mark.parametrize("n_jobs", [4, -1])
@pytest.mark.parametrize(
    "family", [Beta, SigmoidBeta, Gamma], ids=["beta", "sigmoid_beta", "gamma"]
)
def test_parallel_family_parameter_fit_matches_sequential(
    family: type[Beta | Gamma], n_jobs: int
) -> None:
    X = torch.rand(50, 30) * 0.8 + 0.1
    sequential, parallel = family({"n_jobs": 1}), family({"n_jobs": n_jobs})
    sequential.initialize_family_parameters(X)
    parallel.initialize_family_parameters(X)

    torch.testing.assert_close(
        parallel.family_params["nu"], sequential.family_params["nu"], rtol=0, atol=0
    )


@pytest.mark.parametrize("value", [0.0, -1.0], ids=["zero", "negative"])
def test_gamma_fit_rejects_values_that_are_not_positive(value: float) -> None:
    X = torch.rand(20, 5) + 0.5
    X[3, 2] = value

    with pytest.raises(ValueError, match="1 of 100 values are 0 or negative"):
        Gamma().initialize_family_parameters(X)
