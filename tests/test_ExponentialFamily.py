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
    GLMFamily,
    LogNormal,
    NegativeBinomial,
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


def test_bernoulli_log_partition_and_gradient_stay_finite_for_large_theta() -> None:
    theta = torch.tensor([-1000.0, 0.0, 89.0, 1000.0], requires_grad=True)
    log_partition = Bernoulli().log_partition(theta)
    log_partition.sum().backward()

    torch.testing.assert_close(
        log_partition.detach(), torch.tensor([0.0, math.log(2.0), 89.0, 1000.0])
    )
    assert theta.grad is not None
    torch.testing.assert_close(theta.grad, torch.sigmoid(theta.detach()))


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
@pytest.mark.parametrize("family", [Gamma, LogNormal], ids=["gamma", "lognormal"])
def test_positive_families_reject_values_that_are_not_positive(
    family: type[Gamma | LogNormal], value: float
) -> None:
    X = torch.rand(20, 5) + 0.5
    X[3, 2] = value

    with pytest.raises(
        ValueError,
        match=rf"The {family.__name__} family .* 1 of 100 values are 0 or negative",
    ):
        family().initialize_family_parameters(X)


@pytest.mark.parametrize("family", list(GLMFamily), ids=lambda family: family.value)
def test_every_family_fills_in_defaults_and_keyword_arguments(
    family: GLMFamily,
) -> None:
    distribution = family.distribution()
    defaults = distribution().family_params
    family_params = distribution({"n_jobs": 2}, max_val=7).family_params

    assert set(defaults) <= set(family_params)
    assert family_params["n_jobs"] == 2
    assert family_params["max_val"] == 7


def test_negative_binomial_log_pmf_matches_scipy() -> None:
    X = torch.linspace(0, 60, 61)
    family = NegativeBinomial()
    for nu in (2.0, 5.0, 20.0):
        family.family_params["nu"] = torch.tensor(nu)
        for probability in (0.2, 0.5, 0.8):
            theta = torch.full_like(X, math.log(1 - probability))
            np.testing.assert_array_almost_equal(
                family.log_distribution(X, theta).numpy(),
                scipy.stats.nbinom.logpmf(X.numpy(), nu, probability),
                decimal=4,
            )


@pytest.mark.parametrize("m", [1.0, 2.5])
def test_negative_binomial_maps_zero_counts_to_minus_m(m: float) -> None:
    family = NegativeBinomial({"m": m})
    family.family_params["nu"] = torch.tensor(4.0)

    theta = family.invert_g(torch.tensor([0.0, 4.0, 12.0]))

    torch.testing.assert_close(theta, torch.tensor([-m, math.log(0.5), math.log(0.75)]))


def test_negative_binomial_dispersion_is_estimated_per_feature() -> None:
    true_nu = torch.tensor([2.0, 10.0])
    X = torch.stack(
        [
            torch.distributions.NegativeBinomial(nu, probs=torch.tensor(0.5)).sample((
                20000,
            ))
            for nu in true_nu
        ],
        dim=1,
    )
    family = NegativeBinomial()

    family.initialize_family_parameters(X)

    torch.testing.assert_close(family.family_params["nu"], true_nu, rtol=0.15, atol=0.0)


def test_negative_binomial_dispersion_falls_back_without_overdispersion() -> None:
    family = NegativeBinomial()

    family.initialize_family_parameters(torch.full((10, 2), 3.0))

    assert torch.all(family.family_params["nu"] == family.family_params["max_val"])


def scipy_dispersion(column: np.ndarray, max_val: float = 1e4) -> float:
    """The same profile score, solved per column by scipy, as a reference."""
    column = column.astype(np.float64)
    mean = column.mean()

    def score(nu: float) -> float:
        return float(
            (scipy.special.digamma(column + nu) - scipy.special.digamma(nu)).sum()
            - column.size * np.log1p(mean / nu)
        )

    if score(max_val) > 0:
        return max_val
    return float(scipy.optimize.brentq(score, 1e-3, max_val, xtol=1e-9))


def test_negative_binomial_mle_matches_a_scipy_solve() -> None:
    counts = np.random.default_rng(0).negative_binomial(4.0, 4.0 / 7.0, size=(500, 8))
    X = torch.tensor(counts.astype(np.float32))
    family = NegativeBinomial({"method": "mle"})

    family.initialize_family_parameters(X)

    reference = torch.tensor(
        [scipy_dispersion(column) for column in counts.T], dtype=torch.float32
    )
    torch.testing.assert_close(family.family_params["nu"], reference, rtol=1e-4, atol=0)


@pytest.mark.parametrize("true_nu", [0.5, 3.0])
def test_negative_binomial_mle_recovers_the_dispersion(true_nu: float) -> None:
    counts = np.random.default_rng(1).negative_binomial(
        true_nu, true_nu / (true_nu + 2.0), size=(20000, 2)
    )
    family = NegativeBinomial({"method": "mle"})

    family.initialize_family_parameters(torch.tensor(counts.astype(np.float32)))

    torch.testing.assert_close(
        family.family_params["nu"],
        torch.full((2,), true_nu),
        rtol=0.15,
        atol=0.0,
    )


def test_negative_binomial_mle_does_not_depend_on_the_chunk_size() -> None:
    X = torch.tensor(
        np.random
        .default_rng(2)
        .negative_binomial(3.0, 3.0 / 5.0, size=(200, 5))
        .astype(np.float32)
    )
    chunked = NegativeBinomial({"method": "mle", "chunk_size": 7})
    whole = NegativeBinomial({"method": "mle", "chunk_size": 10**9})

    chunked.initialize_family_parameters(X)
    whole.initialize_family_parameters(X)

    torch.testing.assert_close(chunked.family_params["nu"], whole.family_params["nu"])


def test_negative_binomial_mle_falls_back_without_overdispersion() -> None:
    family = NegativeBinomial({"method": "mle"})

    family.initialize_family_parameters(torch.full((10, 2), 3.0))

    assert torch.all(family.family_params["nu"] == family.family_params["max_val"])


def test_negative_binomial_mle_finds_little_overdispersion_in_poisson_data() -> None:
    X = torch.tensor(
        np.random.default_rng(3).poisson(4.0, size=(500, 4)).astype(np.float32)
    )
    family = NegativeBinomial({"method": "mle"})

    family.initialize_family_parameters(X)

    # The fitted variance is mean * (1 + mean / nu), so mean / nu is the excess over
    # Poisson. Sampling noise keeps nu finite, but the excess stays small.
    assert torch.all(X.mean(dim=0) / family.family_params["nu"] < 0.2)


def test_negative_binomial_rejects_an_unknown_method() -> None:
    family = NegativeBinomial({"method": "bayes"})

    with pytest.raises(ValueError, match="Use 'moments' or 'mle'"):
        family.initialize_family_parameters(torch.ones(4, 2))


def test_lognormal_log_pdf_matches_scipy() -> None:
    X = torch.logspace(-3, 3, 200)
    for nu in (0.5, 1.0, 2.5):
        for theta in (-1.0, 0.0, 2.0):
            family = LogNormal()
            family.family_params["nu"] = torch.full_like(X, nu)
            log_pdf = family.log_distribution(X, torch.full_like(X, theta))
            np.testing.assert_array_almost_equal(
                log_pdf.numpy(),
                scipy.stats.lognorm.logpdf(X.numpy(), s=nu, scale=math.exp(theta)),
                decimal=3,
            )


@pytest.mark.parametrize("family", list(GLMFamily))
def test_the_log_base_measure_completes_the_density(family: GLMFamily) -> None:
    """log h is what the optimisation objective leaves out of the log density."""
    rng = np.random.default_rng(0)
    shape = (20, 3)
    if family in {GLMFamily.beta, GLMFamily.sigmoid_beta}:
        X = rng.random(size=shape).clip(0.05, 0.95)
    elif family in {GLMFamily.gamma, GLMFamily.lognormal}:
        X = rng.random(size=shape) + 0.5
    elif family is GLMFamily.bernoulli:
        X = rng.binomial(1, 0.4, size=shape)
    elif family is GLMFamily.gaussian:
        X = rng.normal(size=shape)
    else:
        X = rng.poisson(3.0, size=shape)
    X = torch.tensor(X.astype(np.float32))
    distribution = family.distribution()()
    distribution.initialize_family_parameters(X)
    theta = distribution.invert_g(X)

    log_density = distribution.log_distribution(X, theta)
    without_base = distribution.exponential_term(X, theta) - distribution.log_partition(
        theta
    )

    torch.testing.assert_close(
        log_density - without_base,
        distribution.log_base_measure(X),
        rtol=1e-5,
        atol=1e-5,
    )
