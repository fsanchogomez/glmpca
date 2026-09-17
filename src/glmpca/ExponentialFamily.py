from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy
import torch
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

saturation_eps = 10**-10


def _n_workers(n_jobs: int) -> int:
    if n_jobs >= 0:
        return n_jobs
    available = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else os.cpu_count() or 1
    )
    return max(available + 1 + n_jobs, 1)


def _fit_columns(
    fit: Callable[[np.ndarray], tuple[float, ...]],
    columns: Iterable[np.ndarray],
    n_columns: int,
    n_jobs: int,
) -> list[tuple[float, ...]]:
    workers = _n_workers(n_jobs)
    if workers == 1:
        return list(tqdm(map(fit, columns), total=n_columns))
    with ThreadPoolExecutor(workers) as executor:
        return list(tqdm(executor.map(fit, columns), total=n_columns))


def _require_positive(X: torch.Tensor, family_name: str) -> None:
    not_positive = int((X <= 0).sum())
    if not_positive:
        msg = (
            f"The {family_name} family is defined only for values greater than 0, "
            f"but {not_positive} of {X.numel()} values are 0 or negative. For data "
            "with zeros, use another family, for example 'poisson' for counts."
        )
        raise ValueError(msg)


class GLMFamily(str, Enum):
    gaussian = "gaussian"
    poisson = "poisson"
    bernoulli = "bernoulli"
    beta = "beta"
    gamma = "gamma"
    lognormal = "lognormal"
    sigmoid_beta = "sigmoid_beta"
    negative_binomial = "negative_binomial"

    def distribution(self) -> type[ExponentialFamily]:
        return {
            GLMFamily.gaussian: Gaussian,
            GLMFamily.poisson: Poisson,
            GLMFamily.bernoulli: Bernoulli,
            GLMFamily.beta: Beta,
            GLMFamily.gamma: Gamma,
            GLMFamily.lognormal: LogNormal,
            GLMFamily.sigmoid_beta: SigmoidBeta,
            GLMFamily.negative_binomial: NegativeBinomial,
        }[self]


class ExponentialFamily:
    r"""Encodes an exponential family distribution using PyTorch autodiff structures.

    ExponentialFamily corresponds to the superclass providing a backbone for
    a subclass for any exponential family distribution.
    Each subclass should contain the following methods, defined based on the
    distribution of choice (same notation as in Mourragui et al, 2023):

        - sufficient_statistics (:math:`T`)
        - natural_parametrization (:math:`\eta`)
        - log_partition (:math:`A`)
        - invert_g (:math:`g^{-1}`)
        - initialize_family_parameters: computes parameters used in other methods, e.g.,
        gene-level dispersion for Negative Binomial.

    We added a "base_measure" for the sake of completeness, but this method is not
    necessary for running GLM-PCA.
    The log-likelihood and exponential term are defined directly from the
    aforementionned methods.

    Parameters
    ----------
    family_name : str
        Name of the family.

    """

    family_name: str
    family_params: dict[str, Any]

    def __init__(
        self, family_params: dict[str, Any] | None = None, **kwargs: object
    ) -> None:
        self.family_name = "base"
        self.family_params = dict(family_params) if family_params else {}

    def sufficient_statistics(self, X: torch.Tensor) -> torch.Tensor:
        return X

    def natural_parametrization(self, theta: torch.Tensor) -> torch.Tensor:
        return theta

    def log_partition(self, theta: torch.Tensor) -> torch.Tensor | float:
        return 0.0

    def base_measure(self, X: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(X)

    def invert_g(self, X: torch.Tensor) -> torch.Tensor:
        return X

    def exponential_term(self, X: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        return torch.multiply(
            self.sufficient_statistics(X), self.natural_parametrization(theta)
        )

    def distribution(self, X: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        f = self.base_measure(X)
        expt = self.exponential_term(X, theta) - self.log_partition(theta)

        return torch.multiply(f, torch.exp(expt))

    def log_distribution(self, X: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        f = self.base_measure(X)
        expt = self.exponential_term(X, theta) - self.log_partition(theta)

        return expt - torch.log(f)

    def neg_log_likelihood(self, X: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Computes negative log-likelihood between dataset X and parameters theta"""
        expt = self.exponential_term(X, theta) - self.log_partition(theta)
        return -torch.sum(expt)

    def load_family_params_to_gpu(self, device: torch.device) -> None:
        for key in self.family_params:
            value = self.family_params[key]
            if type(value) is torch.Tensor:
                self.family_params[key] = value.to(device)

    def initialize_family_parameters(self, X: torch.Tensor) -> None:
        """General method to initialize certain parameters (e.g. for Beta or Negative
        Binomial)."""


class Gaussian(ExponentialFamily):
    r"""Gaussian with standard deviation one.

    GLMPCA with Gaussian as family is equivalent to the standard PCA.
    """

    def __init__(
        self, family_params: dict[str, Any] | None = None, **kwargs: object
    ) -> None:
        self.family_name = "gaussian"
        default_family_params: dict[str, Any] = {}
        self.family_params = (
            dict(family_params) if family_params else default_family_params
        )
        self.family_params.update(kwargs)
        for key, value in default_family_params.items():
            self.family_params.setdefault(key, value)

    def sufficient_statistics(self, X: torch.Tensor) -> torch.Tensor:
        return X

    def natural_parametrization(self, theta: torch.Tensor) -> torch.Tensor:
        return theta

    def log_partition(self, theta: torch.Tensor) -> torch.Tensor:
        return torch.square(theta) / 2.0

    def base_measure(self, X: torch.Tensor) -> torch.Tensor:
        return torch.exp(-torch.square(X) / 2.0) / np.sqrt(2.0 * torch.pi)

    def invert_g(self, X: torch.Tensor) -> torch.Tensor:
        return X


class Bernoulli(ExponentialFamily):
    r"""Bernoulli distribution

    family_params of interest:
        - "max_val" (int) corresponding to the max value (replaces infinity).
        Empirically, values above 10 yield similar results.

    """

    def __init__(
        self, family_params: dict[str, Any] | None = None, **kwargs: object
    ) -> None:
        self.family_name = "bernoulli"
        default_family_params: dict[str, Any] = {"max_val": 30}
        self.family_params = (
            dict(family_params) if family_params else default_family_params
        )
        self.family_params.update(kwargs)
        for key, value in default_family_params.items():
            self.family_params.setdefault(key, value)

    def sufficient_statistics(self, X: torch.Tensor) -> torch.Tensor:
        return X

    def natural_parametrization(self, theta: torch.Tensor) -> torch.Tensor:
        return theta

    def log_partition(self, theta: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(theta)

    def base_measure(self, X: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(X)

    def invert_g(self, X: torch.Tensor) -> torch.Tensor:
        return torch.log(X / (1 - X)).clip(
            -self.family_params["max_val"], self.family_params["max_val"]
        )

    def neg_log_likelihood(self, X: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Computes negative log-likelihood between dataset X and parameters theta"""
        expt = self.exponential_term(X, theta) - self.log_partition(theta)
        return -torch.sum(expt)


class Poisson(ExponentialFamily):
    r"""Poisson distribution

    family_params of interest:
        - "m" (float): saturated parameter of zero counts is -m instead of -inf
        (Landgraf and Lee, 2020). Large values let zero counts dominate the
        projection. Defaults to 1.

    """

    def __init__(
        self, family_params: dict[str, Any] | None = None, **kwargs: object
    ) -> None:
        self.family_name = "poisson"
        default_family_params: dict[str, Any] = {"m": 1.0}
        self.family_params = (
            dict(family_params) if family_params else default_family_params
        )
        self.family_params.update(kwargs)
        for key, value in default_family_params.items():
            self.family_params.setdefault(key, value)

    def sufficient_statistics(self, X: torch.Tensor) -> torch.Tensor:
        return X

    def natural_parametrization(self, theta: torch.Tensor) -> torch.Tensor:
        return theta

    def log_partition(self, theta: torch.Tensor) -> torch.Tensor:
        return torch.exp(theta)

    def base_measure(self, X: torch.Tensor) -> torch.Tensor:
        return torch.exp(-torch.lgamma(X + 1))

    def invert_g(self, X: torch.Tensor) -> torch.Tensor:
        return torch.where(X > 0, torch.log(X), -self.family_params["m"])

    def log_distribution(self, X: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """The computation of gamma function for the base measure (h) would lead to inf,
        hence a re-design of the method."""
        log_f = torch.lgamma(X + 1)
        expt = self.exponential_term(X, theta) - self.log_partition(theta)

        return expt - log_f


class NegativeBinomial(ExponentialFamily):
    r"""Negative binomial with a dispersion per feature.

    Counts with more variance than the mean (overdispersion) are negative binomial
    rather than Poisson. With the dispersion `nu` fixed per feature, the family is
    exponential with natural parameter `theta = log(mu / (mu + nu))`, which is always
    below 0.

    family_params of interest:
        - "nu" (torch.Tensor): dispersion of each feature, computed by
        initialize_family_parameters. A large value gives a Poisson-like feature.
        - "m" (float): saturated parameter of zero counts is -m instead of -inf,
        as in Poisson. Defaults to 1.
        - "eps" (float): theta is clipped to at most -eps, because the log-partition
        is infinite at 0. Defaults to 1e-6.
        - "max_val" (float): dispersion given to a feature that is not overdispersed.
        Defaults to 1e4.
        - "method" (str): "moments" for the method of moments, or "mle" for the
        profile-likelihood estimate. The MLE fits low counts with strong overdispersion
        better, but it costs about 200 times more. Defaults to "moments".
        - "chunk_size" (int): rows per pass of the MLE, which bounds its float64
        temporaries. Defaults to 8192, and GLMPCA replaces it with its own chunk_size.

    """

    def __init__(
        self, family_params: dict[str, Any] | None = None, **kwargs: object
    ) -> None:
        self.family_name = "negative_binomial"
        default_family_params: dict[str, Any] = {
            "m": 1.0,
            "eps": 1e-6,
            "max_val": 1e4,
            "method": "moments",
            "chunk_size": 8192,
        }
        self.family_params = (
            dict(family_params) if family_params else default_family_params
        )
        self.family_params.update(kwargs)
        for key, value in default_family_params.items():
            self.family_params.setdefault(key, value)

    def sufficient_statistics(self, X: torch.Tensor) -> torch.Tensor:
        return X

    def natural_parametrization(self, theta: torch.Tensor) -> torch.Tensor:
        return theta

    def log_partition(self, theta: torch.Tensor) -> torch.Tensor:
        """`-nu log(1 - exp(theta))`, through `expm1` for the small values."""
        theta = theta.clip(max=-self.family_params["eps"])
        return -self.family_params["nu"] * torch.log(-torch.expm1(theta))

    def base_measure(self, X: torch.Tensor) -> torch.Tensor:
        nu = self.family_params["nu"]
        return torch.exp(torch.lgamma(X + nu) - torch.lgamma(nu) - torch.lgamma(X + 1))

    def invert_g(self, X: torch.Tensor) -> torch.Tensor:
        nu = self.family_params["nu"]
        return torch.where(
            X > 0, torch.log(X / (X + nu)), -self.family_params["m"]
        ).clip(max=-self.family_params["eps"])

    def initialize_family_parameters(self, X: torch.Tensor) -> None:
        """Dispersion per feature, by the method chosen in family_params["method"]."""
        method = self.family_params["method"]
        if method == "moments":
            nu = self._dispersion_by_moments(X)
        elif method == "mle":
            nu = self._dispersion_by_mle(X)
        else:
            msg = (
                f"method={method!r} is not valid for the negative binomial family. "
                f"Use 'moments' or 'mle'."
            )
            raise ValueError(msg)
        max_val = self.family_params["max_val"]
        self.family_params["nu"] = nu.clip(
            min=self.family_params["eps"], max=max_val
        ).to(X.dtype)

    def _dispersion_by_moments(self, X: torch.Tensor) -> torch.Tensor:
        """`mean^2 / (var - mean)` per feature.

        A feature whose variance does not exceed its mean carries no overdispersion,
        and gets "max_val", which makes it Poisson-like.
        """
        mean = torch.mean(X, dim=0)
        variance = torch.var(X, dim=0)
        return torch.where(
            variance > mean,
            mean.square() / (variance - mean).clip(min=1e-12),
            self.family_params["max_val"],
        )

    def _dispersion_by_mle(self, X: torch.Tensor) -> torch.Tensor:
        """Profile-likelihood dispersion per feature, by bisection on `log nu`.

        With the mean of a feature fixed at its sample mean, the score of the
        log-likelihood in the dispersion `r` is

            sum_i [digamma(x_i + r) - digamma(r)] - n log1p(mean / r),

        which falls with `r` and crosses 0 at the maximum. Both terms are about
        `n·mean/r` and cancel almost completely, so the sums run in float64: in float32
        only the rounding error is left, and every feature looks Poisson-like.

        A feature whose score stays positive at "max_val" carries no evidence of
        overdispersion and gets "max_val".
        """
        n = X.shape[0]
        chunk = int(self.family_params["chunk_size"])
        max_val = float(self.family_params["max_val"])
        blocks = range(0, n, chunk)

        total = torch.zeros(X.shape[1], dtype=torch.float64, device=X.device)
        for start in blocks:
            total += X[start : start + chunk].double().sum(dim=0)
        mean = total / n

        def score(nu: torch.Tensor) -> torch.Tensor:
            accumulated = torch.zeros_like(mean)
            for start in blocks:
                block = X[start : start + chunk].double()
                accumulated += (torch.digamma(block + nu) - torch.digamma(nu)).sum(
                    dim=0
                )
            return accumulated - n * torch.log1p(mean / nu)

        low = torch.full_like(mean, 1e-3)
        high = torch.full_like(mean, max_val)
        saturated = score(high) > 0
        # 30 halvings of the interval [1e-3, max_val] in logs leave a relative error
        # below 1e-7, which is far under the sampling error of the estimate itself.
        for _ in range(30):
            middle = ((low.log() + high.log()) / 2).exp()
            positive = score(middle) > 0
            low = torch.where(positive, middle, low)
            high = torch.where(positive, high, middle)
        nu = ((low.log() + high.log()) / 2).exp()
        return torch.where(saturated, torch.full_like(nu, max_val), nu)

    def log_distribution(self, X: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """The base measure overflows for large counts, so it is kept in logs."""
        nu = self.family_params["nu"]
        log_h = torch.lgamma(X + nu) - torch.lgamma(nu) - torch.lgamma(X + 1)
        return self.exponential_term(X, theta) - self.log_partition(theta) + log_h


class Beta(ExponentialFamily):
    r"""Beta distribution, using a standard formulation.

    Original formulation presented in [Mourragui et al, 2023].

    family_params of interest:
        - "min_val" (int): min data value (replaces 0 and 1).
        - "n_jobs" (int): number of jobs, specifically for computing the "nu" parameter.
        - "method" (str): method use to compute the "nu" parameter per feature.
        Two possibles: "MLE" and "MM". Defaults to "MLE".
        - "eps" (float): minimum difference used for inverting the g function.
        Defaults to 1e-4
        - "maxiter" (int): maximum number of iterations for the inversion of the
        g function. Defaults to 100.

    """

    def __init__(
        self, family_params: dict[str, Any] | None = None, **kwargs: object
    ) -> None:
        self.family_name = "beta"
        if family_params is None or "nu" not in family_params:
            print("Beta distribution not initialized yet")
        default_family_params: dict[str, Any] = {
            "min_val": 1e-5,
            "n_jobs": 1,
            "eps": 1e-4,
            "maxiter": 100,
            "method": "MLE",
        }
        self.family_params = (
            dict(family_params) if family_params else default_family_params
        )
        self.family_params.update(kwargs)
        for key, value in default_family_params.items():
            self.family_params.setdefault(key, value)

    def sufficient_statistics(self, X: torch.Tensor) -> torch.Tensor:
        X = X.clip(self.family_params["min_val"], 1 - self.family_params["min_val"])
        return torch.stack([torch.log(X), torch.log(1 - X)])

    def natural_parametrization(self, theta: torch.Tensor) -> torch.Tensor:
        nat_params = torch.stack([
            theta * self.family_params["nu"],
            (1 - theta) * self.family_params["nu"],
        ])
        if nat_params.shape[1] == 1:
            nat_params = nat_params.flatten()
        return nat_params

    def base_measure(self, X: torch.Tensor) -> torch.Tensor:
        return torch.mul(X, 1 - X)

    def log_partition(self, theta: torch.Tensor) -> torch.Tensor:
        numerator = torch.sum(torch.lgamma(self.natural_parametrization(theta)), dim=0)
        denominator = torch.lgamma(self.family_params["nu"])
        return numerator - denominator

    def exponential_term(self, X: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        return torch.sum(
            torch.multiply(
                self.sufficient_statistics(X), self.natural_parametrization(theta)
            ),
            dim=0,
        )

    def _derivative_neg_log_likelihood(
        self, X: torch.Tensor, theta: torch.Tensor
    ) -> torch.Tensor:
        X = X.clip(self.family_params["eps"], 1 - self.family_params["eps"])
        return (
            torch.log(X / (1 - X))
            + torch.digamma((1 - theta) * self.family_params["nu"])
            - torch.digamma(theta * self.family_params["nu"])
        )

    def initialize_family_parameters(self, X: torch.Tensor) -> None:
        p = X.shape[1]
        values = X.cpu().numpy()

        def compute_beta_param(x: np.ndarray) -> tuple[float, ...]:
            y = x[x > self.family_params["eps"]]
            y = y[y < 1 - self.family_params["eps"]]
            return scipy.stats.beta.fit(
                y, floc=0, fscale=1, method=self.family_params["method"]
            )

        self.family_params["nu"] = torch.Tensor(
            _fit_columns(
                compute_beta_param,
                (values[:, idx] for idx in range(p)),
                p,
                self.family_params["n_jobs"],
            )
        )
        self.family_params["nu"] = torch.sum(self.family_params["nu"][:, :2], dim=1)
        self.family_params["nu"] = self.family_params["nu"].to(X.device)
        assert self.family_params["nu"].shape[0] == p

    def invert_g(self, X: torch.Tensor) -> torch.Tensor:
        """Dichotomy to find where derivative maxes out"""

        X = X.clip(self.family_params["eps"], 1 - self.family_params["eps"])

        # Initialize dichotomy parameters.
        min_val = torch.zeros_like(X)
        max_val = torch.ones_like(X)
        theta = (min_val + max_val) / 2

        llik = self._derivative_neg_log_likelihood(X, theta)
        for idx in tqdm(range(self.family_params["maxiter"])):
            min_val[llik > 0] = theta[llik > 0]
            max_val[llik < 0] = theta[llik < 0]
            theta = (min_val + max_val) / 2
            llik = self._derivative_neg_log_likelihood(X, theta)

            if torch.max(torch.abs(llik)) < self.family_params["eps"]:
                print(f"CONVERGENCE AFTER {idx} ITERATIONS")
                break

        if idx == self.family_params["maxiter"] - 1:
            print("CONVERGENCE NOT REACHED")

        return theta


class SigmoidBeta(Beta):
    r"""Beta distribution re-parametrized using a Sigmoid.

    This distribution is similar to the previous Beta (which it
    inherits from) but the natural parameter is re-parametrized using
    a Sigmoid. This is shown experimentally to stabilize the
    optimisation by removing the ]0,1[ constraint.

    family_params of interest:
        - "min_val" (int): min data value (replaces 0 and 1).
        - "n_jobs" (int): number of jobs, specifically for computing the "nu" parameter.
        - "method" (str): method use to compute the "nu" parameter per feature.
        Two possibles: "MLE" and "MM". Defaults to "MLE".
        - "eps" (float): minimum difference used for inverting the g function.
        Defaults to 1e-4
        - "maxiter" (int): maximum number of iterations for the inversion of the
        g function. Defaults to 100.

    """

    def natural_parametrization(self, theta: torch.Tensor) -> torch.Tensor:
        nat_params = torch.stack([
            torch.sigmoid(theta) * self.family_params["nu"],
            (1 - torch.sigmoid(theta)) * self.family_params["nu"],
        ])
        if nat_params.shape[1] == 1:
            nat_params = nat_params.flatten()
        return nat_params

    def _derivative_neg_log_likelihood(
        self, X: torch.Tensor, theta: torch.Tensor
    ) -> torch.Tensor:
        X = X.clip(self.family_params["eps"], 1 - self.family_params["eps"])
        return (
            torch.log(X / (1 - X))
            + torch.digamma((1 - theta) * self.family_params["nu"])
            - torch.digamma(theta * self.family_params["nu"])
        )

    def invert_g(self, X: torch.Tensor) -> torch.Tensor:
        """Dichotomy to find where derivative maxes out"""

        X = X.clip(self.family_params["eps"], 1 - self.family_params["eps"])

        # Initialize dichotomy parameters.
        min_val = torch.zeros_like(X)
        max_val = torch.ones_like(X)
        logit_theta = (min_val + max_val) / 2

        llik = self._derivative_neg_log_likelihood(X, logit_theta)
        for idx in tqdm(range(self.family_params["maxiter"])):
            min_val[llik > 0] = logit_theta[llik > 0]
            max_val[llik < 0] = logit_theta[llik < 0]
            logit_theta = (min_val + max_val) / 2
            llik = self._derivative_neg_log_likelihood(X, logit_theta)

            if torch.max(torch.abs(llik)) < self.family_params["eps"]:
                print(f"CONVERGENCE AFTER {idx} ITERATIONS")
                break

        if idx == self.family_params["maxiter"] - 1:
            print("CONVERGENCE NOT REACHED")

        return torch.logit(logit_theta)


class Gamma(ExponentialFamily):
    r"""Gamma distribution using a standard formulation.

    Original formulation presented in [Mourragui et al, 2023].

    family_params of interest:
        - "max_val" (int): max data value. Defaults to 1e7.
        - "n_jobs" (int): number of jobs, specifically for computing the "nu" parameter.
        - "eps" (float): minimum difference used for inverting the g function.
        Defaults to 1e-4
        - "maxiter" (int): maximum number of iterations for the inversion of the
        g function. Defaults to 100.

    """

    def __init__(
        self, family_params: dict[str, Any] | None = None, **kwargs: object
    ) -> None:
        self.family_name = "gamma"
        if family_params is None or "nu" not in family_params:
            print("Gamma distribution not initialized yet")
        default_family_params: dict[str, Any] = {
            "max_val": 1e7,
            "n_jobs": 1,
            "eps": 1e-4,
            "maxiter": 100,
        }
        self.family_params = (
            dict(family_params) if family_params else default_family_params
        )
        self.family_params.update(kwargs)
        for key, value in default_family_params.items():
            self.family_params.setdefault(key, value)

    def sufficient_statistics(self, X: torch.Tensor) -> torch.Tensor:
        return torch.stack([torch.log(X), X])

    def natural_parametrization(self, theta: torch.Tensor) -> torch.Tensor:
        nat_params = torch.stack([
            theta,
            -torch.ones_like(theta) * self.family_params["nu"],
        ])
        if nat_params.shape[1] == 1:
            nat_params = nat_params.flatten()
        return nat_params

    def log_partition(self, theta: torch.Tensor) -> torch.Tensor:
        return torch.lgamma(theta + 1) - (theta + 1) * torch.log(
            self.family_params["nu"]
        )

    def exponential_term(self, X: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        return torch.sum(
            torch.multiply(
                self.sufficient_statistics(X), self.natural_parametrization(theta)
            ),
            dim=0,
        )

    def _digamma_implicit_function(
        self, X: torch.Tensor, theta: torch.Tensor
    ) -> torch.Tensor:
        return torch.digamma(theta + 1) - torch.log(X * self.family_params["nu"])

    def invert_g(self, X: torch.Tensor) -> torch.Tensor:
        """Dichotomy to compute inverse of digamma function."""

        # Initialize dichotomy parameters.
        min_val = torch.zeros_like(X)
        max_val = torch.ones_like(X) * self.family_params["max_val"]
        theta = (min_val + max_val) / 2

        llik = self._digamma_implicit_function(X, theta)
        for idx in tqdm(range(self.family_params["maxiter"])):
            max_val[llik > 0] = theta[llik > 0]
            min_val[llik < 0] = theta[llik < 0]
            theta = (min_val + max_val) / 2
            llik = self._digamma_implicit_function(X, theta)

            if torch.max(torch.abs(llik)) < self.family_params["eps"]:
                print(f"CONVERGENCE AFTER {idx} ITERATIONS")
                break

        if idx == self.family_params["maxiter"] - 1:
            print("CONVERGENCE NOT REACHED")

        return theta

    def initialize_family_parameters(self, X: torch.Tensor) -> None:
        _require_positive(X, "Gamma")

        p = X.shape[1]
        values = X.cpu().numpy()

        self.family_params["nu"] = torch.Tensor(
            _fit_columns(
                lambda column: scipy.stats.gamma.fit(column, floc=0),
                (values[:, idx] for idx in range(p)),
                p,
                self.family_params["n_jobs"],
            )
        )
        self.family_params["nu"] = 1.0 / self.family_params["nu"][:, -1]
        self.family_params["nu"] = self.family_params["nu"].to(X.device)
        assert self.family_params["nu"].shape[0] == p


class LogNormal(ExponentialFamily):
    r"""Log-normal distribution using a standard formulation.

    Original formulation presented in [Mourragui et al, 2023].

    family_params of interest:
        - "min_val" (int): min data value. Defaults to 1e-5.

    """

    def __init__(
        self, family_params: dict[str, Any] | None = None, **kwargs: object
    ) -> None:
        self.family_name = "lognormal"
        if family_params is None or "nu" not in family_params:
            print("Log Normal distribution not initialized yet")
        default_family_params: dict[str, Any] = {
            "min_val": 1e-5,
        }
        self.family_params = (
            dict(family_params) if family_params else default_family_params
        )
        self.family_params.update(kwargs)
        for key, value in default_family_params.items():
            self.family_params.setdefault(key, value)

    def sufficient_statistics(self, X: torch.Tensor) -> torch.Tensor:
        log_X = torch.log(X)
        return torch.stack([log_X, torch.square(log_X)])

    def natural_parametrization(self, theta: torch.Tensor) -> torch.Tensor:
        nat_params = torch.stack([
            theta / torch.square(self.family_params["nu"]),
            -torch.ones_like(theta) / (2 * torch.square(self.family_params["nu"])),
        ])
        if nat_params.shape[1] == 1:
            nat_params = nat_params.flatten()
        return nat_params

    def base_measure(self, X: torch.Tensor) -> torch.Tensor:
        return 1 / (np.sqrt(2 * torch.pi) * X)

    def log_partition(self, theta: torch.Tensor) -> torch.Tensor:
        return torch.square(theta) / (
            2 * torch.square(self.family_params["nu"])
        ) + torch.log(self.family_params["nu"])

    def exponential_term(self, X: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        return torch.sum(
            torch.multiply(
                self.sufficient_statistics(X), self.natural_parametrization(theta)
            ),
            dim=0,
        )

    def invert_g(self, X: torch.Tensor) -> torch.Tensor:
        return torch.log(X.clip(self.family_params["min_val"]))

    def initialize_family_parameters(self, X: torch.Tensor) -> None:
        _require_positive(X, "LogNormal")
        self.family_params["nu"] = torch.sqrt(torch.var(torch.log(X), dim=0))
