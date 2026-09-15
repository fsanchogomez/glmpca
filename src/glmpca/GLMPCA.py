from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal, overload

import anndata as ad
import numpy as np
import torch
import torch.optim
from geoopt import EuclideanStiefel, ManifoldParameter
from scipy.sparse import csc_array, csc_matrix, csr_array, csr_matrix
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .ExponentialFamily import ExponentialFamily, GLMFamily

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.optim.optimizer import ParamsT

LEARNING_RATE_LIMIT = 10 ** (-10)


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "mps":
        msg = (
            "The mps device is not supported: torch.linalg.qr, used at every "
            "optimisation step, is too slow on mps and can stall the GPU."
        )
        raise ValueError(msg)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        msg = f"device={device!r} was requested, but CUDA is not available."
        raise ValueError(msg)
    return resolved


class _RiemannianAdagrad(torch.optim.Optimizer):
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


class GLMPCA:
    r"""Performs GLM-PCA on a data matrix to reduce its dimensionality.

    This class computes the generalized-linear model principal components (GLM-PCs)
    of a dataset by exploiting the framework of saturated parameters. Specifically,
    given an exponential distribution chosen based on prior knowledge, GLM-PCA will find
    a collection of directions which minimize the reconstruction error, computed as the
    negative log-likelihood of the chosen exponential distribution.

    By making use of an alternative formulation, our implementation can exploit
    automatic differentiation and can therefore rely on mini-batch Stochastic
    Gradient Descent. As a consequence, it scales to large datasets.

    Another interesting feature of our implementation is that it does not require
    any cumbersome Lagrangian derivations. If you wish to test an exponential family
    distribution not present in our implementation, you may add it by creating a
    subclass of `ExponentialFamily`, with its corresponding density function. This
    would suffice to use it for GLM-PCA.

    Parameters
    ----------
    n_pc : int
        Number of principal components to compute.

    family: str or ExponentialFamily
        Name of the exponential distribution to use. Possible families: "gaussian",
        "poisson", "bernoulli", "beta", "gamma", "lognormal", "sigmoid_beta".
        Defaults to "gaussian".

    family_params : dict
        Dictionary with additional exponential distribution parameters. The list of
        parameters depends on the specific `ExponentialFamily` class chosen.

        - "n_jobs" (int) for parallelization, specifically for "beta" and "gamma".
        - "min_val" (float) for truncating in "beta".
        - "m" (float) for the saturated parameter of zero counts in "poisson".
        - "eps" (float) for convergence in inverse computation in "beta".

        Defaults to None.

    max_iter : int
        Maximum number of epochs in the GLM-PCA optimisation. Defaults to 100.

    learning_rate: float
        Learning rate to be used in the GLM-PCA optimisation. If learning_rate is too
        high and lead to NaN, our implementation automatically restarts the optimisation
        with a smaller value. Defaults to 0.2.

    batch_size : int
        Size of the batch in the SGD optimisation step. If the matrix to fit has
        fewer rows, the number of rows is used instead and a warning is issued.
        Defaults to 256.

    step_size: int
        Step size in optimiser scheduler. Defaults to 20.
        See more: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html

    gamma: float
        Reduction parameter for optimiser scheduler. Defaults to 0.5.
        See more: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html


    n_init: int
        Number of GLM-PCA initializations. Useful if you want to explore different
        random seeds and starting points. Defaults to 1.

    init: str
        Method to initialize loadings. "spectral" performs SVD on  the saturated
        parameters from a small random batch of the dataset, "random" performs a random
        initialization on the Stiefel manifold. Defaults to "spectral".

    n_jobs: int
        Number of jobs used in parallel operations. Defaults to 1.

    device: str, torch.device or None
        Device used to compute the saturated parameters and to train. None selects
        "cuda" if it is available, else "cpu". The "mps" device is not supported.
        Fitted attributes are always stored on the CPU. Defaults to None.

    """

    def __init__(
        self,
        n_pc: int,
        family: str | ExponentialFamily = "gaussian",
        family_params: dict[str, Any] | None = None,
        max_iter: int = 100,
        learning_rate: float = 0.2,
        batch_size: int = 256,
        step_size: int = 20,
        gamma: float = 0.5,
        n_init: int = 1,
        init: Literal["spectral", "random"] = "spectral",
        n_jobs: int = 1,
        device: str | torch.device | None = None,
    ) -> None:
        self.n_pc = n_pc
        self.family = family
        self.family_params = family_params
        self.log_part_theta_matrices_ = None
        self.max_iter = np.abs(max_iter)
        self.learning_rate_ = learning_rate
        self.initial_learning_rate_ = learning_rate
        self.n_jobs = n_jobs
        self.batch_size = batch_size
        self.n_init = n_init
        self.gamma = gamma
        self.step_size = step_size
        self.init = init

        self.saturated_loadings_: torch.Tensor | None = None
        # saturated_intercept_: before projecting
        self.saturated_intercept_: torch.Tensor | None = None
        # reconstruction_intercept: after projecting
        self.reconstruction_intercept_ = None

        # Whether to perform sample or gene projection
        self.sample_projection = False

        self.exp_family_params = None
        self.loadings_learning_scores_ = []
        self.loadings_learning_rates_ = []

        # Initialize device
        self.device = device

        # Set up exponential family
        if isinstance(family, str):
            self.exponential_family = GLMFamily(family).distribution()(
                self.family_params
            )
        else:
            self.exponential_family = family

    def fit(self, X: torch.Tensor | np.ndarray | ad.AnnData) -> bool:
        r"""Fits a GLM-PCA to a specific dataset.

        Parameters
        ----------
        X : torch.Tensor, np.ndarray or AnnData
            Dataset with cells in rows and features in columns.

        Returns
        -------
        bool
            Returns True if the fitting procedure was been successful.

        """
        device = _resolve_device(self.device)

        if isinstance(X, ad.AnnData):
            counts = X.X
            if isinstance(counts, csr_matrix | csc_matrix | csr_array | csc_array):
                counts = counts.toarray()
            X_fit = torch.Tensor(np.asarray(counts).T)
        elif isinstance(X, np.ndarray):
            X_fit = torch.Tensor(X)
        elif isinstance(X, torch.Tensor):
            X_fit = X.clone()
        else:
            msg = f"X format unrecognised: {type(X)} != np.ndarray or torch.Tensor"
            raise ValueError(msg)

        batch_size = self.batch_size
        if X_fit.shape[0] < batch_size:
            msg = (
                f"batch_size={batch_size} is larger than the number of observations "
                f"to fit ({X_fit.shape[0]}). Using batch_size={X_fit.shape[0]}."
            )
            warnings.warn(msg, UserWarning, stacklevel=2)
            batch_size = X_fit.shape[0]

        # Fit exponential family params (e.g., dispersion for negative binomial)
        self.exponential_family.initialize_family_parameters(X_fit)
        self.exponential_family.load_family_params_to_gpu(device)

        # Compute saturated parameters, alongside exponential family parameters
        saturated_parameters = self.exponential_family.invert_g(X_fit.to(device)).cpu()

        # Initialize the learning procedure
        self.learning_rate_ = self.initial_learning_rate_
        self.loadings_learning_scores_ = []
        self.loadings_learning_rates_ = []

        # Use saturated parameters to find loadings by projected gradient descent
        runs = [
            self._saturated_loading_iter(
                saturated_parameters, X_fit, batch_size, device
            )
            for _ in range(self.n_init)
        ]

        runs = [(loadings.cpu(), intercept.cpu()) for loadings, intercept in runs]
        self.exponential_family.load_family_params_to_gpu(torch.device("cpu"))

        # Select best model
        with torch.no_grad():
            training_cost = torch.stack([
                self._optim_cost(loadings, intercept, X_fit, saturated_parameters)
                for loadings, intercept in runs
            ])
        best_model_idx = int(torch.argmin(training_cost))
        self.saturated_loadings_, self.saturated_intercept_ = runs[best_model_idx]

        return True

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        r"""Transforms and projects dataset X onto the principal components.

        Parameters
        ----------
        X : torch.Tensor
            Dataset with cells in rows and features in columns.

        Returns
        -------
        torch.Tensor
            Projected saturated parameters.

        """
        loadings, intercept = self.saturated_loadings_, self.saturated_intercept_
        if loadings is None or intercept is None:
            msg = "GLMPCA is not fitted. Call fit() before transform()."
            raise RuntimeError(msg)

        loadings, intercept = loadings.to(X.device), intercept.to(X.device)
        self.exponential_family.load_family_params_to_gpu(X.device)
        saturated_parameters = self.exponential_family.invert_g(X)

        # Compute intercept term
        n = X.shape[0]
        intercept_term = intercept.unsqueeze(0).repeat(n, 1)

        projected_parameters = saturated_parameters - intercept_term
        projected_parameters = projected_parameters.matmul(loadings)

        return projected_parameters

    def _saturated_loading_iter(
        self,
        saturated_parameters: torch.Tensor,
        X: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Computes the loadings solution of the GLM-PCA optimisation problem.

        Parameters
        ----------
        saturated_parameters : torch.Tensor
            Saturated parameters of the dataset X ($g^{-1}\left(X\right)$)
        X : torch.Tensor
            Dataset with cells in rows and features in columns.
        batch_size : int
            Size of the batch in the SGD optimisation step.
        device : torch.device
            Device used to train.

        Returns
        -------
        torch.Tensor
            Projected saturated parameters.

        """
        if self.learning_rate_ < LEARNING_RATE_LIMIT:
            msg = "LEARNING RATE IS TOO SMALL : DID NOT CONVERGE"
            raise ValueError(msg)

        # Set up list of saving scores
        self.loadings_learning_scores_.append([])
        self.loadings_learning_rates_.append([])

        _optimizer, _loadings, _intercept, _lr_scheduler = (
            self._create_saturated_loading_optim(
                parameters=saturated_parameters.data.clone(), X=X, device=device
            )
        )

        # Load dataset
        train_data = TensorDataset(X, saturated_parameters.data.clone())
        train_loader = DataLoader(
            dataset=train_data, batch_size=batch_size, shuffle=True, drop_last=True
        )

        # Run epoch in a for loop
        self._loadings_epochs = [_loadings.clone().detach()]
        self._intercept_epochs = [_intercept.clone().detach()]
        for _ in tqdm(range(self.max_iter)):
            for batch_data, batch_parameters in train_loader:
                cost_step = self._optim_cost(
                    loadings=_loadings,
                    intercept=_intercept,
                    batch_data=batch_data.to(device),
                    batch_parameters=batch_parameters.to(device),
                )

                self.loadings_learning_scores_[-1].append(
                    cost_step.detach().cpu().numpy()
                )
                cost_step.backward()
                _optimizer.step()
                _optimizer.zero_grad()
                self.loadings_learning_rates_[-1].append(_lr_scheduler.get_last_lr())
            _lr_scheduler.step()

            self._loadings_epochs.append(_loadings.clone().detach())
            self._intercept_epochs.append(_intercept.clone().detach())

            # If NaN or Inf is found in the parameters, start over optimisation with
            # reduced learning rate.
            if np.isinf(self.loadings_learning_scores_[-1][-1]) or np.isnan(
                self.loadings_learning_scores_[-1][-1]
            ):
                print("\tRESTART BECAUSE INF/NAN FOUND", flush=True)
                self.learning_rate_ = self.learning_rate_ * self.gamma
                self.loadings_learning_scores_ = self.loadings_learning_scores_[:-1]
                self.loadings_learning_rates_ = self.loadings_learning_rates_[:-1]

                # Remove memory
                del (
                    train_data,
                    train_loader,
                    _optimizer,
                    _loadings,
                    _intercept,
                    _lr_scheduler,
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()

                self._loadings_epochs = []
                self._intercept_epochs = []

                return self._saturated_loading_iter(
                    saturated_parameters=saturated_parameters,
                    X=X,
                    batch_size=batch_size,
                    device=device,
                )

        return (_loadings, _intercept)

    def _create_saturated_loading_optim(
        self, parameters: torch.Tensor, X: torch.Tensor, device: torch.device
    ) -> tuple[
        torch.optim.Optimizer,
        torch.Tensor,
        torch.Tensor,
        torch.optim.lr_scheduler.StepLR,
    ]:
        r"""Initializes the optimisation problem.

        Parameters
        ----------
        saturated_parameters : torch.Tensor
            Saturated parameters of the dataset X ($g^{-1}\left(X\right)$)
        X : torch.Tensor
            Dataset with cells in rows and features in columns.
        device : torch.device
            Device on which the loadings and the intercept are created.

        Returns
        -------
        optimizer: _RiemannianAdagrad
            Riemannian Adagrad optimiser instance

        loadings: geoopt.ManifoldParameter
            geoopt parameter with loadings constrained to the Stiefel manifold

        intercept: geoopt.ManifoldParameter
            geoopt parameter with intercept.

        lr_scheduler: torch.optim.scheduler
            Scheduler instance.

        """
        # Initialize loadings with spectrum (2**13 as maximum value for SVD to be
        # relatively fast)
        random_batch_size = min(X.shape[0], 2**13)
        random_idx = np.random.choice(
            np.arange(parameters.shape[0]), replace=False, size=random_batch_size
        )
        if self.init == "spectral":
            _, _, v = torch.linalg.svd(
                parameters[random_idx] - torch.mean(parameters[random_idx], dim=0)
            )
            loadings = ManifoldParameter(v[: self.n_pc, :].T.to(device))
            loadings.manifold = EuclideanStiefel()
        elif self.init == "random":
            loadings = ManifoldParameter(
                EuclideanStiefel().random(parameters.shape[1], self.n_pc, device=device)
            )

        # Initialize intercept
        if self.exponential_family.family_name in ["poisson"]:
            intercept = ManifoldParameter(
                torch.median(parameters[random_idx], dim=0).values.to(device)
            )
        else:
            intercept = ManifoldParameter(
                torch.mean(parameters[random_idx], dim=0).to(device)
            )

        # Create optimizer
        # TODO: allow for other optimizer to be used.
        # TODO: learning rate for intercept.
        print(f"LEARNING RATE: {self.learning_rate_}")
        optimizer = _RiemannianAdagrad(
            params=[
                {"params": loadings, "lr": self.learning_rate_},
                {"params": intercept, "lr": self.learning_rate_ * 0.01},
            ]
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.step_size, gamma=self.gamma
        )

        return optimizer, loadings, intercept, lr_scheduler

    def _optim_cost(
        self,
        loadings: torch.Tensor,
        intercept: torch.Tensor,
        batch_data: torch.Tensor,
        batch_parameters: torch.Tensor,
    ) -> torch.Tensor:
        n = batch_data.shape[0]
        intercept_term = intercept.unsqueeze(0).repeat(n, 1)

        projected_parameters = batch_parameters - intercept_term
        projected_parameters = projected_parameters.matmul(loadings).matmul(loadings.T)
        projected_parameters = projected_parameters + intercept_term

        return self.exponential_family.neg_log_likelihood(
            batch_data, projected_parameters
        )
