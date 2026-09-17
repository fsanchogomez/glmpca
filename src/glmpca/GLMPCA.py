from __future__ import annotations

import copy
import warnings
from typing import Any, Literal

import anndata as ad
import numpy as np
import scipy.stats
import torch
import torch.optim
from anndata.abc import CSCDataset, CSRDataset
from scipy.sparse import csc_array, csc_matrix, csr_array, csr_matrix
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from .ExponentialFamily import ExponentialFamily, GLMFamily
from .manifolds import (
    EuclideanStiefel,
    ManifoldParameter,
    RiemannianAdagrad,
    RiemannianAdam,
)

LEARNING_RATE_LIMIT = 10 ** (-10)
DEFAULT_CHUNK_ROWS = 8192
DEPTH_CORRELATION_LIMIT = 0.8


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


def _to_tensor(
    X: torch.Tensor | np.ndarray | ad.AnnData, *, copy: bool = False
) -> torch.Tensor:
    if isinstance(X, ad.AnnData):
        counts = X.X
        if isinstance(counts, CSRDataset | CSCDataset):
            counts = counts.to_memory()
        if isinstance(counts, csr_matrix | csc_matrix | csr_array | csc_array):
            counts = counts.toarray()
        return torch.from_numpy(np.ascontiguousarray(counts, dtype=np.float32))
    if isinstance(X, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
    if isinstance(X, torch.Tensor):
        return X.clone() if copy else X
    msg = (
        f"X format unrecognised: {type(X)} != torch.Tensor, np.ndarray or "
        f"anndata.AnnData"
    )
    raise ValueError(msg)


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

    n_jobs: int or None
        Number of jobs for the per-feature fits of the family parameters. If given,
        it is written to family_params["n_jobs"] and replaces any value there. Only
        "beta", "sigmoid_beta" and "gamma" use it; the other families ignore it.
        Defaults to None, which keeps the family setting.

    device: str, torch.device or None
        Device used to compute the saturated parameters and to train. None selects
        "cuda" if it is available, else "cpu". The "mps" device is not supported.
        Fitted attributes are always stored on the CPU. Defaults to None.

    chunk_size: int
        Number of rows handled at a time when fit computes the saturated parameters and
        when it scores a run on the whole dataset. It bounds the memory of those two
        steps and does not change the result. Lower it for a large dataset on a small
        machine. Defaults to 8192.

    optimizer: str
        "adagrad" for the Riemannian Adagrad of this package, or "adam" for its
        Riemannian Adam. The learning rate defaults are calibrated for Adagrad,
        Adam usually performs better with a smaller learning rate (e.g., 0.01).
        Defaults to "adagrad".

    keep_depth_pc: bool
        Whether to keep the components that follow the sequencing depth of the cells.
        With an AnnData input and keep_depth_pc False, fit drops every component whose
        absolute Spearman correlation with the total counts per cell is above 0.9, and
        records all the correlations in depth_correlations_. A tensor or ndarray input
        is never filtered. Defaults to False.

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
        n_jobs: int | None = None,
        device: str | torch.device | None = None,
        chunk_size: int = DEFAULT_CHUNK_ROWS,
        keep_depth_pc: bool = False,
        optimizer: Literal["adagrad", "adam"] = "adagrad",
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
        self.chunk_size = chunk_size
        self.keep_depth_pc = keep_depth_pc
        self.optimizer = optimizer

        self.saturated_loadings_: torch.Tensor | None = None
        # Spearman correlation of each component with the sequencing depth
        self.depth_correlations_: torch.Tensor | None = None
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
            self.exponential_family = copy.copy(family)
            self.exponential_family.family_params = dict(family.family_params)
        if n_jobs is not None:
            self.exponential_family.family_params["n_jobs"] = n_jobs
        self.exponential_family.family_params["chunk_size"] = chunk_size

    def fit(self, X: torch.Tensor | np.ndarray | ad.AnnData) -> bool:
        r"""Fits a GLM-PCA to a specific dataset.

        Parameters
        ----------
        X : torch.Tensor, np.ndarray or AnnData
            Dataset with cells in rows and features in columns.

        Returns
        -------
        bool
            Returns True if the fitting procedure was successful.

        """
        device = _resolve_device(self.device)
        if self.init not in ("spectral", "random"):
            msg = f"init={self.init!r} is not valid. Use 'spectral' or 'random'."
            raise ValueError(msg)
        if self.optimizer not in ("adagrad", "adam"):
            msg = f"optimizer={self.optimizer!r} is not valid. Use 'adagrad' or 'adam'."
            raise ValueError(msg)
        if self.chunk_size < 1:
            msg = f"chunk_size={self.chunk_size} is not valid. Use 1 or more."
            raise ValueError(msg)

        X_fit = _to_tensor(X)
        if X_fit.shape[0] < 2:
            msg = (
                f"A fit needs at least 2 rows (cells), but the input has "
                f"{X_fit.shape[0]}. Check that the data has cells in rows and "
                f"features in columns."
            )
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
        saturated_parameters = torch.empty_like(X_fit)
        for start in range(0, X_fit.shape[0], self.chunk_size):
            stop = start + self.chunk_size
            saturated_parameters[start:stop] = self.exponential_family.invert_g(
                X_fit[start:stop].to(device)
            ).cpu()

        # Initialize the learning procedure
        self.loadings_learning_scores_ = []
        self.loadings_learning_rates_ = []

        # Use saturated parameters to find loadings by projected gradient descent
        runs = []
        for _ in range(self.n_init):
            self.learning_rate_ = self.initial_learning_rate_
            runs.append(
                self._saturated_loading_iter(
                    saturated_parameters, X_fit, batch_size, device
                )
            )

        runs = [(loadings.cpu(), intercept.cpu()) for loadings, intercept in runs]
        self.exponential_family.load_family_params_to_gpu(torch.device("cpu"))

        # Select best model
        with torch.no_grad():
            training_cost = torch.stack([
                self._full_cost(loadings, intercept, X_fit, saturated_parameters)
                for loadings, intercept in runs
            ])
        best_model_idx = int(torch.argmin(training_cost))
        best_loadings, best_intercept = runs[best_model_idx]
        self.saturated_loadings_ = best_loadings.detach()
        self.saturated_intercept_ = best_intercept.detach()

        if isinstance(X, ad.AnnData):
            self._check_depth_components(X_fit, saturated_parameters)

        return True

    def _check_depth_components(
        self, X_fit: torch.Tensor, saturated_parameters: torch.Tensor
    ) -> None:
        r"""Measures every component against the sequencing depth of the cells.

        The first components of a count matrix often carry the depth of each cell and
        not its biology. This measures each component against the total counts per cell
        with a Spearman correlation, which needs no linear relation. Components above
        DEPTH_CORRELATION_LIMIT are dropped, or kept with a message when
        keep_depth_pc is True.
        """
        loadings, intercept = self.saturated_loadings_, self.saturated_intercept_
        assert loadings is not None
        assert intercept is not None

        scores = (saturated_parameters - intercept.unsqueeze(0)).matmul(loadings)
        depth = X_fit.sum(dim=1).numpy()
        correlations = torch.tensor([
            scipy.stats.spearmanr(scores[:, component].numpy(), depth).statistic
            for component in range(scores.shape[1])
        ])
        # A component that never moves has no correlation to report.
        self.depth_correlations_ = torch.nan_to_num(correlations, nan=0.0)

        keep = self.depth_correlations_.abs() <= DEPTH_CORRELATION_LIMIT
        if bool(keep.all()):
            return

        following = [
            f"{component} (Spearman {self.depth_correlations_[component]:+.2f})"
            for component in (~keep).nonzero().flatten().tolist()
        ]
        if self.keep_depth_pc:
            msg = (
                f"{len(following)} of {keep.numel()} components follow the sequencing "
                f"depth and were kept because keep_depth_pc is True: "
                f"{', '.join(following)}."
            )
            warnings.warn(msg, UserWarning, stacklevel=3)
            return

        if not bool(keep.any()):
            msg = (
                f"Every component follows the sequencing depth: "
                f"{', '.join(following)}. Pass keep_depth_pc=True to keep them, or "
                f"check the data for a depth effect that dominates the biology."
            )
            raise ValueError(msg)

        msg = (
            f"Dropped {len(following)} of {keep.numel()} components that follow the "
            f"sequencing depth: {', '.join(following)}. Pass keep_depth_pc=True to "
            f"keep them."
        )
        warnings.warn(msg, UserWarning, stacklevel=3)
        self.saturated_loadings_ = loadings[:, keep]

    def transform(self, X: torch.Tensor | np.ndarray | ad.AnnData) -> torch.Tensor:
        r"""Transforms and projects dataset X onto the principal components.

        Parameters
        ----------
        X : torch.Tensor, np.ndarray or AnnData
            Dataset with cells in rows and features in columns. An np.ndarray or an
            AnnData input is converted on the CPU, as in fit.

        Returns
        -------
        torch.Tensor
            Projected saturated parameters, on the device of the converted dataset.

        """
        loadings, intercept = self.saturated_loadings_, self.saturated_intercept_
        if loadings is None or intercept is None:
            msg = "GLMPCA is not fitted. Call fit() before transform()."
            raise RuntimeError(msg)

        X_transform = _to_tensor(X)
        device = X_transform.device
        loadings, intercept = loadings.to(device), intercept.to(device)
        self.exponential_family.load_family_params_to_gpu(device)
        saturated_parameters = self.exponential_family.invert_g(X_transform)

        # Compute intercept term
        intercept_term = intercept.unsqueeze(0)

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
        tuple[torch.Tensor, torch.Tensor]
            Projected saturated parameters (loadings, intercept).

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
        with tqdm(total=self.max_iter, unit="epoch", dynamic_ncols=True) as epochs:
            for _ in range(self.max_iter):
                epoch_costs: list[float] = []
                for batch_data, batch_parameters in train_loader:
                    cost_step = self._optim_cost(
                        loadings=_loadings,
                        intercept=_intercept,
                        batch_data=batch_data.to(device),
                        batch_parameters=batch_parameters.to(device),
                    )

                    cost_value = cost_step.detach().cpu().numpy()
                    self.loadings_learning_scores_[-1].append(cost_value)
                    epoch_costs.append(float(cost_value))
                    cost_step.backward()
                    _optimizer.step()
                    _optimizer.zero_grad()
                    self.loadings_learning_rates_[-1].append(
                        _lr_scheduler.get_last_lr()
                    )
                learning_rate = _lr_scheduler.get_last_lr()[0]
                _lr_scheduler.step()

                # The cost is the negative log-likelihood, averaged over the batches
                # of the epoch.
                epochs.set_postfix(
                    lr=f"{learning_rate:.2e}",
                    cost=f"{np.mean(epoch_costs):.2f}",
                    refresh=False,
                )
                epochs.update(1)

                # If NaN or Inf is found in the parameters, start over optimisation with
                # reduced learning rate.
                if np.isinf(self.loadings_learning_scores_[-1][-1]) or np.isnan(
                    self.loadings_learning_scores_[-1][-1]
                ):
                    tqdm.write("\tRESTART BECAUSE INF/NAN FOUND")
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
        parameters : torch.Tensor
            Saturated parameters of the dataset X ($g^{-1}\left(X\right)$)
        X : torch.Tensor
            Dataset with cells in rows and features in columns.
        device : torch.device
            Device on which the loadings and the intercept are created.

        Returns
        -------
        optimizer: _RiemannianAdagrad
            Riemannian Adagrad optimiser instance

        loadings: ManifoldParameter
            Parameter with loadings constrained to the Stiefel manifold

        intercept: ManifoldParameter
            Parameter with the intercept.

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
                parameters[random_idx] - torch.mean(parameters[random_idx], dim=0),
                full_matrices=False,
            )
            loadings = ManifoldParameter(v[: self.n_pc, :].T.to(device))
            loadings.manifold = EuclideanStiefel()
        elif self.init == "random":
            loadings = ManifoldParameter(
                EuclideanStiefel().random(
                    parameters.shape[1], self.n_pc, device=device
                ),
                manifold=EuclideanStiefel(),
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
        # TODO: learning rate for intercept.
        tqdm.write(f"LEARNING RATE: {self.learning_rate_}")
        algorithm = RiemannianAdagrad if self.optimizer == "adagrad" else RiemannianAdam
        optimizer = algorithm(
            params=[
                {"params": loadings, "lr": self.learning_rate_},
                {"params": intercept, "lr": self.learning_rate_ * 0.01},
            ]
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.step_size, gamma=self.gamma
        )

        return optimizer, loadings, intercept, lr_scheduler

    def _full_cost(
        self,
        loadings: torch.Tensor,
        intercept: torch.Tensor,
        X: torch.Tensor,
        parameters: torch.Tensor,
    ) -> torch.Tensor:
        r"""Sums the cost over chunks of chunk_size rows, to never expand X."""
        total = torch.zeros((), dtype=X.dtype)
        for start in range(0, X.shape[0], self.chunk_size):
            stop = start + self.chunk_size
            total = total + self._optim_cost(
                loadings, intercept, X[start:stop], parameters[start:stop]
            )
        return total

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
