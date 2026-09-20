from __future__ import annotations

import copy
import warnings
from typing import TYPE_CHECKING, Any, Literal

import anndata as ad
import numpy as np
import torch
import torch.optim
from anndata.abc import CSCDataset, CSRDataset
from scipy.sparse import csc_array, csc_matrix, csr_array, csr_matrix
from tqdm.auto import tqdm

from .ExponentialFamily import ExponentialFamily, GLMFamily
from .manifolds import (
    Grassmann,
    ManifoldParameter,
    RiemannianAdagrad,
    RiemannianAdam,
    RiemannianConjugateGradient,
    canonical_basis,
)
from .sparse import SparseRows

if TYPE_CHECKING:
    from collections.abc import Iterator

_OPTIMIZERS = {
    "adagrad": RiemannianAdagrad,
    "adam": RiemannianAdam,
    "cg": RiemannianConjugateGradient,
}

DEFAULT_LEARNING_RATE = 0.2
SPARSE_FAMILIES = ("gaussian", "poisson", "negative_binomial", "bernoulli")
INTERCEPT_RATE_SCALE = 0.01
DEPTH_RATE_SCALE = 0.01
LEARNING_RATE_LIMIT = 1e-8
PLATEAU_PATIENCE = 10
PLATEAU_THRESHOLD = 1e-4
DEFAULT_CHUNK_ROWS = 8192
OFFSET_NEWTON_ITERATIONS = 50
OFFSET_HALVINGS = 30
OFFSET_TOLERANCE = 1e-6
DEVICE_MEMORY_SHARE = 0.8
WORKING_COPIES = 8


def _announce_device(device: torch.device) -> None:
    """Prints the device on which the fit will be run."""
    if device.type == "cuda":
        tqdm.write(f"DEVICE: {device} ({torch.cuda.get_device_name(device)})")
    else:
        tqdm.write(f"DEVICE: {device}")


def _fits_in(free_bytes: int, tensors: tuple[torch.Tensor, ...], working: int) -> bool:
    """Whether `tensors` and `working` more bytes fit in a share of `free_bytes`."""
    needed = sum(tensor.nbytes for tensor in tensors) + working
    return needed <= DEVICE_MEMORY_SHARE * free_bytes


def _depth_of(centred: torch.Tensor, loadings: torch.Tensor) -> torch.Tensor:
    r"""The offset of every cell that least squares would give, once `1 mu.T` is out.

    The value that leaves the least outside the subspace is

        s = <w, centred> / <w, 1>,    w = (I - V V.T) 1,

    which is the least-squares solution of `min_s ||(centred - s 1) (I - V V.T)||`. It
    is the start of `_fitted_depth`, which moves it to the optimum of the likelihood.
    """
    ones = torch.ones(loadings.shape[0], device=loadings.device, dtype=loadings.dtype)
    outside = ones - loadings @ (loadings.T @ ones)
    scale = float(outside @ ones)
    if abs(scale) < 1e-8:
        # The all-ones direction lies in the subspace, which already carries the offset.
        return torch.zeros(centred.shape[0], device=centred.device, dtype=centred.dtype)
    return (centred @ outside) / scale


def _row_costs(
    family: ExponentialFamily, data: torch.Tensor, theta: torch.Tensor
) -> torch.Tensor:
    """The negative log-likelihood of every row, without `log h`."""
    return -(family.exponential_term(data, theta) - family.log_partition(theta)).sum(
        dim=1, dtype=torch.float64
    )


def _fitted_depth(
    family: ExponentialFamily,
    data: torch.Tensor,
    centred: torch.Tensor,
    intercept: torch.Tensor,
    loadings: torch.Tensor,
) -> torch.Tensor:
    r"""The offset of every cell that the likelihood gives, the rest held fixed.

    With `P = V V.T` and `w = (I - P) 1`, the fitted parameters of a cell are

        theta_hat = centred P + mu + s w,

    linear in its offset `s`, so every cell is a 1-D problem. Newton steps from the
    least-squares offset solve it, with the step of a cell halved until its cost does
    not rise. A cell leaves the loop once its offset stops moving. `fit` gives its own
    cells their offsets this way after the fit, and `transform` gives new cells theirs,
    so both use the same offset for the same cell.
    """
    depth = _depth_of(centred, loadings)
    ones = torch.ones(loadings.shape[0], device=loadings.device, dtype=loadings.dtype)
    outside = ones - loadings @ (loadings.T @ ones)
    if abs(float(outside @ ones)) < 1e-8:
        return depth
    inside = centred @ loadings @ loadings.T + intercept.unsqueeze(0)
    outside = outside.unsqueeze(0)
    active = torch.arange(depth.shape[0], device=depth.device)
    for _ in range(OFFSET_NEWTON_ITERATIONS):
        rows, base, start = data[active], inside[active], depth[active]
        with torch.enable_grad():
            trial = start.clone().requires_grad_(True)
            cost = _row_costs(family, rows, base + trial.unsqueeze(1) * outside)
            (gradient,) = torch.autograd.grad(cost.sum(), trial, create_graph=True)
            (curvature,) = torch.autograd.grad(gradient.sum(), trial)
        cost, gradient = cost.detach(), gradient.detach()
        step = torch.where(curvature > 0, gradient / curvature, gradient)
        scale = torch.ones_like(step)
        pending = torch.arange(step.shape[0], device=step.device)
        for _ in range(OFFSET_HALVINGS):
            moved = start[pending] - scale[pending] * step[pending]
            worse = ~(
                _row_costs(
                    family, rows[pending], base[pending] + moved.unsqueeze(1) * outside
                )
                <= cost[pending]
            )
            pending = pending[worse]
            if pending.numel() == 0:
                break
            scale[pending] = scale[pending] / 2
        else:
            scale[pending] = 0.0
        change = scale * step
        depth[active] = start - change
        still = change.abs() > OFFSET_TOLERANCE * (1.0 + depth[active].abs())
        active = active[still]
        if active.numel() == 0:
            break
    return depth


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


def _to_sparse(X: torch.Tensor | np.ndarray | ad.AnnData) -> SparseRows:
    """The input as a float32 CSR matrix, for keep_sparse=True."""
    if isinstance(X, ad.AnnData):
        counts = X.X
        if isinstance(counts, CSRDataset | CSCDataset):
            counts = counts.to_memory()
    elif isinstance(X, torch.Tensor):
        counts = X.detach().cpu().numpy()
    elif isinstance(X, np.ndarray):
        counts = X
    else:
        msg = (
            f"X format unrecognised: {type(X)} != torch.Tensor, np.ndarray or "
            f"anndata.AnnData"
        )
        raise ValueError(msg)
    return SparseRows(csr_matrix(counts, dtype=np.float32))


class _Rows:
    """The matrix of a fit and its saturated parameters, read by blocks of rows.

    `data` is a tensor, or a `SparseRows` with keep_sparse. `theta` holds the saturated
    parameters when they were computed once, or is None, and then every block computes
    its own from its data. That is what keeps a CSR input from being densified whole.
    """

    def __init__(
        self,
        data: torch.Tensor | SparseRows,
        family: ExponentialFamily,
        theta: torch.Tensor | None,
    ) -> None:
        self.data = data
        self.family = family
        self.theta = theta

    @property
    def shape(self) -> torch.Size:
        return self.data.shape

    def block(
        self, rows: slice | torch.Tensor, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The data and the saturated parameters of `rows`, on `device`."""
        data = self.data[rows].to(device, non_blocking=True)
        if self.theta is None:
            return data, self.family.invert_g(data)
        return data, self.theta[rows].to(device, non_blocking=True)

    def slices(self, size: int) -> Iterator[slice]:
        for start in range(0, self.shape[0], size):
            yield slice(start, start + size)


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
        - "eps" (float) for convergence in inverse computation in "beta".

        Defaults to None.

    max_iter : int
        Maximum number of epochs in the GLM-PCA optimisation. Defaults to 100.

    learning_rate: float
        Learning rate to be used in the GLM-PCA optimisation. If learning_rate is too
        high and leads to NaN, our implementation automatically restarts the
        optimisation with a smaller value. Defaults to 0.2.

        The scheduler lowers this rate when the cost stops falling (see gamma) and
        never takes it below LEARNING_RATE_LIMIT. A fit that reaches that floor stops
        with a warning, because the steps no longer move the loadings.

    batch_size : int
        Size of the batch in the SGD optimisation step. If the matrix to fit has
        fewer rows, the number of rows is used instead and a warning is issued.
        Defaults to 256.

    gamma: float
        Factor that multiplies the learning rate when the cost reaches a plateau, that
        is, when PLATEAU_PATIENCE epochs pass without the cost of an epoch falling by
        PLATEAU_THRESHOLD of the cost that the fit had when the rate last fell.
        Defaults to 0.5.
        See more: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html

    n_init: int
        Number of GLM-PCA initializations. Useful if you want to explore different
        random seeds and starting points. Defaults to 1.

    init: str
        Method to initialize loadings. "spectral" performs SVD on the saturated
        parameters from a small random batch of the dataset, and "random" performs a
        random initialization on the Stiefel manifold. Defaults to "spectral".

    depth_factor: bool
        Whether to fit an offset for every cell beside the offset of every feature, as
        `fast_poisson` does for its size factor. The saturated parameters are centred
        by both offsets before the projection, so a component never has to carry the
        depth of a cell. It has its own learning rate, DEPTH_RATE_SCALE of
        learning_rate, as the intercept does, and `transform` gives an unseen cell the
        offset that least squares would give it. Defaults to True.

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
        "adagrad" for the Riemannian Adagrad of this package, "adam" for its Riemannian
        Adam, or "cg" for its Riemannian conjugate gradients. The learning rate defaults
        are calibrated for Adagrad, Adam usually performs better with a smaller learning
        rate (e.g., 0.01). Defaults to "adagrad".

        "cg" is different in kind: it works on the whole matrix rather than on
        mini-batches, and it has no learning rate, because it chooses every step by a
        line search. learning_rate, gamma and the scheduler do not reach it, and
        batch_size does not either. One of its steps costs several passes over the
        matrix, and it lowers the cost at every one of them.

    keep_sparse: bool
        Whether to hold the input as a CSR matrix for the whole fit, instead of as a
        dense matrix beside a dense copy of its saturated parameters. Every block of
        rows is densified, and its saturated parameters computed, when it is read. The
        memory of the data falls from two dense matrices to the non-zero entries, and a
        pass over the matrix takes longer, by about half on a count matrix with 7%
        non-zeros. transform then reads its input the same way. Only the families
        whose data has zeros take it: "gaussian", "poisson", "negative_binomial" and
        "bernoulli". Defaults to False.

    """

    def __init__(
        self,
        n_pc: int,
        family: str | ExponentialFamily = "gaussian",
        family_params: dict[str, Any] | None = None,
        max_iter: int = 100,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        batch_size: int = 256,
        gamma: float = 0.5,
        n_init: int = 1,
        init: Literal["spectral", "random"] = "spectral",
        depth_factor: bool = True,
        n_jobs: int | None = None,
        device: str | torch.device | None = None,
        chunk_size: int = DEFAULT_CHUNK_ROWS,
        optimizer: Literal["adagrad", "adam", "cg"] = "adagrad",
        keep_sparse: bool = False,
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
        self.init = init
        self.depth_factor = depth_factor
        self.chunk_size = chunk_size
        self.optimizer = optimizer
        self.keep_sparse = keep_sparse

        self.saturated_loadings_: torch.Tensor | None = None
        # Log-likelihood of the fit, the real one, with the base measure
        self.log_likelihood_: float | None = None
        # saturated_intercept_: before projecting
        self.saturated_intercept_: torch.Tensor | None = None
        # saturated_depth_: before projecting
        self.saturated_depth_: torch.Tensor | None = None
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
        if self.optimizer not in _OPTIMIZERS:
            choices = ", ".join(repr(name) for name in _OPTIMIZERS)
            msg = f"optimizer={self.optimizer!r} is not valid. Use one of {choices}."
            raise ValueError(msg)
        if self.chunk_size < 1:
            msg = f"chunk_size={self.chunk_size} is not valid. Use 1 or more."
            raise ValueError(msg)
        if (
            self.optimizer == "cg"
            and self.initial_learning_rate_ != DEFAULT_LEARNING_RATE
        ):
            msg = (
                f"learning_rate={self.initial_learning_rate_} is ignored by "
                f"optimizer='cg', which has none: it chooses every step by a line "
                f"search on the whole matrix. gamma and the scheduler do not reach it "
                f"either."
            )
            warnings.warn(msg, UserWarning, stacklevel=2)

        family = self.exponential_family
        if self.keep_sparse and family.family_name not in SPARSE_FAMILIES:
            msg = (
                f"keep_sparse=True does not fit family={family.family_name!r}, whose "
                f"data has no zeros, so a sparse matrix would save nothing. Use one of "
                f"{', '.join(repr(name) for name in SPARSE_FAMILIES)}."
            )
            raise ValueError(msg)

        X_fit = _to_sparse(X) if self.keep_sparse else _to_tensor(X)
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

        _announce_device(device)

        # Fit exponential family params (e.g., dispersion for negative binomial)
        family.initialize_family_parameters(X_fit)
        family.load_family_params_to_gpu(device)

        # Compute saturated parameters, alongside exponential family parameters
        saturated_parameters = None
        if isinstance(X_fit, torch.Tensor):
            saturated_parameters = torch.empty_like(X_fit)
            for start in range(0, X_fit.shape[0], self.chunk_size):
                stop = start + self.chunk_size
                saturated_parameters[start:stop] = family.invert_g(
                    X_fit[start:stop].to(device)
                ).cpu()
        rows = _Rows(X_fit, family, saturated_parameters)

        # log h(x) of every cell, summed over its features. It turns the cost into
        # the real log-likelihood.
        log_base_measure = torch.empty(X_fit.shape[0])
        for chunk in rows.slices(self.chunk_size):
            log_base_measure[chunk] = (
                family.log_base_measure(X_fit[chunk].to(device)).sum(dim=1).cpu()
            )

        resident = False
        if isinstance(X_fit, torch.Tensor) and saturated_parameters is not None:
            resident = device.type != "cuda" or _fits_in(
                torch.cuda.mem_get_info(device)[0],
                (X_fit, saturated_parameters),
                WORKING_COPIES * min(self.chunk_size, X_fit.shape[0]) * X_fit[0].nbytes,
            )
            if resident and device.type == "cuda":
                rows = _Rows(X_fit.to(device), family, saturated_parameters.to(device))
        if device.type == "cuda":
            tqdm.write(
                f"DATA: {'held on the device' if resident else 'copied by chunks'}"
            )
        after = device if resident else torch.device("cpu")

        # Initialize the learning procedure
        self.loadings_learning_scores_ = []
        self.loadings_learning_rates_ = []

        # Use saturated parameters to find loadings by projected gradient descent
        runs = []
        for _ in range(self.n_init):
            self.learning_rate_ = self.initial_learning_rate_
            runs.append(
                self._saturated_loading_iter(rows, batch_size, device, log_base_measure)
            )

        runs = [
            (
                loadings.to(after),
                intercept.to(after),
                None if depth is None else depth.to(after),
            )
            for loadings, intercept, depth in runs
        ]
        family.load_family_params_to_gpu(after)

        # Select best model
        best_model_idx = 0
        if len(runs) > 1:
            with torch.no_grad():
                training_cost = torch.stack([
                    self._full_cost(loadings, intercept, rows, after, depth)
                    for loadings, intercept, depth in runs
                ])
            best_model_idx = int(torch.argmin(training_cost))
        best_loadings, best_intercept, best_depth = runs[best_model_idx]
        best_loadings = best_loadings.detach()
        self.saturated_intercept_ = best_intercept.detach()
        self.saturated_depth_ = None
        if best_depth is not None:
            offsets = []
            for chunk in rows.slices(self.chunk_size):
                data, theta = rows.block(chunk, after)
                offsets.append(
                    _fitted_depth(
                        family,
                        data,
                        theta - self.saturated_intercept_.unsqueeze(0),
                        self.saturated_intercept_,
                        best_loadings,
                    )
                )
            self.saturated_depth_ = torch.cat(offsets)
        with torch.no_grad():
            training_cost = self._full_cost(
                best_loadings,
                self.saturated_intercept_,
                rows,
                after,
                self.saturated_depth_,
            )
        self.log_likelihood_ = float(training_cost.neg().cpu()) + float(
            log_base_measure.sum()
        )
        self.saturated_loadings_ = canonical_basis(
            best_loadings, self._centred_chunks(rows, after)
        ).cpu()
        self.saturated_intercept_ = self.saturated_intercept_.cpu()
        if self.saturated_depth_ is not None:
            self.saturated_depth_ = self.saturated_depth_.cpu()
        family.load_family_params_to_gpu(torch.device("cpu"))

        return True

    def _centred_chunks(
        self, rows: _Rows, device: torch.device
    ) -> Iterator[torch.Tensor]:
        """The rows that transform projects, in blocks of chunk_size.

        They are centred with the offsets of `_fitted_depth`, which transform gives the
        same cells, so the scores canonical_basis orders are the scores that transform
        reports. Those offsets depend on the loadings only through their span, so the
        rotation of canonical_basis leaves them alone.
        """
        intercept = self.saturated_intercept_
        assert intercept is not None
        for chunk in rows.slices(self.chunk_size):
            _, theta = rows.block(chunk, device)
            block = theta - intercept.unsqueeze(0)
            if self.saturated_depth_ is not None:
                block = block - self.saturated_depth_[chunk].unsqueeze(1)
            yield block

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

        X_transform = _to_sparse(X) if self.keep_sparse else _to_tensor(X)
        device = X_transform.device
        loadings, intercept = loadings.to(device), intercept.to(device)
        self.exponential_family.load_family_params_to_gpu(device)

        rows = _Rows(X_transform, self.exponential_family, None)
        scores = [torch.empty(0, loadings.shape[1], device=device)]
        for chunk in rows.slices(self.chunk_size):
            data, theta = rows.block(chunk, device)
            projected_parameters = theta - intercept.unsqueeze(0)
            if self.depth_factor:
                depth = _fitted_depth(
                    self.exponential_family,
                    data,
                    projected_parameters,
                    intercept,
                    loadings,
                )
                projected_parameters = projected_parameters - depth.unsqueeze(1)
            scores.append(projected_parameters.matmul(loadings))

        return torch.cat(scores)

    def _saturated_loading_iter(
        self,
        rows: _Rows,
        batch_size: int,
        device: torch.device,
        log_base_measure: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        r"""Computes the loadings solution of the GLM-PCA optimisation problem.

        Parameters
        ----------
        rows : _Rows
            The dataset, cells in rows and features in columns, with its saturated
            parameters ($g^{-1}\left(X\right)$), read by blocks of rows.
        batch_size : int
            Size of the batch in the SGD optimisation step.
        device : torch.device
            Device used to train.
        log_base_measure : torch.Tensor
            `log h(x)` of every cell, summed over its features.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]
            The loadings, the intercept of every feature, and the offset of every cell
            when depth_factor is on, else None.

        """
        if self.learning_rate_ < LEARNING_RATE_LIMIT:
            msg = "LEARNING RATE IS TOO SMALL : DID NOT CONVERGE"
            raise ValueError(msg)

        # Set up list of saving scores
        self.loadings_learning_scores_.append([])
        self.loadings_learning_rates_.append([])

        _optimizer, _loadings, _intercept, _depth, _lr_scheduler = (
            self._create_saturated_loading_optim(rows=rows, device=device)
        )
        n = rows.shape[0]

        def closure() -> float:
            """The cost of the whole matrix, with the gradient summed over chunks."""
            _optimizer.zero_grad()
            total = torch.zeros((), device=device, dtype=torch.float64)
            for chunk in rows.slices(self.chunk_size):
                data, parameters = rows.block(chunk, device)
                cost = self._optim_cost(
                    loadings=_loadings,
                    intercept=_intercept,
                    batch_data=data,
                    batch_parameters=parameters,
                    batch_depth=None if _depth is None else _depth[chunk],
                )
                cost.backward()
                total += cost.detach()
            return float(total)

        # Run epoch in a for loop
        with tqdm(total=self.max_iter, unit="epoch", dynamic_ncols=True) as epochs:
            cost_anchor: float | None = None
            failed_searches = 0
            for _ in range(self.max_iter):
                epoch_log_likelihood = 0.0
                epoch_cost = 0.0
                learning_rate = self.learning_rate_
                if isinstance(_optimizer, RiemannianConjugateGradient):
                    # One line-search step on the whole matrix. The bar shows the step
                    # it accepted, and the log-likelihood of every cell, not a sample.
                    epoch_cost = float(_optimizer.step(closure))
                    epoch_log_likelihood = float(log_base_measure.sum()) - epoch_cost
                    self.loadings_learning_scores_[-1].append(np.float32(epoch_cost))
                    self.loadings_learning_rates_[-1].append([_optimizer.last_step])
                    epochs.set_postfix(
                        step=f"{_optimizer.last_step:.2e}",
                        log_lik=f"{epoch_log_likelihood:.4E}",
                        refresh=False,
                    )
                    # One failed search only restarts the direction; two in a row
                    # mean that steepest descent cannot improve the cost either.
                    failed_searches = (
                        failed_searches + 1 if _optimizer.last_step == 0.0 else 0
                    )
                    if failed_searches > 1:
                        epochs.update(1)
                        msg = (
                            "The line search found no step that lowers the cost, from "
                            "the conjugate direction or from the gradient, so the fit "
                            "stops here. This is how optimizer='cg' converges."
                        )
                        warnings.warn(msg, UserWarning, stacklevel=2)
                        break
                else:
                    step_costs = []
                    order = torch.randperm(n)[: n - n % batch_size]
                    for batch in order.split(batch_size):
                        batch_data, batch_parameters = rows.block(batch, device)
                        cost_step = self._optim_cost(
                            loadings=_loadings,
                            intercept=_intercept,
                            batch_data=batch_data,
                            batch_parameters=batch_parameters,
                            batch_depth=(
                                None if _depth is None else _depth[batch.to(device)]
                            ),
                        )
                        step_costs.append(cost_step.detach())
                        cost_step.backward()
                        _optimizer.step()
                        _optimizer.zero_grad()
                        self.loadings_learning_rates_[-1].append(
                            _lr_scheduler.get_last_lr()
                        )
                    costs = torch.stack(step_costs).cpu().numpy()
                    self.loadings_learning_scores_[-1].extend(costs)
                    epoch_cost = float(costs.sum(dtype=np.float64))
                    # The cost leaves out log h, which the bar puts back.
                    epoch_log_likelihood = (
                        float(log_base_measure[order].sum()) - epoch_cost
                    )
                    learning_rate = _lr_scheduler.get_last_lr()[0]
                    if cost_anchor is None:
                        cost_anchor = max(abs(epoch_cost), 1e-12)
                    _lr_scheduler.step(epoch_cost / cost_anchor)
                    if _lr_scheduler.get_last_lr()[0] != learning_rate:
                        cost_anchor = max(abs(epoch_cost), 1e-12)

                    # The log-likelihood of the cells seen in this epoch.
                    epochs.set_postfix(
                        lr=f"{learning_rate:.2e}",
                        log_lik=f"{epoch_log_likelihood:.4E}",
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
                        _optimizer,
                        _loadings,
                        _intercept,
                        _depth,
                        _lr_scheduler,
                    )
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                    return self._saturated_loading_iter(
                        rows=rows,
                        batch_size=batch_size,
                        device=device,
                        log_base_measure=log_base_measure,
                    )

                if (
                    not isinstance(_optimizer, RiemannianConjugateGradient)
                    and learning_rate <= LEARNING_RATE_LIMIT
                ):
                    msg = (
                        f"The learning rate reached its floor of "
                        f"{LEARNING_RATE_LIMIT:.0e}. The loadings have stopped moving, "
                        f"so the fit will stop after this epoch. Increase gamma for a "
                        f"longer fit."
                    )
                    warnings.warn(msg, UserWarning, stacklevel=2)
                    break

        return (_loadings, _intercept, _depth)

    def _create_saturated_loading_optim(
        self, rows: _Rows, device: torch.device
    ) -> tuple[
        torch.optim.Optimizer,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.optim.lr_scheduler.ReduceLROnPlateau,
    ]:
        r"""Initializes the optimisation problem.

        Parameters
        ----------
        rows : _Rows
            The dataset, cells in rows and features in columns, with its saturated
            parameters ($g^{-1}\left(X\right)$), read by blocks of rows.
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
        n, p = rows.shape
        random_batch_size = min(n, 2**13)
        random_idx = np.random.choice(
            np.arange(n), replace=False, size=random_batch_size
        )
        _, subset = (
            part.cpu() for part in rows.block(torch.from_numpy(random_idx), device)
        )
        if self.init == "spectral":
            _, _, v = torch.linalg.svd(
                subset - torch.mean(subset, dim=0),
                full_matrices=False,
            )
            loadings = ManifoldParameter(
                v[: self.n_pc, :].T.to(device), manifold=Grassmann()
            )
        elif self.init == "random":
            loadings = ManifoldParameter(
                Grassmann().random(p, self.n_pc, device=device),
                manifold=Grassmann(),
            )

        # Initialize intercept
        if self.exponential_family.family_name in ["poisson"]:
            intercept = ManifoldParameter(torch.median(subset, dim=0).values.to(device))
        else:
            intercept = ManifoldParameter(torch.mean(subset, dim=0).to(device))

        # The offset of every cell, started at the mean residual of that cell once the
        # offset of the feature is out, which is where least squares would put it.
        depth = None
        groups = [
            {"params": loadings, "lr": self.learning_rate_},
            {"params": intercept, "lr": self.learning_rate_ * INTERCEPT_RATE_SCALE},
        ]
        floors = [LEARNING_RATE_LIMIT, LEARNING_RATE_LIMIT * INTERCEPT_RATE_SCALE]
        if self.depth_factor:
            start = torch.empty(n)
            for chunk in rows.slices(self.chunk_size):
                _, theta = rows.block(chunk, device)
                start[chunk] = (theta - intercept.detach()).mean(dim=1).cpu()
            depth = ManifoldParameter(start.to(device))
            groups.append({
                "params": depth,
                "lr": self.learning_rate_ * DEPTH_RATE_SCALE,
            })
            floors.append(LEARNING_RATE_LIMIT * DEPTH_RATE_SCALE)

        tqdm.write(f"GLMPCA FAMILY: {self.family}")
        tqdm.write(f"INITIAL LEARNING RATE: {self.learning_rate_}")
        algorithm = _OPTIMIZERS[self.optimizer]
        optimizer = algorithm(params=groups)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=self.gamma,
            patience=PLATEAU_PATIENCE,
            threshold=PLATEAU_THRESHOLD,
            threshold_mode="abs",
            min_lr=floors,
            eps=0.0,
        )

        return optimizer, loadings, intercept, depth, lr_scheduler

    def _full_cost(
        self,
        loadings: torch.Tensor,
        intercept: torch.Tensor,
        rows: _Rows,
        device: torch.device,
        depth: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""Sums the cost over chunks of chunk_size rows, to never expand X."""
        total = torch.zeros((), device=device)
        for chunk in rows.slices(self.chunk_size):
            data, parameters = rows.block(chunk, device)
            total = total + self._optim_cost(
                loadings,
                intercept,
                data,
                parameters,
                None if depth is None else depth[chunk],
            )
        return total

    def _optim_cost(
        self,
        loadings: torch.Tensor,
        intercept: torch.Tensor,
        batch_data: torch.Tensor,
        batch_parameters: torch.Tensor,
        batch_depth: torch.Tensor | None = None,
    ) -> torch.Tensor:
        intercept_term = intercept.unsqueeze(0)
        if batch_depth is not None:
            # The offset of a cell joins the offset of a feature, so the subspace
            # never has to carry either of them.
            intercept_term = intercept_term + batch_depth.unsqueeze(1)

        projected_parameters = batch_parameters - intercept_term
        projected_parameters = projected_parameters.matmul(loadings).matmul(loadings.T)
        projected_parameters = projected_parameters + intercept_term

        return self.exponential_family.neg_log_likelihood(
            batch_data, projected_parameters
        )
