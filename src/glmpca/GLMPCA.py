from __future__ import annotations

import copy
import warnings
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import anndata as ad
import numpy as np
import torch
import torch.optim
from anndata.abc import CSCDataset, CSRDataset
from scipy.sparse import csc_array, csc_matrix, csr_array, csr_matrix, issparse
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
from .sparse import BackedRows, SparseRows, binary
from .utils import cell_depth, components_to_keep

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

_OPTIMIZERS = {
    "adagrad": RiemannianAdagrad,
    "adam": RiemannianAdam,
    "cg": RiemannianConjugateGradient,
}

_Result = TypeVar("_Result")

DEFAULT_LEARNING_RATE = 0.2
SPARSE_FAMILIES = (
    "gaussian",
    "poisson",
    "negative_binomial",
    "bernoulli",
    "binomial",
)
PRECOMPUTED_FAMILIES = ("beta", "sigmoid_beta", "gamma")
"""The families whose `invert_g` is a bisection, too slow to repeat at every step, so
the fit computes their saturated parameters once and holds them."""
INTERCEPT_RATE_SCALE = 0.01
LEARNING_RATE_LIMIT = 1e-8
PLATEAU_PATIENCE = 10
PLATEAU_THRESHOLD = 1e-4
DEFAULT_BATCH_SIZE = 4096
DEFAULT_RANDOM_STATE = 42
SPECTRAL_ROWS = 4096
SPECTRAL_ITERATIONS = 16
SPECTRAL_BYTES = 2**30
"""Bytes of the dense sample of the spectral start, which bound its number of cells."""
DEVICE_MEMORY_SHARE = 0.8
WORKING_COPIES = 8
BLOCK_BYTES = 256 * 2**20
"""Bytes of one dense float32 copy of a block of rows on the CPU."""


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


def _free_device_memory(device: torch.device) -> int:
    """The free memory of a CUDA device, with what this process holds but does not use.

    `torch.cuda.mem_get_info` counts the blocks that the caching allocator of this
    process keeps after the passes that freed them as used, although this process can
    use them again.
    """
    free = torch.cuda.mem_get_info(device)[0]
    if torch.cuda.is_initialized():
        free += torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    return free


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
    if issparse(X):
        return torch.from_numpy(np.ascontiguousarray(X.toarray(), dtype=np.float32))
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
    elif isinstance(X, np.ndarray) or issparse(X):
        counts = X
    else:
        msg = (
            f"X format unrecognised: {type(X)} != torch.Tensor, np.ndarray or "
            f"anndata.AnnData"
        )
        raise ValueError(msg)
    return SparseRows(csr_matrix(counts, dtype=np.float32))


def _is_sparse(X: torch.Tensor | np.ndarray | ad.AnnData) -> bool:
    """Whether `X` holds its counts as a sparse matrix."""
    counts = X.X if isinstance(X, ad.AnnData) else X
    return issparse(counts) or isinstance(counts, CSRDataset | CSCDataset)


class _Rows:
    """The matrix of a fit and its saturated parameters, read by blocks of rows.

    `data` is a tensor, or a `SparseRows` when sparse. `theta` holds the saturated
    parameters of PRECOMPUTED_FAMILIES, computed once, or is None, and then every block
    computes its own from its data, so the fit holds one matrix, not two.
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

    def read(self, rows: slice | torch.Tensor, device: torch.device) -> torch.Tensor:
        """The data of `rows`, dense on `device`."""
        if isinstance(self.data, SparseRows):
            return self.data.dense(rows, device)
        return self.data[rows].to(device, non_blocking=True)

    def block(
        self, rows: slice | torch.Tensor, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The data and the saturated parameters of `rows`, on `device`."""
        data = self.read(rows, device)
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
        fewer rows, the number of rows is used instead. It is a setting of the
        optimisation only: a batch larger than a block of rows (chunk_size) is read
        block by block, and the gradients of its blocks add up before the one step of
        the batch, which is then the step of the whole batch. Defaults to 4096.

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
        Method to initialize loadings and intercepts. "spectral" performs SVD on
        the saturated parameters of a batch (max of batch size and 4096) of
        random cells and "random" performs a random initialization on the
        Grassmann manifold. Defaults to "spectral".

    drop_depth: bool
        Whether to drop, after the fit, every component whose scores follow the depth
        of the cells, the total of their counts as given, |Spearman rho| above
        DEPTH_CORRELATION_CUTOFF (`components_to_keep`), so fewer than n_pc components
        can stay. log_likelihood_ is that of the fit before the drop. Defaults to False.

    n_jobs: int or None
        Number of jobs for the per-feature fits of the family parameters. If given,
        it is written to family_params["n_jobs"] and replaces any value there. Only
        "beta", "sigmoid_beta" and "gamma" use it; the other families ignore it.
        Defaults to None, which keeps the family setting.

    device: str, torch.device or None
        Device used to compute the saturated parameters and to train. None selects
        "cuda" if it is available, else "cpu". The "mps" device is not supported.
        Fitted attributes are always stored on the CPU. Defaults to None.

    chunk_size: int or None
        Number of rows handled at a time in every pass over the whole matrix: the
        saturated parameters, the cost that "cg" and n_init use, the offsets and the
        transform. It bounds the memory of those passes and does not change the result.
        It is also the largest block of an SGD batch. None chooses it from the device
        (`_chunk_rows`): on the CPU the rows of BLOCK_BYTES, and on CUDA the rows whose
        WORKING_COPIES fit in a share of the free memory, halved and run again if
        the GPU still runs out of memory. Defaults to None.

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

    keep_sparse: bool or None
        Whether to hold the input as a CSR matrix for the whole fit (True) or dense
        (False). None, the default, keeps the storage of the input: sparse when its
        counts are a sparse matrix, else dense. A block of rows is densified when it
        is read, and its saturated parameters are computed then, except for
        PRECOMPUTED_FAMILIES, whose dense data holds them once beside it.

    compile: bool
        Whether to compile the cost of the training loop with `torch.compile`, which
        joins its element-wise operations. This speeds up the training iterations in
        for a fixed compilation time. Recommended for long fits (>100 iters).
        It needs a C++ compiler on the CPU, and Triton on CUDA. Defaults to False.

    random_state: int or None
        Seed of the fit: the cells of the spectral start, the randomized SVD of that
        start and the order of the minibatches. fit seeds a numpy generator of its
        own, and torch inside `torch.random.fork_rng`, so the global random state of
        the caller is left as it was. None does not seed, and two fits then differ.
        Defaults to DEFAULT_RANDOM_STATE (42).

    binarize: bool
        Whether fit and transform read every value that is not zero as 1. A backed
        AnnData is binarized block by block as it is read, any other input once. The
        input of the caller is left as it was. Defaults to False.

    """

    def __init__(
        self,
        n_pc: int,
        family: str | ExponentialFamily = "gaussian",
        family_params: dict[str, Any] | None = None,
        max_iter: int = 100,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        batch_size: int = DEFAULT_BATCH_SIZE,
        gamma: float = 0.5,
        n_init: int = 1,
        init: Literal["spectral", "random"] = "spectral",
        drop_depth: bool = False,
        n_jobs: int | None = None,
        device: str | torch.device | None = None,
        chunk_size: int | None = None,
        optimizer: Literal["adagrad", "adam", "cg"] = "adagrad",
        keep_sparse: bool | None = None,
        compile: bool = False,
        random_state: int | None = DEFAULT_RANDOM_STATE,
        binarize: bool = False,
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
        self.batch_size_: int | None = None
        self.n_init = n_init
        self.gamma = gamma
        self.init = init
        self.drop_depth = drop_depth
        self.chunk_size = chunk_size
        self._chunk = chunk_size or batch_size
        self.optimizer = optimizer
        self.keep_sparse = keep_sparse
        self.keep_sparse_: bool | None = None
        self.compile = compile
        self.random_state = random_state
        self.binarize = binarize
        self._rng = np.random.default_rng(random_state)

        self.saturated_loadings_: torch.Tensor | None = None
        # Log-likelihood of the fit, the real one, with the base measure
        self.log_likelihood_: float | None = None
        # saturated_intercept_: before projecting
        self.saturated_intercept_: torch.Tensor | None = None
        # drop_depth: the Spearman rho of every component with the depth, and how many
        # components went.
        self.depth_correlations_: np.ndarray | None = None
        self.n_dropped_components_: int | None = None
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
        self.exponential_family.family_params["chunk_size"] = self._chunk

    def fit(
        self,
        X: torch.Tensor | np.ndarray | ad.AnnData,
        depth: np.ndarray | None = None,
    ) -> bool:
        r"""Fits a GLM-PCA to a specific dataset.

        Parameters
        ----------
        X : torch.Tensor, np.ndarray or AnnData
            Dataset with cells in rows and features in columns.
        depth : np.ndarray or None
            The depth of every cell for drop_depth, for an `X` that is no longer the
            counts, as after a normalization. None takes the row sums of `X`.

        Returns
        -------
        bool
            Returns True if the fitting procedure was successful.

        """
        device = _resolve_device(self.device)
        seeded = self.random_state is not None
        with torch.random.fork_rng(
            devices=[device] if device.type == "cuda" else [], enabled=seeded
        ):
            if seeded:
                torch.manual_seed(self.random_state)
            self._rng = np.random.default_rng(self.random_state)
            return self._fit(X, depth)

    def _fit(
        self, X: torch.Tensor | np.ndarray | ad.AnnData, depth: np.ndarray | None
    ) -> bool:
        """The fit itself, which fit runs with the random state of the model."""
        device = _resolve_device(self.device)
        if self.init not in ("spectral", "random"):
            msg = f"init={self.init!r} is not valid. Use 'spectral' or 'random'."
            raise ValueError(msg)
        if self.optimizer not in _OPTIMIZERS:
            choices = ", ".join(repr(name) for name in _OPTIMIZERS)
            msg = f"optimizer={self.optimizer!r} is not valid. Use one of {choices}."
            raise ValueError(msg)
        if self.chunk_size is not None and self.chunk_size < 1:
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

        X_fit = self._rows_source(X, device)
        self.keep_sparse_ = isinstance(X_fit, SparseRows)
        if X_fit.shape[0] < 2:
            msg = (
                f"A fit needs at least 2 rows (cells), but the input has "
                f"{X_fit.shape[0]}. Check that the data has cells in rows and "
                f"features in columns."
            )
            raise ValueError(msg)
        if X_fit.shape[0] < self.n_pc:
            msg = (
                f"n_pc={self.n_pc} components need at least {self.n_pc} rows (cells), "
                f"but the input has {X_fit.shape[0]}. Use n_pc={X_fit.shape[0]} or "
                f"fewer."
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
        self.batch_size_ = batch_size

        _announce_device(device)
        n_rows, row_bytes = X_fit.shape[0], X_fit.shape[1] * X_fit.dtype.itemsize
        self._chunk = self._chunk_rows(device, n_rows, row_bytes)
        family.family_params["chunk_size"] = self._chunk_rows(
            torch.device("cpu"), n_rows, row_bytes
        )

        # Fit exponential family params (e.g., dispersion for negative binomial)
        family.initialize_family_parameters(X_fit)
        family.load_family_params_to_gpu(device)

        saturated_parameters = None
        if (
            isinstance(X_fit, torch.Tensor)
            and family.family_name in PRECOMPUTED_FAMILIES
        ):
            saturated_parameters = torch.empty_like(X_fit)
            for start in range(0, X_fit.shape[0], self._chunk):
                stop = start + self._chunk
                saturated_parameters[start:stop] = family.invert_g(
                    X_fit[start:stop].to(device)
                ).cpu()
        rows = _Rows(X_fit, family, saturated_parameters)

        # log h(x) of every cell, summed over its features. It turns the cost into
        # the real log-likelihood.
        log_base_measure = torch.empty(X_fit.shape[0], dtype=torch.float64)

        def base_measure_pass() -> None:
            for chunk in rows.slices(self._chunk):
                data = rows.read(chunk, device)
                log_base_measure[chunk] = (
                    family.log_base_measure(data).sum(dim=1, dtype=torch.float64).cpu()
                )

        self._with_smaller_blocks(device, base_measure_pass)

        resident = False
        if isinstance(X_fit, torch.Tensor):
            held = (
                (X_fit,)
                if saturated_parameters is None
                else (X_fit, saturated_parameters)
            )
            resident = device.type != "cuda" or _fits_in(
                _free_device_memory(device),
                held,
                WORKING_COPIES * min(self._chunk, n_rows) * row_bytes,
            )
            if resident and device.type == "cuda":
                torch.cuda.empty_cache()
                rows = _Rows(
                    X_fit.to(device),
                    family,
                    None
                    if saturated_parameters is None
                    else saturated_parameters.to(device),
                )
                self._chunk = self._chunk_rows(device, n_rows, row_bytes)
        if device.type == "cuda":
            tqdm.write(
                f"DATA: {'held on the device' if resident else 'copied by chunks'}"
            )
        after = device if resident else torch.device("cpu")

        # Initialize the learning procedure
        self.loadings_learning_scores_ = []
        self.loadings_learning_rates_ = []

        cost = (
            torch.compile(self._optim_cost, dynamic=False)
            if self.compile
            else self._optim_cost
        )

        # Use saturated parameters to find loadings by projected gradient descent
        runs = []
        for _ in range(self.n_init):
            self.learning_rate_ = self.initial_learning_rate_
            runs.append(
                self._saturated_loading_iter(
                    rows, batch_size, device, log_base_measure, cost
                )
            )

        runs = [
            (loadings.to(after), intercept.to(after)) for loadings, intercept in runs
        ]
        family.load_family_params_to_gpu(after)
        self._chunk = self._chunk_rows(after, n_rows, row_bytes)

        # Select best model
        best_model_idx = 0
        if len(runs) > 1:
            with torch.no_grad():
                training_cost = torch.stack([
                    self._with_smaller_blocks(
                        after,
                        lambda run=run: self._full_cost(run[0], run[1], rows, after),
                    )
                    for run in runs
                ])
            best_model_idx = int(torch.argmin(training_cost))
        best_loadings, best_intercept = runs[best_model_idx]
        best_loadings = best_loadings.detach()
        self.saturated_intercept_ = best_intercept.detach()
        with torch.no_grad():
            training_cost = self._with_smaller_blocks(
                after,
                lambda: self._full_cost(
                    best_loadings, self.saturated_intercept_, rows, after
                ),
            )
        self.log_likelihood_ = float(training_cost.neg().cpu()) + float(
            log_base_measure.sum()
        )
        self.saturated_loadings_ = self._with_smaller_blocks(
            after,
            lambda: canonical_basis(best_loadings, self._centered_chunks(rows, after)),
        ).cpu()
        self.depth_correlations_ = self.n_dropped_components_ = None
        if self.drop_depth:
            cells = cell_depth(X) if depth is None else np.asarray(depth, np.float64)
            basis = self.saturated_loadings_.to(after)
            scores = self._with_smaller_blocks(
                after,
                lambda: torch.cat([
                    (chunk @ basis).cpu()
                    for chunk in self._centered_chunks(rows, after)
                ]),
            )
            kept, self.depth_correlations_ = components_to_keep(scores, cells)
            self.n_dropped_components_ = basis.shape[1] - len(kept)
            self.saturated_loadings_ = self.saturated_loadings_[
                :, torch.from_numpy(kept)
            ]
        self.saturated_intercept_ = self.saturated_intercept_.cpu()
        family.load_family_params_to_gpu(torch.device("cpu"))

        return True

    def _rows_source(
        self,
        X: torch.Tensor | np.ndarray | ad.AnnData,
        device: torch.device,
        *,
        announce: bool = True,
    ) -> torch.Tensor | SparseRows:
        """How fit and transform hold `X`: in the file, sparse, or dense.

        An AnnData opened with `backed="r"` stays in its file and is read by blocks
        (`BackedRows`), unless keep_sparse is False, which loads it dense. Any other
        input keeps its storage, sparse or dense, unless keep_sparse says otherwise; a
        family outside SPARSE_FAMILIES is held dense. With binarize, the values that
        are not zero are read as 1. With announce, it says what it chose.
        """
        if isinstance(X, ad.AnnData) and X.isbacked and self.keep_sparse is not False:
            family = self.exponential_family.family_name
            if family not in SPARSE_FAMILIES:
                msg = (
                    f"A backed AnnData is read by blocks of rows, which family="
                    f"{family!r} cannot take. Load it into memory, or use one of "
                    f"{', '.join(repr(name) for name in SPARSE_FAMILIES)}."
                )
                raise ValueError(msg)
            if announce:
                tqdm.write("STORAGE: backed (read by blocks from the file)")
            return BackedRows(X.X, binarize=self.binarize)
        sparse = (
            _is_sparse(X) and self.exponential_family.family_name in SPARSE_FAMILIES
            if self.keep_sparse is None
            else self.keep_sparse
        )
        if announce:
            reason = "as the input" if self.keep_sparse is None else "keep_sparse"
            tqdm.write(f"STORAGE: {'sparse' if sparse else 'dense'} ({reason})")
        rows = _to_sparse(X) if sparse else _to_tensor(X)
        if not self.binarize:
            return rows
        if isinstance(rows, SparseRows):
            return SparseRows(binary(rows.matrix))
        return (rows != 0).to(rows.dtype)

    def _chunk_rows(self, device: torch.device, n_rows: int, row_bytes: int) -> int:
        """The rows of a block: of one pass over the matrix, and of one SGD block.

        chunk_size when it is given. Otherwise, on the CPU, the rows of one dense
        float32 copy of BLOCK_BYTES, and on CUDA the rows whose WORKING_COPIES fit in
        DEVICE_MEMORY_SHARE of the free memory, so a pass takes few, large blocks.
        Never more rows than the matrix has.
        """
        if self.chunk_size is not None:
            return self.chunk_size
        if device.type != "cuda":
            return max(1, min(BLOCK_BYTES // max(row_bytes, 1), n_rows))
        free = _free_device_memory(device)
        rows = int(DEVICE_MEMORY_SHARE * free) // (WORKING_COPIES * max(row_bytes, 1))
        return max(1, min(rows, n_rows))

    def _with_smaller_blocks(
        self, device: torch.device, run: Callable[[], _Result]
    ) -> _Result:
        """`run()`, run again with blocks of half the rows when CUDA runs out of memory.

        The free memory that `_chunk_rows` reads can shrink when another process takes
        some, and `run` must then be a pass that it is safe to start over.
        """
        try:
            return run()
        except torch.OutOfMemoryError:
            if device.type != "cuda" or self._chunk <= 1:
                raise
            torch.cuda.empty_cache()
            self._chunk = max(1, self._chunk // 2)
            tqdm.write(f"GPU MEMORY: out of memory, blocks of {self._chunk} rows")
        return self._with_smaller_blocks(device, run)

    def _centered_chunks(
        self, rows: _Rows, device: torch.device
    ) -> Iterator[torch.Tensor]:
        """The rows that transform projects, in blocks of chunk_size.

        They are centered with the intercept, as transform centers the same cells, so
        the scores canonical_basis orders are the scores that transform reports.
        """
        intercept = self.saturated_intercept_
        assert intercept is not None
        for chunk in rows.slices(self._chunk):
            _, theta = rows.block(chunk, device)
            yield theta - intercept.unsqueeze(0)

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

        X_transform = self._rows_source(X, torch.device("cpu"), announce=False)
        device = X_transform.device
        loadings, intercept = loadings.to(device), intercept.to(device)
        self.exponential_family.load_family_params_to_gpu(device)

        rows = _Rows(X_transform, self.exponential_family, None)
        scores = [torch.empty(0, loadings.shape[1], device=device)]
        chunk_rows = self._chunk_rows(
            device,
            X_transform.shape[0],
            X_transform.shape[1] * X_transform.dtype.itemsize,
        )
        for chunk in rows.slices(chunk_rows):
            _, theta = rows.block(chunk, device)
            scores.append((theta - intercept.unsqueeze(0)).matmul(loadings))

        return torch.cat(scores)

    def _saturated_loading_iter(
        self,
        rows: _Rows,
        batch_size: int,
        device: torch.device,
        log_base_measure: torch.Tensor,
        cost: Callable[..., torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        cost : Callable
            `_optim_cost`, or its compiled form when compile is on.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The loadings and the intercept of every feature.

        """
        if self.learning_rate_ < LEARNING_RATE_LIMIT:
            msg = "LEARNING RATE IS TOO SMALL : DID NOT CONVERGE"
            raise ValueError(msg)

        # Set up list of saving scores
        self.loadings_learning_scores_.append([])
        self.loadings_learning_rates_.append([])

        _optimizer, _loadings, _intercept, _lr_scheduler = (
            self._init_saturated_loading_optim(
                rows=rows, device=device, batch_size=batch_size
            )
        )
        n = rows.shape[0]

        def backward(parts: Callable[[], Iterable[slice | torch.Tensor]]) -> float:
            """The cost of the rows of `parts`, with its gradient summed over them.

            It starts from a zero gradient, so a run that CUDA stops for memory can
            start over with smaller blocks.
            """

            def run() -> float:
                _optimizer.zero_grad()
                total = torch.zeros((), device=device, dtype=torch.float64)
                for part in parts():
                    data, parameters = rows.block(part, device)
                    part_cost = cost(
                        loadings=_loadings,
                        intercept=_intercept,
                        batch_data=data,
                        batch_parameters=parameters,
                    )
                    part_cost.backward()
                    total += part_cost.detach()
                return float(total)

            return self._with_smaller_blocks(device, run)

        def closure() -> float:
            """The cost of the whole matrix, with the gradient summed over blocks."""
            return backward(lambda: rows.slices(self._chunk))

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
                        step_costs.append(
                            backward(lambda batch=batch: batch.split(self._chunk))
                        )
                        _optimizer.step()
                        _optimizer.zero_grad()
                        self.loadings_learning_rates_[-1].append(
                            _lr_scheduler.get_last_lr()
                        )
                    costs = np.asarray(step_costs, dtype=np.float32)
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
                        _lr_scheduler,
                    )
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                    return self._saturated_loading_iter(
                        rows=rows,
                        batch_size=batch_size,
                        device=device,
                        log_base_measure=log_base_measure,
                        cost=cost,
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

        return (_loadings, _intercept)

    def _init_saturated_loading_optim(
        self,
        rows: _Rows,
        device: torch.device,
        batch_size: int,
    ) -> tuple[
        torch.optim.Optimizer,
        torch.Tensor,
        torch.Tensor,
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
        batch_size : int
            The spectral start and the intercept take
            min(max(SPECTRAL_ROWS, batch_size), n) random cells, no more than
            SPECTRAL_BYTES hold densely, but at least n_pc.

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
        n, p = rows.shape
        row_bytes = p * 4
        random_batch_size = max(
            min(max(SPECTRAL_ROWS, batch_size), n, SPECTRAL_BYTES // row_bytes),
            self.n_pc,
        )
        random_idx = self._rng.choice(n, replace=False, size=random_batch_size)
        _, subset = (
            part.cpu() for part in rows.block(torch.from_numpy(random_idx), device)
        )
        components = self.n_pc
        if self.exponential_family.family_name in ["poisson"]:
            center = torch.median(subset, dim=0).values
        else:
            center = torch.mean(subset, dim=0)
        if self.init == "spectral":
            centered = subset - subset.mean(dim=0)
            _, _, v = torch.svd_lowrank(
                centered,
                q=min(2 * components, *centered.shape),
                niter=SPECTRAL_ITERATIONS,
            )
            loadings = ManifoldParameter(
                v[:, :components].to(device), manifold=Grassmann()
            )
        elif self.init == "random":
            loadings = ManifoldParameter(
                Grassmann().random(p, components, device=device),
                manifold=Grassmann(),
            )

        intercept = ManifoldParameter(center.to(device))
        groups = [
            {"params": loadings, "lr": self.learning_rate_},
            {"params": intercept, "lr": self.learning_rate_ * INTERCEPT_RATE_SCALE},
        ]
        floors = [LEARNING_RATE_LIMIT, LEARNING_RATE_LIMIT * INTERCEPT_RATE_SCALE]

        tqdm.write(f"GLMPCA FAMILY: {self.family}")
        tqdm.write(f"INITIAL LEARNING RATE: {self.learning_rate_}")
        blocks = f" (blocks of {self._chunk} rows)" if self._chunk < batch_size else ""
        tqdm.write(f"BATCH SIZE: {batch_size}{blocks}")
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

        return optimizer, loadings, intercept, lr_scheduler

    def _full_cost(
        self,
        loadings: torch.Tensor,
        intercept: torch.Tensor,
        rows: _Rows,
        device: torch.device,
    ) -> torch.Tensor:
        r"""Sums the cost over chunks of chunk_size rows, to never expand X.

        The total runs in float64. It is what `n_init` compares between runs and what
        `log_likelihood_` reports, and it costs one scalar addition per chunk, unlike
        the cost of a chunk itself, whose float32 sum the gradients never see.
        """
        total = torch.zeros((), device=device, dtype=torch.float64)
        for chunk in rows.slices(self._chunk):
            data, parameters = rows.block(chunk, device)
            total = total + self._optim_cost(loadings, intercept, data, parameters)
        return total

    def _optim_cost(
        self,
        loadings: torch.Tensor,
        intercept: torch.Tensor,
        batch_data: torch.Tensor,
        batch_parameters: torch.Tensor,
    ) -> torch.Tensor:
        intercept_term = intercept.unsqueeze(0)

        projected_parameters = batch_parameters - intercept_term
        projected_parameters = projected_parameters.matmul(loadings).matmul(loadings.T)
        projected_parameters = projected_parameters + intercept_term

        return self.exponential_family.neg_log_likelihood(
            batch_data, projected_parameters
        )
