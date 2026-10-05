"""Tests for ``GLMPCA``: family selection, the fit/transform contract, AnnData input."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal, cast

import anndata as ad
import glmpca.GLMPCA as glmpca_module
import numpy as np
import pytest
import scipy.stats
import torch
from glmpca.ExponentialFamily import (
    Beta,
    ExponentialFamily,
    GLMFamily,
    NegativeBinomial,
    _n_workers,
)
from glmpca.GLMPCA import (
    DEFAULT_BATCH_SIZE,
    DEPTH_RATE_SCALE,
    DEPTH_WORKING_COPIES,
    DEVICE_MEMORY_SHARE,
    GLMPCA,
    HOST_MEMORY_SHARE,
    HOST_STAGING_COPIES,
    INTERCEPT_RATE_SCALE,
    LEARNING_RATE_LIMIT,
    PLATEAU_PATIENCE,
    WORKING_COPIES,
    _fits_in,
    _free_device_memory,
    _host_memory,
    _Rows,
    _to_tensor,
)
from glmpca.manifolds import ManifoldParameter, RiemannianAdagrad
from glmpca.sparse import BackedRows, SparseRows, binary, densified
from scipy import sparse

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from glmpca.LSIfamilies import WeightedGaussian

N_CELLS = 40
N_FEATURES = 12
N_PC = 2
N_LSI_FEATURES = 40


@pytest.fixture(autouse=True)
def seed() -> None:
    torch.manual_seed(0)
    np.random.seed(0)


def sample(family: GLMFamily) -> torch.Tensor:
    """Data inside the support of ``family``."""
    shape = (N_CELLS, N_FEATURES)
    if family is GLMFamily.gaussian:
        return torch.randn(shape)
    if family in {
        GLMFamily.poisson,
        GLMFamily.signac_lsi,
        GLMFamily.gensim_lsi,
    }:
        return torch.poisson(torch.full(shape, 3.0))
    if family is GLMFamily.negative_binomial:
        return torch.distributions.NegativeBinomial(
            5.0, probs=torch.tensor(0.4)
        ).sample(shape)
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
def test_a_single_cell_transforms_as_it_does_among_others(family: GLMFamily) -> None:
    X = sample(family)
    model = GLMPCA(
        N_PC, family=family, max_iter=2, batch_size=16, chunk_size=N_CELLS - 1
    )

    model.fit(X)

    torch.testing.assert_close(
        model.transform(X[:1]), model.transform(X)[:1], rtol=1e-4, atol=1e-4
    )


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


def test_an_exponential_family_instance_is_copied_with_its_parameters() -> None:
    family = NegativeBinomial({"max_val": 50.0})

    used = GLMPCA(N_PC, family=family, chunk_size=64).exponential_family

    assert used is not family
    assert type(used) is NegativeBinomial
    assert used.family_params["max_val"] == 50.0
    # GLMPCA gives its own chunk_size to the copy, and leaves the caller's instance.
    assert used.family_params["chunk_size"] == 64
    assert family.family_params["chunk_size"] == 8192


@pytest.mark.parametrize("family", list(GLMFamily))
def test_n_jobs_is_propagated_to_every_family(family: GLMFamily) -> None:
    model = GLMPCA(N_PC, family=family, n_jobs=3)
    assert model.exponential_family.family_params["n_jobs"] == 3


def test_n_jobs_replaces_the_n_jobs_in_family_params() -> None:
    model = GLMPCA(N_PC, family="beta", family_params={"n_jobs": 2}, n_jobs=3)
    assert model.exponential_family.family_params["n_jobs"] == 3


def test_n_jobs_reaches_the_copy_of_a_family_instance_only() -> None:
    family = Beta({"n_jobs": 2})

    model = GLMPCA(N_PC, family=family, n_jobs=3)

    assert model.exponential_family.family_params["n_jobs"] == 3
    assert family.family_params["n_jobs"] == 2


@pytest.mark.parametrize("family", list(GLMFamily))
def test_every_family_fits_with_n_jobs(family: GLMFamily) -> None:
    model = GLMPCA(N_PC, family=family, n_jobs=2, max_iter=1, batch_size=16)
    assert model.fit(sample(family))


@pytest.mark.parametrize("family", list(GLMFamily))
def test_every_family_fits_with_partial_family_params(family: GLMFamily) -> None:
    model = GLMPCA(
        N_PC, family=family, family_params={"n_jobs": 2}, max_iter=1, batch_size=16
    )
    assert model.fit(sample(family))


def test_without_n_jobs_the_family_setting_is_kept() -> None:
    model = GLMPCA(N_PC, family="beta", family_params={"n_jobs": 2})
    assert model.exponential_family.family_params["n_jobs"] == 2


def test_the_family_fit_receives_the_n_jobs_given_to_glmpca(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[int] = []

    def recording_n_workers(n_jobs: int) -> int:
        received.append(n_jobs)
        return _n_workers(n_jobs)

    monkeypatch.setattr("glmpca.ExponentialFamily._n_workers", recording_n_workers)
    model = GLMPCA(N_PC, family="beta", n_jobs=2, max_iter=1, batch_size=16)
    model.fit(sample(GLMFamily.beta))

    assert received == [2]


def test_an_unknown_family_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="nope"):
        GLMPCA(N_PC, family="nope")


@pytest.mark.parametrize("init", ["spectrl", ""])
def test_an_unknown_init_is_rejected_before_the_family_is_fitted(init: str) -> None:
    family = Beta()
    model = GLMPCA(N_PC, family=family, init=cast("Any", init))

    with pytest.raises(ValueError, match="init="):
        model.fit(sample(GLMFamily.beta))

    assert "nu" not in family.family_params


def test_transform_before_fit_fails_with_advice() -> None:
    with pytest.raises(RuntimeError, match=r"Call fit\(\)"):
        GLMPCA(N_PC, family="poisson").transform(sample(GLMFamily.poisson))


def test_each_init_run_starts_from_the_initial_learning_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = GLMPCA(N_PC, family="poisson", n_init=3, learning_rate=0.2, batch_size=16)
    start_rates: list[float] = []

    def run_that_restarts_once(
        rows: _Rows,
        batch_size: int,
        device: torch.device,
        log_base_measure: torch.Tensor,
        cost: Callable[..., torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        start_rates.append(model.learning_rate_)
        model.learning_rate_ *= model.gamma
        return torch.eye(N_FEATURES, N_PC), torch.zeros(N_FEATURES), None

    monkeypatch.setattr(model, "_saturated_loading_iter", run_that_restarts_once)
    model.fit(sample(GLMFamily.poisson))

    assert start_rates == [0.2, 0.2, 0.2]
    assert model.learning_rate_ == 0.1


@pytest.mark.parametrize(
    ("free", "fits"), [(20_000, True), (10_000, False), (0, False)]
)
def test_the_data_stays_on_the_device_only_when_it_fits(free: int, fits: bool) -> None:
    data = torch.zeros(100, 10)

    assert _fits_in(free, (data, data), working=1_000) is fits


def test_the_fit_says_which_device_it_uses(capsys: pytest.CaptureFixture[str]) -> None:
    model = GLMPCA(N_PC, family="poisson", max_iter=2, batch_size=16)

    model.fit(sample(GLMFamily.poisson))

    assert "DEVICE: cpu" in capsys.readouterr().out


def test_a_scheduled_learning_rate_under_the_limit_stops_the_fit() -> None:
    X = sample(GLMFamily.poisson)[:16]
    model = GLMPCA(
        N_PC,
        family="poisson",
        max_iter=3 * PLATEAU_PATIENCE,
        batch_size=16,
        learning_rate=2 * LEARNING_RATE_LIMIT,
        gamma=1e-3,
    )

    with pytest.warns(UserWarning, match="reached its floor"):
        assert model.fit(X)

    (epochs_run,) = [len(scores) for scores in model.loadings_learning_scores_]
    assert epochs_run < model.max_iter
    rates = [rate[0] for rate in model.loadings_learning_rates_[-1]]
    assert rates[-1] == LEARNING_RATE_LIMIT
    assert min(rates) >= LEARNING_RATE_LIMIT


def test_a_fit_that_keeps_improving_does_not_lower_the_learning_rate() -> None:
    X = sample(GLMFamily.poisson)
    model = GLMPCA(N_PC, family="poisson", max_iter=PLATEAU_PATIENCE, batch_size=16)

    model.fit(X)

    # The plateau scheduler waits for PLATEAU_PATIENCE epochs without progress, so a
    # fit this short always runs at the rate it started from.
    rates = [rate[0] for rate in model.loadings_learning_rates_[-1]]
    assert rates == [model.initial_learning_rate_] * len(rates)


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
def test_anndata_input_is_fitted_with_cells_in_rows(
    matrix_type: Callable[[np.ndarray], Any],
) -> None:
    counts = (
        np.random
        .default_rng(0)
        .poisson(3.0, size=(N_CELLS, N_FEATURES))
        .astype(np.float32)
    )
    adata = ad.AnnData(matrix_type(counts))
    loadings = fitted_loadings(adata)

    assert loadings.shape == (N_FEATURES, N_PC)
    torch.testing.assert_close(loadings, fitted_loadings(torch.Tensor(counts)))


@pytest.mark.parametrize(
    "matrix_type",
    [np.asarray, sparse.csr_matrix, sparse.csc_matrix],
    ids=lambda matrix_type: matrix_type.__name__,
)
def test_backed_anndata_input_is_fitted_like_in_memory_input(
    matrix_type: Callable[[np.ndarray], Any], tmp_path: Path
) -> None:
    counts = (
        np.random
        .default_rng(0)
        .poisson(3.0, size=(N_CELLS, N_FEATURES))
        .astype(np.float32)
    )
    path = tmp_path / "counts.h5ad"
    ad.AnnData(matrix_type(counts)).write_h5ad(path)
    adata = ad.read_h5ad(path, backed="r")
    try:
        torch.testing.assert_close(
            fitted_loadings(adata), fitted_loadings(torch.Tensor(counts))
        )
    finally:
        adata.file.close()


@pytest.mark.parametrize("init", ["spectral", "random"])
@pytest.mark.parametrize("family", list(GLMFamily))
def test_fitted_loadings_are_orthonormal(
    family: GLMFamily, init: Literal["spectral", "random"]
) -> None:
    X = sample(family)
    model = GLMPCA(N_PC, family=family, init=init, max_iter=5, batch_size=16)
    model.fit(X)

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
        ours: RiemannianAdagrad([ours], lr=0.1),
        theirs: torch.optim.Adagrad([theirs], lr=0.1, eps=1e-10),
    }
    for _ in range(50):
        for point, optimizer in optimizers.items():
            optimizer.zero_grad()
            (point - target).square().sum().backward()
            optimizer.step()

    torch.testing.assert_close(ours.detach(), theirs.detach(), rtol=0, atol=1e-6)


def test_the_mps_device_is_rejected() -> None:
    model = GLMPCA(N_PC, family="poisson", device="mps")
    with pytest.raises(ValueError, match="mps"):
        model.fit(sample(GLMFamily.poisson))


@pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA is available")
def test_cuda_is_rejected_when_it_is_not_available() -> None:
    model = GLMPCA(N_PC, family="poisson", device="cuda")
    with pytest.raises(ValueError, match="CUDA is not available"):
        model.fit(sample(GLMFamily.poisson))


@pytest.mark.parametrize("family", list(GLMFamily))
def test_fitted_attributes_are_on_the_cpu(family: GLMFamily) -> None:
    model = GLMPCA(N_PC, family=family, max_iter=2, batch_size=16)
    model.fit(sample(family))

    assert model.saturated_loadings_ is not None
    assert model.saturated_intercept_ is not None
    assert model.saturated_loadings_.device.type == "cpu"
    assert model.saturated_intercept_.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("family", list(GLMFamily))
def test_a_cuda_fit_matches_the_cpu_fit(family: GLMFamily) -> None:
    X = sample(family)
    models = {}
    for device in ("cpu", "cuda"):
        torch.manual_seed(0)
        np.random.seed(0)
        models[device] = GLMPCA(
            N_PC, family=family, max_iter=2, batch_size=16, device=device
        )
        models[device].fit(X)

    cpu_loadings = models["cpu"].saturated_loadings_
    cuda_loadings = models["cuda"].saturated_loadings_
    assert cpu_loadings is not None
    assert cuda_loadings is not None
    cosines = torch.linalg.svdvals(cpu_loadings.detach().T @ cuda_loadings.detach())
    torch.testing.assert_close(cosines, torch.ones(N_PC), rtol=0, atol=1e-4)
    assert models["cuda"].transform(X.cuda()).device.type == "cuda"


@pytest.mark.parametrize(
    "matrix_type",
    [np.asarray, sparse.csr_matrix, sparse.csc_matrix],
    ids=lambda matrix_type: matrix_type.__name__,
)
def test_transform_accepts_anndata_and_ndarray_like_a_tensor(
    matrix_type: Callable[[np.ndarray], Any],
) -> None:
    counts = (
        np.random
        .default_rng(0)
        .poisson(3.0, size=(N_CELLS, N_FEATURES))
        .astype(np.float32)
    )
    model = GLMPCA(N_PC, family="poisson", max_iter=2, batch_size=8)
    model.fit(torch.Tensor(counts))

    expected = model.transform(torch.Tensor(counts))

    torch.testing.assert_close(
        model.transform(ad.AnnData(matrix_type(counts))), expected
    )
    torch.testing.assert_close(model.transform(counts), expected)


def test_transform_rejects_an_unknown_input_type() -> None:
    model = GLMPCA(N_PC, family="poisson", max_iter=2, batch_size=8)
    model.fit(sample(GLMFamily.poisson))

    with pytest.raises(ValueError, match="X format unrecognised"):
        model.transform([[1.0, 2.0]])  # ty: ignore[invalid-argument-type]


def test_the_fitted_attributes_do_not_require_gradients() -> None:
    model = GLMPCA(N_PC, family="poisson", max_iter=2, batch_size=8)
    model.fit(sample(GLMFamily.poisson))

    assert model.saturated_loadings_ is not None
    assert model.saturated_intercept_ is not None
    assert not model.saturated_loadings_.requires_grad
    assert not model.saturated_intercept_.requires_grad
    assert model.saturated_loadings_.numpy().shape == (N_FEATURES, N_PC)
    assert model.saturated_intercept_.numpy().shape == (N_FEATURES,)


def test_the_transform_output_does_not_require_gradients() -> None:
    X = sample(GLMFamily.poisson)
    model = GLMPCA(N_PC, family="poisson", max_iter=2, batch_size=8)
    model.fit(X)

    scores = model.transform(X)

    assert not scores.requires_grad
    assert scores.numpy().shape == (N_CELLS, N_PC)


def test_the_family_params_dictionary_of_the_caller_is_not_changed() -> None:
    params = {"n_jobs": 2}
    model = GLMPCA(N_PC, family="beta", family_params=params, max_iter=1, batch_size=16)

    model.fit(sample(GLMFamily.beta))

    assert params == {"n_jobs": 2}
    assert "nu" in model.exponential_family.family_params


def test_a_family_instance_of_the_caller_is_not_changed_by_a_fit() -> None:
    family = Beta({"n_jobs": 2})
    model = GLMPCA(N_PC, family=family, n_jobs=3, max_iter=1, batch_size=16)

    model.fit(sample(GLMFamily.beta))

    assert family.family_params["n_jobs"] == 2
    assert "nu" not in family.family_params


def test_the_family_keeps_one_parameters_dictionary_over_a_fit() -> None:
    model = GLMPCA(N_PC, family="beta", max_iter=1, batch_size=16)
    params = model.exponential_family.family_params

    model.fit(sample(GLMFamily.beta))

    assert model.exponential_family.family_params is params
    assert "nu" in params


@pytest.mark.parametrize("family", list(GLMFamily))
def test_an_input_with_one_row_is_rejected(family: GLMFamily) -> None:
    model = GLMPCA(N_PC, family=family, max_iter=1, batch_size=8)

    with pytest.raises(ValueError, match="at least 2 rows"):
        model.fit(sample(family)[:1])


def test_an_anndata_with_one_row_is_rejected() -> None:
    counts = (
        np.random.default_rng(0).poisson(3.0, size=(1, N_FEATURES)).astype(np.float32)
    )
    model = GLMPCA(N_PC, family="poisson", max_iter=1, batch_size=8)

    with pytest.raises(ValueError, match="at least 2 rows"):
        model.fit(ad.AnnData(counts))


def test_an_input_with_two_rows_is_fitted() -> None:
    model = GLMPCA(N_PC, family="beta", max_iter=1, batch_size=2)

    assert model.fit(sample(GLMFamily.beta)[:2])


@pytest.mark.parametrize("family", list(GLMFamily))
def test_the_chunked_saturation_equals_one_call(family: GLMFamily) -> None:
    X = sample(family)

    torch.manual_seed(0)
    np.random.seed(0)
    chunked = GLMPCA(N_PC, family=family, max_iter=1, batch_size=8, chunk_size=7)
    chunked.fit(X)

    torch.manual_seed(0)
    np.random.seed(0)
    whole = GLMPCA(N_PC, family=family, max_iter=1, batch_size=8, chunk_size=10**9)
    whole.fit(X)

    assert chunked.saturated_loadings_ is not None
    assert whole.saturated_loadings_ is not None
    chunked_span = chunked.saturated_loadings_ @ chunked.saturated_loadings_.T
    whole_span = whole.saturated_loadings_ @ whole.saturated_loadings_.T
    torch.testing.assert_close(chunked_span, whole_span)


def test_the_chunked_cost_equals_the_cost_of_the_whole_matrix() -> None:
    X = sample(GLMFamily.poisson)
    model = GLMPCA(N_PC, family="poisson", max_iter=1, batch_size=8, chunk_size=7)
    model.fit(X)
    assert model.saturated_loadings_ is not None
    assert model.saturated_intercept_ is not None
    parameters = model.exponential_family.invert_g(X)

    with torch.no_grad():
        whole = model._optim_cost(
            model.saturated_loadings_, model.saturated_intercept_, X, parameters
        )
        chunked = model._full_cost(
            model.saturated_loadings_,
            model.saturated_intercept_,
            _Rows(X, model.exponential_family, parameters),
            torch.device("cpu"),
        )

    # _full_cost totals its chunks in float64, for the value that n_init compares.
    assert chunked.dtype is torch.float64
    torch.testing.assert_close(chunked, whole.double(), rtol=1e-5, atol=1e-4)


def test_a_float32_array_is_not_copied() -> None:
    counts = (
        np.random
        .default_rng(0)
        .poisson(3.0, size=(N_CELLS, N_FEATURES))
        .astype(np.float32)
    )

    X = _to_tensor(counts)

    assert X.data_ptr() == counts.__array_interface__["data"][0]


def test_fit_does_not_change_the_input() -> None:
    X = sample(GLMFamily.poisson)
    before = X.clone()

    GLMPCA(N_PC, family="poisson", max_iter=2, batch_size=8).fit(X)

    torch.testing.assert_close(X, before)


@pytest.mark.parametrize(("batch_size", "rows"), [(4, 8), (16, 16), (4096, N_CELLS)])
def test_the_spectral_start_takes_at_least_spectral_rows_cells(
    monkeypatch: pytest.MonkeyPatch, batch_size: int, rows: int
) -> None:
    monkeypatch.setattr("glmpca.GLMPCA.SPECTRAL_ROWS", 8)
    shapes = []
    svd_lowrank = torch.svd_lowrank

    def recording_svd(
        A: torch.Tensor, q: int, niter: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shapes.append(tuple(A.shape))
        return svd_lowrank(A, q=q, niter=niter)

    monkeypatch.setattr(torch, "svd_lowrank", recording_svd)
    model = GLMPCA(N_PC, family="poisson", max_iter=1, batch_size=batch_size)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(sample(GLMFamily.poisson))

    assert shapes == [(rows, N_FEATURES)]
    assert model.saturated_loadings_ is not None
    assert model.saturated_loadings_.shape == (N_FEATURES, N_PC)


def test_fewer_cells_than_components_are_rejected() -> None:
    model = GLMPCA(N_CELLS + 1, family="poisson", max_iter=1)

    with pytest.raises(ValueError, match=rf"n_pc={N_CELLS + 1} .* has {N_CELLS}\."):
        model.fit(sample(GLMFamily.poisson))


@pytest.mark.parametrize("compile_cost", [False, True])
def test_compile_compiles_the_cost_of_the_training_loop(
    monkeypatch: pytest.MonkeyPatch, compile_cost: bool
) -> None:
    compiled = []

    def recording_compile(
        function: Callable[..., torch.Tensor], *, dynamic: bool
    ) -> Callable[..., torch.Tensor]:
        compiled.append((function, dynamic))
        return function

    monkeypatch.setattr(torch, "compile", recording_compile)
    model = GLMPCA(
        N_PC, family="poisson", max_iter=1, batch_size=16, compile=compile_cost
    )
    model.fit(sample(GLMFamily.poisson))

    assert compiled == ([(model._optim_cost, False)] if compile_cost else [])


def test_a_compiled_fit_equals_an_eager_fit() -> None:
    likelihoods = []
    for compile_cost in (False, True):
        torch.manual_seed(0)
        np.random.seed(0)
        model = GLMPCA(
            N_PC,
            family="poisson",
            optimizer="cg",
            depth_factor=True,
            max_iter=3,
            compile=compile_cost,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(sample(GLMFamily.poisson))
        likelihoods.append(model.log_likelihood_)

    assert likelihoods[1] == pytest.approx(likelihoods[0], rel=1e-6)


def lsi_counts(n_cells: int = 300, n_features: int = 40) -> torch.Tensor:
    """Counts of three groups of cells, with depths that vary between cells."""
    generator = torch.Generator().manual_seed(0)
    depth = torch.exp(5.0 + 0.6 * torch.randn(n_cells, 1, generator=generator))
    profiles = torch.softmax(
        1.5 * torch.randn(3, n_features, generator=generator), dim=1
    )
    groups = torch.randint(0, 3, (n_cells,), generator=generator)
    return torch.poisson(depth * profiles[groups] / 4, generator=generator)


def lsi_fit(family: str, *, depth_factor: bool = False) -> GLMPCA:
    torch.manual_seed(0)
    np.random.seed(0)
    model = GLMPCA(
        3, family=family, optimizer="cg", max_iter=100, depth_factor=depth_factor
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(lsi_counts())
    return model


@pytest.mark.parametrize("family", ["signac_lsi", "gensim_lsi"])
def test_an_lsi_family_finds_the_svd_of_its_weighted_matrix(family: str) -> None:
    model = lsi_fit(family)
    weighted = model.exponential_family.weigh(lsi_counts())
    _, _, vt = torch.linalg.svd(weighted, full_matrices=False)

    assert model.saturated_loadings_ is not None
    cosines = torch.linalg.svdvals(model.saturated_loadings_.T @ vt[:3].T)
    assert float(cosines.min()) > 0.999


@pytest.mark.parametrize("family", ["signac_lsi", "gensim_lsi"])
def test_an_lsi_family_holds_no_intercept_and_no_additive_offset(family: str) -> None:
    model = lsi_fit(family, depth_factor=True)

    assert model.saturated_intercept_ is not None
    assert not model.saturated_intercept_.any()
    assert model.saturated_depth_ is None


@pytest.mark.parametrize(("cutoff", "dropped", "kept"), [(2.0, 0, 3), (-1.0, 3, 0)])
def test_the_depth_factor_of_an_lsi_family_drops_the_components_past_the_cutoff(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    cutoff: float,
    dropped: int,
    kept: int,
) -> None:
    monkeypatch.setattr("glmpca.GLMPCA.DEPTH_CORRELATION_CUTOFF", cutoff)

    model = lsi_fit("signac_lsi", depth_factor=True)

    assert model.depth_correlations_ is not None
    assert model.depth_correlations_.shape == (3,)
    assert model.n_dropped_components_ == dropped
    assert model.saturated_loadings_ is not None
    assert model.saturated_loadings_.shape == (N_LSI_FEATURES, kept)
    assert f"DEPTH: {dropped} of 3 components dropped" in capsys.readouterr().out


def test_the_lsi_depth_factor_drops_the_component_that_follows_the_depth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    whole = lsi_fit("gensim_lsi", depth_factor=True)
    assert whole.depth_correlations_ is not None
    deepest = int(np.argmax(np.abs(whole.depth_correlations_)))
    monkeypatch.setattr(
        "glmpca.GLMPCA.DEPTH_CORRELATION_CUTOFF",
        float(np.abs(whole.depth_correlations_[deepest])) - 1e-6,
    )

    model = lsi_fit("gensim_lsi", depth_factor=True)

    assert model.n_dropped_components_ == 1
    assert model.saturated_loadings_ is not None
    assert model.saturated_loadings_.shape == (N_LSI_FEATURES, 2)
    scores = model.transform(lsi_counts()).numpy()
    depth = lsi_counts().sum(dim=1).numpy()
    for score in scores.T:
        rho = scipy.stats.spearmanr(score, depth).statistic
        assert abs(rho) < abs(whole.depth_correlations_[deepest])


@pytest.mark.parametrize("family", ["signac_lsi", "gensim_lsi"])
def test_an_lsi_fit_weighs_the_counts_once(
    monkeypatch: pytest.MonkeyPatch, family: str
) -> None:
    family_type = GLMFamily(family).distribution()
    calls = []
    weigh = family_type.weigh

    def counting_weigh(self: WeightedGaussian, X: torch.Tensor) -> torch.Tensor:
        calls.append(X.shape[0])
        return weigh(self, X)

    monkeypatch.setattr(family_type, "weigh", counting_weigh)
    torch.manual_seed(0)
    np.random.seed(0)
    model = GLMPCA(3, family=family, optimizer="cg", max_iter=5, batch_size=128)
    model.fit(lsi_counts())

    assert sum(calls) == lsi_counts().shape[0]


@pytest.mark.parametrize("family", ["signac_lsi", "gensim_lsi"])
def test_an_lsi_log_likelihood_is_the_gaussian_density_of_the_weighted_matrix(
    family: str,
) -> None:
    model = lsi_fit(family)
    weighted = model.exponential_family.weigh(lsi_counts()).double()
    assert model.saturated_loadings_ is not None
    loadings = model.saturated_loadings_.double()
    fitted = weighted @ loadings @ loadings.T

    expected = float(
        -0.5 * (weighted - fitted).square().sum()
        - weighted.numel() * 0.5 * np.log(2 * np.pi)
    )
    assert model.log_likelihood_ == pytest.approx(expected, rel=1e-5)


@pytest.mark.parametrize("family", ["signac_lsi", "gensim_lsi"])
def test_an_lsi_fit_is_the_same_with_sparse_storage(family: str) -> None:
    fits = []
    for keep_sparse in (False, True):
        torch.manual_seed(0)
        np.random.seed(0)
        model = GLMPCA(
            3, family=family, optimizer="cg", max_iter=20, keep_sparse=keep_sparse
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(lsi_counts().numpy())
        fits.append(model)

    assert fits[1].log_likelihood_ == pytest.approx(fits[0].log_likelihood_, rel=1e-5)
    projectors = [
        model.saturated_loadings_ @ model.saturated_loadings_.T for model in fits
    ]
    torch.testing.assert_close(projectors[1], projectors[0], atol=1e-4, rtol=0)


def test_signac_lsi_scales_its_scores_and_gensim_lsi_does_not() -> None:
    signac = lsi_fit("signac_lsi")
    gensim = lsi_fit("gensim_lsi")

    scores = signac.transform(lsi_counts())
    torch.testing.assert_close(scores.mean(dim=0), torch.zeros(3), atol=1e-4, rtol=0)
    torch.testing.assert_close(scores.std(dim=0), torch.ones(3), atol=1e-4, rtol=0)
    assert gensim.score_mean_ is None
    assert gensim.score_sd_ is None


@pytest.mark.parametrize("keep_sparse", [False, True])
def test_a_binomial_fit_takes_its_number_of_trials(keep_sparse: bool) -> None:
    X = torch.poisson(torch.full((N_CELLS, N_FEATURES), 3.0))
    model = GLMPCA(
        N_PC,
        family="binomial",
        family_params={"n_trials": 6},
        max_iter=2,
        batch_size=16,
        keep_sparse=keep_sparse,
    )

    model.fit(X.numpy())

    assert model.exponential_family.family_params["n_clipped"] == int((X > 6).sum())
    assert model.log_likelihood_ is not None
    assert np.isfinite(model.log_likelihood_)


@pytest.mark.parametrize(
    ("family", "memory", "chosen", "report"),
    [
        (GLMFamily.poisson, 10**15, False, "STORAGE: dense"),
        (GLMFamily.poisson, 10, True, "STORAGE: sparse"),
        (GLMFamily.gamma, 10, False, ""),
    ],
)
def test_keep_sparse_none_chooses_the_storage_from_the_memory(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    family: GLMFamily,
    memory: int,
    chosen: bool,
    report: str,
) -> None:
    monkeypatch.setattr("glmpca.GLMPCA._host_memory", lambda: memory)
    monkeypatch.setattr(
        GLMPCA, "_rows_that_fit", lambda self, X, device: (10**9, "memory")
    )
    model = GLMPCA(N_PC, family=family, max_iter=1, batch_size=16)

    model.fit(sample(family))

    assert model.keep_sparse is None
    assert model.keep_sparse_ is chosen
    assert report in capsys.readouterr().out


@pytest.mark.parametrize(
    ("gpu", "chosen", "report"),
    [
        (10**15, False, "allowed (host memory))"),
        (10**3, True, "allowed (GPU memory))"),
    ],
)
def test_on_cuda_the_storage_is_dense_only_if_it_also_fits_on_the_gpu(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    gpu: int,
    chosen: bool,
    report: str,
) -> None:
    monkeypatch.setattr("glmpca.GLMPCA._host_memory", lambda: 10**12)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (gpu, gpu))
    model = GLMPCA(N_PC, family="poisson", batch_size=16)

    sparse = model._use_sparse(sample(GLMFamily.poisson), torch.device("cuda"))

    assert sparse is chosen
    assert report in capsys.readouterr().out


def test_the_free_gpu_memory_counts_what_the_process_caches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (100, 1000))
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device=None: 70)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device=None: 20)

    assert _free_device_memory(torch.device("cuda")) == 150


def test_the_depth_passes_take_smaller_chunks_and_give_the_same_offsets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X = sample(GLMFamily.poisson)
    sizes: list[int] = []
    original = glmpca_module._fitted_depth

    def recording(
        family: ExponentialFamily, data: torch.Tensor, *args: torch.Tensor
    ) -> torch.Tensor:
        sizes.append(data.shape[0])
        return original(family, data, *args)

    whole = GLMPCA(N_PC, family="poisson", max_iter=2, batch_size=8, depth_factor=True)
    whole.fit(X)
    monkeypatch.setattr(glmpca_module, "_fitted_depth", recording)
    chunked = GLMPCA(
        N_PC,
        family="poisson",
        max_iter=2,
        batch_size=8,
        depth_factor=True,
        chunk_size=12,
    )
    chunked.fit(X)
    fitted = len(sizes)
    chunked.transform(X)

    assert max(sizes) == 12 * WORKING_COPIES // DEPTH_WORKING_COPIES
    assert fitted < len(sizes)
    torch.testing.assert_close(
        chunked.saturated_depth_, whole.saturated_depth_, atol=1e-4, rtol=1e-4
    )


def test_transform_does_not_report_the_storage_again(
    capsys: pytest.CaptureFixture[str],
) -> None:
    X = sample(GLMFamily.poisson)
    model = GLMPCA(N_PC, family="poisson", max_iter=1, batch_size=16)
    model.fit(X)
    assert "STORAGE:" in capsys.readouterr().out

    model.transform(X)

    assert "STORAGE:" not in capsys.readouterr().out


@pytest.mark.parametrize(("keep_sparse", "memory"), [(True, 10**15), (False, 10)])
def test_an_explicit_keep_sparse_wins_over_the_memory(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    keep_sparse: bool,
    memory: int,
) -> None:
    monkeypatch.setattr("glmpca.GLMPCA._host_memory", lambda: memory)
    monkeypatch.setattr(
        GLMPCA, "_rows_that_fit", lambda self, X, device: (10**9, "memory")
    )
    model = GLMPCA(
        N_PC, family="poisson", max_iter=1, batch_size=16, keep_sparse=keep_sparse
    )

    model.fit(sample(GLMFamily.poisson))

    assert model.keep_sparse_ is keep_sparse
    assert "STORAGE:" not in capsys.readouterr().out


def test_the_host_memory_is_the_lower_of_the_ram_and_the_cgroup_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "glmpca.GLMPCA.os.sysconf",
        lambda name: {"SC_PAGE_SIZE": 4096, "SC_PHYS_PAGES": 10**6}[name],
    )

    def read_text(path: Path) -> str:
        if str(path) == "/sys/fs/cgroup/memory.max":
            return "1000000\n"
        raise OSError

    monkeypatch.setattr("glmpca.GLMPCA.Path.read_text", read_text)

    assert _host_memory() == 1_000_000


def test_a_given_batch_size_is_used_and_reported(
    capsys: pytest.CaptureFixture[str],
) -> None:
    model = GLMPCA(N_PC, family="poisson", max_iter=1, batch_size=16)

    model.fit(sample(GLMFamily.poisson))

    assert model.batch_size_ == 16
    output = capsys.readouterr().out
    assert "BATCH SIZE: 16\n" in output
    assert "as many cells as fit" not in output


def test_the_default_random_state_makes_two_fits_equal() -> None:
    X = sample(GLMFamily.poisson)
    loadings = []
    for outside_seed in (1, 2):
        torch.manual_seed(outside_seed)
        np.random.seed(outside_seed)
        model = GLMPCA(N_PC, family="poisson", max_iter=3, batch_size=8)
        model.fit(X)
        assert model.random_state == 42
        loadings.append(model.saturated_loadings_)

    torch.testing.assert_close(loadings[0], loadings[1], atol=0, rtol=0)


def test_fit_leaves_the_global_random_state_of_the_caller_alone() -> None:
    X = sample(GLMFamily.poisson)
    torch_state = torch.random.get_rng_state()
    numpy_state = np.random.get_state()

    GLMPCA(N_PC, family="poisson", max_iter=2, batch_size=8).fit(X)

    torch.testing.assert_close(torch.random.get_rng_state(), torch_state)
    after = np.random.get_state()
    assert after[0] == numpy_state[0]
    np.testing.assert_array_equal(after[1], numpy_state[1])


def test_no_random_state_gives_fits_that_differ() -> None:
    X = sample(GLMFamily.poisson)
    fits = [
        GLMPCA(N_PC, family="poisson", max_iter=3, batch_size=8, random_state=None)
        for _ in range(2)
    ]
    for model in fits:
        model.fit(X)

    assert not torch.equal(fits[0].saturated_loadings_, fits[1].saturated_loadings_)


def backed_counts(
    tmp_path: Path, *, dense: bool = False
) -> tuple[ad.AnnData, torch.Tensor]:
    X = sample(GLMFamily.poisson)
    path = tmp_path / "counts.h5ad"
    ad.AnnData(X.numpy() if dense else sparse.csr_matrix(X.numpy())).write_h5ad(path)
    return ad.read_h5ad(path, backed="r"), X


@pytest.mark.parametrize("dense", [False, True])
def test_backed_rows_read_what_sparse_rows_read(tmp_path: Path, dense: bool) -> None:
    adata, X = backed_counts(tmp_path, dense=dense)
    backed = BackedRows(adata.X)
    in_memory = SparseRows(sparse.csr_matrix(X.numpy()))
    rows = torch.tensor([7, 2, 30, 11])

    assert backed.nbytes == 0
    assert backed.shape == in_memory.shape
    torch.testing.assert_close(backed[3:9], in_memory[3:9])
    torch.testing.assert_close(backed[rows], in_memory[rows])


def test_a_csr_block_is_densified_from_its_non_zeros() -> None:
    X = sample(GLMFamily.poisson)

    block = densified(sparse.csr_matrix(X.numpy()[[7, 2, 30]]), torch.device("cpu"))

    torch.testing.assert_close(block, X[[7, 2, 30]])


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("backed", [False, True])
def test_off_the_cpu_a_sparse_block_is_densified_on_the_device(
    tmp_path: Path, backed: bool
) -> None:
    adata, X = backed_counts(tmp_path)
    source = BackedRows(adata.X) if backed else SparseRows(sparse.csr_matrix(X.numpy()))
    rows = torch.tensor([7, 2, 30, 11])

    for block in (slice(3, 9), rows):
        dense = source.dense(block, torch.device("mps"))
        assert dense.device.type == "mps"
        torch.testing.assert_close(dense.cpu(), X[block])


def test_binary_reads_every_non_zero_value_as_1_and_leaves_its_input() -> None:
    X = sample(GLMFamily.poisson).numpy()
    stored_zero = sparse.csr_matrix(
        (np.array([0.0, 3.0]), np.array([0, 1]), np.array([0, 2])), shape=(1, 2)
    )
    csr = sparse.csr_matrix(X)

    np.testing.assert_array_equal(binary(X), X != 0)
    np.testing.assert_array_equal(binary(csr).toarray(), X != 0)
    np.testing.assert_array_equal(binary(stored_zero).toarray(), [[0, 1]])
    np.testing.assert_array_equal(csr.toarray(), X)


@pytest.mark.parametrize("dense", [False, True])
def test_backed_rows_binarize_every_block(tmp_path: Path, dense: bool) -> None:
    adata, X = backed_counts(tmp_path, dense=dense)
    backed = BackedRows(adata.X, binarize=True)
    rows = torch.tensor([7, 2, 30, 11])

    torch.testing.assert_close(backed[3:9], (X[3:9] != 0).float())
    torch.testing.assert_close(backed[rows], (X[rows] != 0).float())


@pytest.mark.parametrize("keep_sparse", [False, True])
def test_a_binarized_fit_equals_a_fit_on_the_binary_matrix(keep_sparse: bool) -> None:
    X = sample(GLMFamily.poisson)
    original = X.clone()
    options = {"max_iter": 3, "batch_size": 8, "keep_sparse": keep_sparse}
    binarized = GLMPCA(N_PC, family="poisson", binarize=True, **options)
    binarized.fit(X)
    expected = GLMPCA(N_PC, family="poisson", **options)
    expected.fit((X != 0).float())

    torch.testing.assert_close(X, original)
    assert binarized.log_likelihood_ == pytest.approx(expected.log_likelihood_)
    torch.testing.assert_close(binarized.transform(X), expected.transform(X != 0))


def test_a_backed_binarized_fit_equals_one_in_memory(tmp_path: Path) -> None:
    adata, X = backed_counts(tmp_path)
    options = {"max_iter": 3, "batch_size": 8, "binarize": True}
    backed = GLMPCA(N_PC, family="poisson", **options)
    backed.fit(adata)
    in_memory = GLMPCA(N_PC, family="poisson", keep_sparse=True, **options)
    in_memory.fit(X)

    assert backed.log_likelihood_ == pytest.approx(in_memory.log_likelihood_, rel=1e-6)
    torch.testing.assert_close(
        backed.transform(adata), in_memory.transform(X), atol=1e-4, rtol=1e-4
    )


def test_a_backed_fit_equals_a_sparse_fit_in_memory(tmp_path: Path) -> None:
    adata, X = backed_counts(tmp_path)
    backed = GLMPCA(N_PC, family="poisson", max_iter=3, batch_size=8)
    backed.fit(adata)
    in_memory = GLMPCA(
        N_PC, family="poisson", max_iter=3, batch_size=8, keep_sparse=True
    )
    in_memory.fit(X)

    assert backed.keep_sparse_ is True
    assert backed.log_likelihood_ == pytest.approx(in_memory.log_likelihood_, rel=1e-6)
    torch.testing.assert_close(
        backed.saturated_loadings_, in_memory.saturated_loadings_, atol=1e-5, rtol=0
    )
    torch.testing.assert_close(
        backed.transform(adata),
        in_memory.transform(X),
        atol=1e-4,
        rtol=1e-4,
    )


def test_a_backed_fit_rejects_a_family_that_reads_whole_columns(
    tmp_path: Path,
) -> None:
    adata, _ = backed_counts(tmp_path)
    model = GLMPCA(N_PC, family="gamma", max_iter=1)

    with pytest.raises(ValueError, match="cannot take"):
        model.fit(adata)


def test_keep_sparse_false_loads_a_backed_anndata(tmp_path: Path) -> None:
    adata, _ = backed_counts(tmp_path)
    model = GLMPCA(N_PC, family="poisson", max_iter=1, batch_size=8, keep_sparse=False)

    model.fit(adata)

    assert model.keep_sparse_ is False


def test_the_batch_size_defaults_to_4096_and_the_chunk_size_to_automatic() -> None:
    model = GLMPCA(N_PC, family="poisson")

    assert model.batch_size == DEFAULT_BATCH_SIZE == 4096
    assert model.chunk_size is None


def test_a_batch_that_does_not_fit_in_memory_stops_with_the_largest_that_does(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("glmpca.GLMPCA._host_memory", lambda: 15_360)
    model = GLMPCA(N_PC, family="poisson", max_iter=1, batch_size=16)

    with pytest.raises(ValueError, match="The largest batch that fits is 10"):
        model.fit(sample(GLMFamily.poisson))


def test_a_batch_that_fits_in_memory_is_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("glmpca.GLMPCA._host_memory", lambda: 15_360)
    model = GLMPCA(N_PC, family="poisson", max_iter=1, batch_size=8)

    model.fit(sample(GLMFamily.poisson))

    assert model.keep_sparse_ is False
    assert model.batch_size_ == 8


@pytest.mark.parametrize(
    ("host", "gpu", "limit"),
    [(2**40, 2**20, "GPU memory"), (2**20, 2**40, "host memory")],
)
def test_on_cuda_a_batch_fits_in_both_the_gpu_and_the_host_memory(
    monkeypatch: pytest.MonkeyPatch, host: int, gpu: int, limit: str
) -> None:
    monkeypatch.setattr("glmpca.GLMPCA._host_memory", lambda: host)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (gpu, gpu))
    X = torch.zeros(10, 5000)
    row_bytes = 4 * 5000
    on_gpu = int(DEVICE_MEMORY_SHARE * gpu) // (WORKING_COPIES * row_bytes)
    staged = int(HOST_MEMORY_SHARE * host - 2 * X.nbytes) // (
        HOST_STAGING_COPIES * row_bytes
    )

    rows, memory = GLMPCA(N_PC)._rows_that_fit(X, torch.device("cuda"))

    assert rows == min(on_gpu, staged)
    assert limit in memory


def test_on_the_cpu_a_pass_takes_the_batch_size_unless_a_chunk_size_is_given() -> None:
    automatic = GLMPCA(N_PC, family="poisson", max_iter=1, batch_size=8)
    automatic.fit(sample(GLMFamily.poisson))
    given = GLMPCA(N_PC, family="poisson", max_iter=1, batch_size=8, chunk_size=5)
    given.fit(sample(GLMFamily.poisson))

    assert automatic._chunk == 8
    assert automatic.exponential_family.family_params["chunk_size"] == 8
    assert given._chunk == 5
    assert given.exponential_family.family_params["chunk_size"] == 5


def test_on_cuda_a_pass_takes_the_rows_that_the_free_memory_holds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    free = 10 * 2**30
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda device=None: (free, 2 * free)
    )
    model = GLMPCA(N_PC, family="poisson", batch_size=8)
    cuda = torch.device("cuda")

    rows = model._chunk_rows(cuda, n_rows=10**9, row_bytes=4 * 5000)

    assert rows == int(DEVICE_MEMORY_SHARE * free) // (WORKING_COPIES * 4 * 5000)
    assert model._chunk_rows(cuda, n_rows=100, row_bytes=4 * 5000) == 100
    assert model._chunk_rows(torch.device("cpu"), n_rows=10**9, row_bytes=4) == 8


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_a_chunk_size_below_one_is_rejected(chunk_size: int) -> None:
    model = GLMPCA(
        N_PC, family="poisson", max_iter=1, batch_size=8, chunk_size=chunk_size
    )

    with pytest.raises(ValueError, match="chunk_size"):
        model.fit(sample(GLMFamily.poisson))


@pytest.mark.parametrize("family", list(GLMFamily))
def test_every_family_fits_with_adam(family: GLMFamily) -> None:
    model = GLMPCA(
        N_PC, family=family, optimizer="adam", max_iter=2, batch_size=16, n_jobs=1
    )

    assert model.fit(sample(family))
    assert model.saturated_loadings_ is not None
    torch.testing.assert_close(
        model.saturated_loadings_.T @ model.saturated_loadings_,
        torch.eye(N_PC),
        atol=1e-5,
        rtol=0,
    )


@pytest.mark.parametrize("optimizer", ["adamw", ""])
def test_an_unknown_optimizer_is_rejected(optimizer: str) -> None:
    model = GLMPCA(N_PC, family="poisson", optimizer=optimizer)  # ty: ignore[invalid-argument-type]

    with pytest.raises(ValueError, match="Use one of 'adagrad', 'adam', 'cg'"):
        model.fit(sample(GLMFamily.poisson))


def depth_gradient() -> torch.Tensor:
    """Counts whose cells differ in depth by two orders of magnitude."""
    torch.manual_seed(0)
    depth = torch.exp(torch.randn(N_CELLS) * 0.8).unsqueeze(1)
    structure = torch.randn(N_CELLS, N_PC) @ torch.randn(N_PC, N_FEATURES) * 0.3
    return torch.poisson(torch.exp(structure) * depth)


def test_the_depth_factor_is_fitted_for_every_cell() -> None:
    model = GLMPCA(N_PC, family="poisson", max_iter=5, batch_size=16, depth_factor=True)

    model.fit(depth_gradient())

    assert model.saturated_depth_ is not None
    assert model.saturated_depth_.shape == (N_CELLS,)
    assert torch.all(torch.isfinite(model.saturated_depth_))


def test_no_depth_factor_by_default_leaves_no_offset() -> None:
    model = GLMPCA(N_PC, family="poisson", max_iter=5, batch_size=16)

    model.fit(depth_gradient())

    assert model.saturated_depth_ is None


def test_the_depth_factor_keeps_the_components_off_the_depth() -> None:
    X = depth_gradient()
    totals = X.sum(dim=1).numpy()
    worst = {}
    for depth_factor in (True, False):
        torch.manual_seed(0)
        np.random.seed(0)
        model = GLMPCA(
            N_PC,
            family="poisson",
            max_iter=40,
            batch_size=16,
            depth_factor=depth_factor,
        )
        model.fit(X)
        embedding = model.transform(X).detach().numpy()
        worst[depth_factor] = max(
            abs(float(scipy.stats.spearmanr(embedding[:, pc], totals).statistic))
            for pc in range(N_PC)
        )

    assert worst[True] < worst[False], worst


def test_the_depth_factor_has_its_own_learning_rate() -> None:
    model = GLMPCA(N_PC, family="poisson", max_iter=2, batch_size=16, depth_factor=True)
    saturated = torch.log(depth_gradient().clip(min=1.0))

    optimizer, _, _, depth, _ = model._init_saturated_loading_optim(
        _Rows(saturated, model.exponential_family, saturated),
        torch.device("cpu"),
        batch_size=16,
    )

    assert depth is not None
    rates = [group["lr"] for group in optimizer.param_groups]
    assert rates == [
        model.learning_rate_,
        model.learning_rate_ * INTERCEPT_RATE_SCALE,
        model.learning_rate_ * DEPTH_RATE_SCALE,
    ]


def test_transform_reproduces_the_fitted_scores_with_a_depth_factor() -> None:
    X = depth_gradient()
    model = GLMPCA(
        N_PC, family="poisson", max_iter=20, batch_size=16, depth_factor=True
    )
    model.fit(X)

    embedding = model.transform(X).detach()

    depth, intercept = model.saturated_depth_, model.saturated_intercept_
    assert depth is not None
    assert intercept is not None
    assert model.saturated_loadings_ is not None
    fitted = (
        model.exponential_family.invert_g(X)
        - intercept.unsqueeze(0)
        - depth.unsqueeze(1)
    ) @ model.saturated_loadings_
    torch.testing.assert_close(embedding, fitted, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("family", ["poisson", "negative_binomial", "gaussian"])
def test_the_offset_of_a_cell_is_the_optimum_of_its_likelihood(family: str) -> None:
    X = depth_gradient()
    model = GLMPCA(N_PC, family=family, max_iter=10, batch_size=16, depth_factor=True)
    model.fit(X)
    depth, intercept = model.saturated_depth_, model.saturated_intercept_
    loadings = model.saturated_loadings_
    assert depth is not None
    assert intercept is not None
    assert loadings is not None
    centered = model.exponential_family.invert_g(X) - intercept.unsqueeze(0)
    projector = loadings @ loadings.T
    outside = torch.ones(N_FEATURES) - projector.sum(dim=1)

    def costs(offset: torch.Tensor) -> torch.Tensor:
        theta = centered @ projector + intercept + offset.unsqueeze(1) * outside
        return -(
            model.exponential_family.exponential_term(X, theta)
            - model.exponential_family.log_partition(theta)
        ).sum(dim=1)

    at_optimum = costs(depth)
    for shift in (-1e-2, 1e-2):
        assert torch.all(costs(depth + shift) >= at_optimum - 1e-3)


@pytest.mark.parametrize(
    "family", ["gaussian", "poisson", "negative_binomial", "bernoulli"]
)
@pytest.mark.parametrize("optimizer", ["adagrad", "cg"])
def test_keep_sparse_fits_what_the_dense_matrix_fits(
    family: str, optimizer: Literal["adagrad", "cg"]
) -> None:
    X = sample(GLMFamily(family)) * (torch.rand(N_CELLS, N_FEATURES) < 0.3)
    fits = {}
    for keep_sparse in (False, True):
        torch.manual_seed(0)
        np.random.seed(0)
        model = GLMPCA(
            N_PC,
            family=family,
            max_iter=5,
            batch_size=16,
            chunk_size=15,
            optimizer=optimizer,
            keep_sparse=keep_sparse,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(ad.AnnData(sparse.csr_matrix(X.numpy())) if keep_sparse else X)
        fits[keep_sparse] = model

    dense, kept = fits[False], fits[True]
    assert dense.saturated_loadings_ is not None
    assert kept.saturated_loadings_ is not None
    torch.testing.assert_close(kept.saturated_loadings_, dense.saturated_loadings_)
    torch.testing.assert_close(kept.saturated_intercept_, dense.saturated_intercept_)
    torch.testing.assert_close(kept.saturated_depth_, dense.saturated_depth_)
    assert kept.log_likelihood_ == pytest.approx(dense.log_likelihood_, rel=1e-6)
    torch.testing.assert_close(kept.transform(X), dense.transform(X))


def test_keep_sparse_reads_a_sparse_anndata() -> None:
    X = torch.poisson(torch.full((N_CELLS, N_FEATURES), 0.5))
    fits = {}
    for keep_sparse in (False, True):
        torch.manual_seed(0)
        np.random.seed(0)
        model = GLMPCA(
            N_PC,
            family="gaussian",
            max_iter=5,
            batch_size=16,
            keep_sparse=keep_sparse,
        )
        model.fit(ad.AnnData(sparse.csr_matrix(X.numpy())))
        fits[keep_sparse] = model

    dense, kept = fits[False], fits[True]
    assert dense.saturated_loadings_ is not None
    assert kept.saturated_loadings_ is not None
    torch.testing.assert_close(
        kept.saturated_loadings_, dense.saturated_loadings_, rtol=1e-4, atol=1e-4
    )


@pytest.mark.parametrize("family", ["beta", "sigmoid_beta", "gamma", "lognormal"])
def test_keep_sparse_rejects_the_families_without_zeros(family: str) -> None:
    model = GLMPCA(N_PC, family=family, max_iter=1, keep_sparse=True)

    with pytest.raises(ValueError, match="keep_sparse=True does not fit"):
        model.fit(sample(GLMFamily(family)))


def test_sparse_rows_hand_out_dense_blocks() -> None:
    matrix = sparse.random(30, 8, density=0.2, format="csr", dtype=np.float32, rng=0)
    rows = SparseRows(matrix)
    order = torch.tensor([4, 0, 29, 4])

    assert rows.shape == (30, 8)
    torch.testing.assert_close(rows[3:9], torch.from_numpy(matrix[3:9].toarray()))
    torch.testing.assert_close(rows[order], torch.from_numpy(matrix.toarray())[order])
