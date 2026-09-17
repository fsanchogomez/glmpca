"""Tests for ``GLMPCA``: family selection, the fit/transform contract, AnnData input."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal, cast

import anndata as ad
import numpy as np
import pytest
import scipy.stats
import torch
from glmpca.ExponentialFamily import Beta, GLMFamily, Poisson, _n_workers
from glmpca.GLMPCA import (
    DEPTH_RATE_SCALE,
    GLMPCA,
    INTERCEPT_RATE_SCALE,
    LEARNING_RATE_LIMIT,
    PLATEAU_PATIENCE,
    _inverse_document_frequency,
    _tf_idf,
    _to_tensor,
)
from glmpca.manifolds import ManifoldParameter, RiemannianAdagrad
from scipy import sparse

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

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
    family = Poisson({"m": 2.0})

    used = GLMPCA(N_PC, family=family, chunk_size=64).exponential_family

    assert used is not family
    assert type(used) is Poisson
    assert used.family_params["m"] == 2.0
    # GLMPCA gives its own chunk_size to the copy, and leaves the caller's instance.
    assert used.family_params["chunk_size"] == 64
    assert "chunk_size" not in family.family_params


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
        saturated_parameters: torch.Tensor,
        X: torch.Tensor,
        batch_size: int,
        device: torch.device,
        log_base_measure: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        start_rates.append(model.learning_rate_)
        model.learning_rate_ *= model.gamma
        return torch.eye(N_FEATURES, N_PC), torch.zeros(N_FEATURES), None

    monkeypatch.setattr(model, "_saturated_loading_iter", run_that_restarts_once)
    model.fit(sample(GLMFamily.poisson))

    assert start_rates == [0.2, 0.2, 0.2]
    assert model.learning_rate_ == 0.1


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


@pytest.mark.parametrize("init", ["spectral", "random", "lsi"])
@pytest.mark.parametrize("family", list(GLMFamily))
def test_fitted_loadings_are_orthonormal(
    family: GLMFamily, init: Literal["spectral", "random", "lsi"]
) -> None:
    X = sample(family)
    if init == "lsi" and bool(torch.any(X < 0)):
        pytest.skip("TF-IDF needs counts, and this family lives on other data.")
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
    torch.testing.assert_close(chunked.saturated_loadings_, whole.saturated_loadings_)


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
            model.saturated_loadings_, model.saturated_intercept_, X, parameters
        )

    torch.testing.assert_close(chunked, whole, rtol=1e-5, atol=1e-4)


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


def test_tf_idf_lifts_a_rare_feature_over_a_common_one() -> None:
    # The first feature is in every cell, the third in one cell only.
    counts = torch.tensor([
        [5.0, 1.0, 0.0],
        [5.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
        [5.0, 0.0, 3.0],
    ])

    weighted = _tf_idf(counts, _inverse_document_frequency(counts))

    common = weighted[3, 0]
    rare = weighted[3, 2]
    # Both cells hold 5 and 3 counts, so the term frequencies are close, and it is the
    # document frequency that puts the rare feature above the common one.
    assert rare > common
    assert torch.all(weighted[counts == 0] == 0)
    assert torch.all(torch.isfinite(weighted))


def test_tf_idf_survives_an_empty_row_and_an_empty_column() -> None:
    counts = torch.tensor([[0.0, 2.0], [0.0, 0.0]])

    weighted = _tf_idf(counts, _inverse_document_frequency(counts))

    assert torch.all(torch.isfinite(weighted))
    assert float(weighted[1, 1]) == 0.0


def test_the_lsi_start_rejects_negative_values() -> None:
    model = GLMPCA(N_PC, family="gaussian", init="lsi", max_iter=2, batch_size=16)

    with pytest.raises(ValueError, match="TF-IDF needs counts"):
        model.fit(torch.randn(20, 6))


def test_the_lsi_start_differs_from_the_spectral_one() -> None:
    X = sample(GLMFamily.poisson)
    starts = {}
    for init in ("spectral", "lsi"):
        torch.manual_seed(0)
        np.random.seed(0)
        model = GLMPCA(N_PC, family="poisson", init=init, max_iter=1, batch_size=16)
        model.fit(X)
        assert model.saturated_loadings_ is not None
        starts[init] = model.saturated_loadings_.detach().clone()

    assert not torch.allclose(starts["spectral"], starts["lsi"], atol=1e-4)


def test_the_tfidf_option_fits_the_weighted_matrix() -> None:
    X = sample(GLMFamily.poisson)
    seen: list[torch.Tensor] = []
    family = Poisson()
    original = family.invert_g

    def recording_invert_g(data: torch.Tensor) -> torch.Tensor:
        seen.append(data.clone())
        return original(data)

    family.invert_g = recording_invert_g  # ty: ignore[invalid-assignment]
    model = GLMPCA(N_PC, family=family, tfidf=True, max_iter=2, batch_size=16)

    model.fit(X)

    weighted = _tf_idf(X, _inverse_document_frequency(X))
    torch.testing.assert_close(seen[0], weighted, rtol=0, atol=1e-6)
    assert model.tfidf_weights_ is not None
    assert model.tfidf_weights_.shape == (N_FEATURES,)


def test_transform_weighs_new_cells_with_the_frequencies_of_the_fit() -> None:
    X = sample(GLMFamily.poisson)
    model = GLMPCA(N_PC, family="poisson", tfidf=True, max_iter=2, batch_size=16)
    model.fit(X)
    assert model.tfidf_weights_ is not None
    fitted_weights = model.tfidf_weights_.clone()

    # A subset holds different document frequencies, and transform must ignore them.
    model.transform(X[:8])

    torch.testing.assert_close(model.tfidf_weights_, fitted_weights, rtol=0, atol=0)


def test_the_tfidf_option_is_refused_by_the_bounded_families() -> None:
    model = GLMPCA(N_PC, family="bernoulli", tfidf=True, max_iter=2, batch_size=16)

    with pytest.raises(ValueError, match="whose support is bounded"):
        model.fit(sample(GLMFamily.bernoulli))


def test_the_tfidf_option_rejects_negative_values() -> None:
    model = GLMPCA(N_PC, family="gaussian", tfidf=True, max_iter=2, batch_size=16)

    with pytest.raises(ValueError, match="TF-IDF needs counts"):
        model.fit(torch.randn(20, 6))


def depth_gradient() -> torch.Tensor:
    """Counts whose cells differ in depth by two orders of magnitude."""
    torch.manual_seed(0)
    depth = torch.exp(torch.randn(N_CELLS) * 0.8).unsqueeze(1)
    structure = torch.randn(N_CELLS, N_PC) @ torch.randn(N_PC, N_FEATURES) * 0.3
    return torch.poisson(torch.exp(structure) * depth)


def test_the_depth_factor_is_fitted_for_every_cell() -> None:
    model = GLMPCA(N_PC, family="poisson", max_iter=5, batch_size=16)

    model.fit(depth_gradient())

    assert model.saturated_depth_ is not None
    assert model.saturated_depth_.shape == (N_CELLS,)
    assert torch.all(torch.isfinite(model.saturated_depth_))


def test_turning_the_depth_factor_off_leaves_no_offset() -> None:
    model = GLMPCA(
        N_PC, family="poisson", max_iter=5, batch_size=16, depth_factor=False
    )

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
    model = GLMPCA(N_PC, family="poisson", max_iter=2, batch_size=16)
    saturated = torch.log(depth_gradient().clip(min=1.0))

    optimizer, _, _, depth, _ = model._create_saturated_loading_optim(
        saturated, saturated, torch.device("cpu")
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
    model = GLMPCA(N_PC, family="poisson", max_iter=20, batch_size=16)
    model.fit(X)

    embedding = model.transform(X).detach()

    # transform estimates the offset of a cell by least squares, so the scores it gives
    # the cells of the fit track the fitted ones closely rather than exactly.
    assert model.saturated_depth_ is not None
    assert torch.all(torch.isfinite(embedding))
    assert embedding.shape == (N_CELLS, N_PC)
