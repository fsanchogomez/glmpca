"""Check LSA against the exact truncated SVD of the TF-IDF matrix."""

from __future__ import annotations

from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pytest
import scipy.stats
import torch
from glmpca.LSA import LSA, TFIDF_SCALE, idf, tfidf
from scipy import sparse

if TYPE_CHECKING:
    from pathlib import Path

N_PC = 4
DEFINED = 3
"""The components that the three groups of `grouped_counts` define; the next one is
noise, and two solvers need not agree on it."""
FLAVORS = ["signac", "gensim"]
SOLVERS = ["arpack", "randomized"]


def grouped_counts(depth_spread: float = 0.3) -> sparse.csr_matrix:
    """Counts of three groups of cells, each with features of its own."""
    rng = np.random.default_rng(0)
    n_cells, n_features = 240, 60
    group = np.repeat(np.arange(3), n_cells // 3)
    rates = np.full((3, n_features), 0.05)
    for g in range(3):
        rates[g, g * 20 : (g + 1) * 20] = 0.8
    depth = np.exp(rng.normal(0.0, depth_spread, size=n_cells))
    counts = rng.poisson(rates[group] * depth[:, None]).astype(np.float32)
    return sparse.csr_matrix(counts)


def tfidf_of_fit(model: LSA, X: sparse.csr_matrix) -> np.ndarray:
    assert model.idf_ is not None
    return tfidf(X, model.flavor, model.idf_, model.pivot_).toarray().astype(np.float64)


def exact_singular_values(model: LSA, X: sparse.csr_matrix) -> np.ndarray:
    return np.linalg.svd(tfidf_of_fit(model, X), compute_uv=False)


def captured(model: LSA, X: sparse.csr_matrix) -> float:
    """`||W V||^2` over its optimum, the sum of the top squared singular values."""
    assert model.components_ is not None
    components = model.components_.numpy().astype(np.float64)
    best = exact_singular_values(model, X)[: components.shape[1]]
    product = tfidf_of_fit(model, X) @ components
    return float(np.square(product).sum() / np.square(best).sum())


LSI_COUNTS = np.array(
    [
        [1, 0, 3, 2, 0],
        [0, 2, 1, 4, 0],
        [5, 1, 0, 1, 0],
        [2, 0, 0, 3, 0],
        [0, 0, 7, 1, 0],
        [1, 1, 1, 1, 0],
    ],
    dtype=np.float32,
)
"""Feature 3 is held by every cell and feature 4 by none."""

GENSIM_WEIGHTS = np.array([
    [0.2228428574, 0.0, 0.57604043, 0.0, 0.0],
    [0.0, 0.7619047619, 0.2228428574, 0.0, 0.0],
    [0.7402679488, 0.380952381, 0.0, 0.0, 0.0],
    [0.4926000006, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.9377515185, 0.0, 0.0],
    [0.2034652176, 0.347826087, 0.2034652176, 0.0, 0.0],
])
"""gensim 4.4.0, `TfidfModel(corpus, normalize=True, smartirs="lfu")`, on LSI_COUNTS."""


def tfidf_of_counts(flavor: str, counts: np.ndarray) -> np.ndarray:
    feature_idf, pivot = idf(
        flavor, counts.sum(axis=0), (counts > 0).sum(axis=0), counts.shape[0]
    )
    return tfidf(sparse.csr_matrix(counts), flavor, feature_idf, pivot).toarray()


def test_the_gensim_tfidf_is_the_tfidf_of_gensim() -> None:
    np.testing.assert_allclose(
        tfidf_of_counts("gensim", LSI_COUNTS), GENSIM_WEIGHTS, atol=1e-6
    )


def test_the_signac_tfidf_is_the_tfidf_of_signac_runtfidf() -> None:
    depth = LSI_COUNTS.sum(axis=1, keepdims=True)
    feature_idf = LSI_COUNTS.shape[0] / np.maximum(LSI_COUNTS.sum(axis=0), 1.0)
    expected = np.log1p(LSI_COUNTS / depth * feature_idf * TFIDF_SCALE)

    np.testing.assert_allclose(
        tfidf_of_counts("signac", LSI_COUNTS), expected, rtol=1e-6
    )


@pytest.mark.parametrize("flavor", FLAVORS)
def test_the_tfidf_keeps_the_entries_of_the_counts(flavor: str) -> None:
    counts = sparse.csr_matrix(LSI_COUNTS)
    feature_idf, pivot = idf(
        flavor, LSI_COUNTS.sum(axis=0), (LSI_COUNTS > 0).sum(axis=0), 6
    )

    result = tfidf(counts, flavor, feature_idf, pivot)

    assert sparse.isspmatrix_csr(result)
    np.testing.assert_array_equal(result.indices, counts.indices)
    np.testing.assert_array_equal(result.indptr, counts.indptr)


@pytest.mark.parametrize("flavor", FLAVORS)
def test_the_fit_keeps_the_idf_of_the_counts(flavor: str) -> None:
    X = grouped_counts()
    model = LSA(N_PC, flavor=flavor, device="cpu")

    model.fit(X)

    dense = X.toarray()
    feature_idf, pivot = idf(
        flavor, dense.sum(axis=0), (dense > 0).sum(axis=0), dense.shape[0]
    )
    assert model.idf_ is not None
    np.testing.assert_allclose(model.idf_, feature_idf)
    assert model.pivot_ == (None if flavor == "signac" else pytest.approx(pivot))


@pytest.mark.parametrize("flavor", FLAVORS)
def test_arpack_gives_the_exact_singular_values(flavor: str) -> None:
    X = grouped_counts()
    model = LSA(N_PC, flavor=flavor, solver="arpack", device="cpu")

    model.fit(X)

    assert model.singular_values_ is not None
    np.testing.assert_allclose(
        model.singular_values_.numpy(),
        exact_singular_values(model, X)[:N_PC],
        rtol=1e-4,
    )
    assert captured(model, X) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("flavor", FLAVORS)
def test_the_randomized_solver_reaches_the_exact_subspace(flavor: str) -> None:
    X = grouped_counts()
    model = LSA(N_PC, flavor=flavor, solver="randomized", device="cpu")

    model.fit(X)

    assert captured(model, X) > 0.999


@pytest.mark.parametrize("flavor", FLAVORS)
@pytest.mark.parametrize("solver", SOLVERS)
def test_transform_gives_the_scores_of_the_fit(flavor: str, solver: str) -> None:
    X = grouped_counts()
    model = LSA(N_PC, flavor=flavor, solver=solver, device="cpu")

    fitted = model.fit_transform(X)

    torch.testing.assert_close(model.transform(X), fitted, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("solver", SOLVERS)
def test_signac_scores_are_standardized_and_gensim_scores_are_w_v(
    solver: str,
) -> None:
    X = grouped_counts()
    signac = LSA(N_PC, flavor="signac", solver=solver, device="cpu")
    gensim = LSA(N_PC, flavor="gensim", solver=solver, device="cpu")

    signac_scores = signac.fit_transform(X)
    gensim_scores = gensim.fit_transform(X)

    torch.testing.assert_close(
        signac_scores.mean(dim=0), torch.zeros(N_PC), atol=1e-5, rtol=0
    )
    torch.testing.assert_close(
        signac_scores.std(dim=0), torch.ones(N_PC), atol=1e-5, rtol=0
    )
    assert gensim.components_ is not None
    matrix = torch.from_numpy(tfidf_of_fit(gensim, X)).float()
    torch.testing.assert_close(
        gensim_scores, matrix @ gensim.components_, atol=1e-4, rtol=1e-4
    )


@pytest.mark.parametrize("solver", SOLVERS)
def test_the_fit_is_deterministic_and_the_largest_loading_is_positive(
    solver: str,
) -> None:
    X = grouped_counts()
    first = LSA(N_PC, solver=solver, device="cpu").fit(X)
    second = LSA(N_PC, solver=solver, device="cpu").fit(X)

    assert first.components_ is not None
    assert second.components_ is not None
    torch.testing.assert_close(first.components_, second.components_)
    largest = first.components_.abs().argmax(dim=0)
    assert torch.all(first.components_[largest, torch.arange(N_PC)] > 0)


def test_the_solvers_find_the_same_components() -> None:
    X = grouped_counts()
    arpack = LSA(N_PC, solver="arpack", device="cpu").fit(X)
    randomized = LSA(N_PC, solver="randomized", device="cpu").fit(X)

    assert arpack.components_ is not None
    assert randomized.components_ is not None
    torch.testing.assert_close(
        randomized.components_[:, :DEFINED],
        arpack.components_[:, :DEFINED],
        atol=1e-2,
        rtol=0,
    )


def test_auto_takes_arpack_in_memory_and_randomized_when_backed(
    tmp_path: Path,
) -> None:
    X = grouped_counts()
    path = tmp_path / "counts.h5ad"
    ad.AnnData(X).write_h5ad(path)

    in_memory = LSA(N_PC, device="cpu").fit(X)
    backed = LSA(N_PC, device="cpu").fit(ad.read_h5ad(path, backed="r"))

    assert in_memory.solver_ == "arpack"
    assert backed.solver_ == "randomized"
    assert in_memory.components_ is not None
    assert backed.components_ is not None
    torch.testing.assert_close(
        backed.components_[:, :DEFINED],
        in_memory.components_[:, :DEFINED],
        atol=1e-2,
        rtol=0,
    )


def test_a_backed_randomized_fit_equals_one_in_memory(tmp_path: Path) -> None:
    X = grouped_counts()
    path = tmp_path / "counts.h5ad"
    ad.AnnData(X).write_h5ad(path)
    options = {"solver": "randomized", "device": "cpu", "chunk_size": 50}

    backed = LSA(N_PC, **options)
    backed_scores = backed.fit_transform(ad.read_h5ad(path, backed="r"))
    in_memory = LSA(N_PC, **options)
    in_memory_scores = in_memory.fit_transform(X)

    assert backed.components_ is not None
    assert in_memory.components_ is not None
    torch.testing.assert_close(
        backed.components_, in_memory.components_, atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(backed_scores, in_memory_scores, atol=1e-4, rtol=1e-4)


def test_arpack_cannot_take_a_backed_anndata(tmp_path: Path) -> None:
    path = tmp_path / "counts.h5ad"
    ad.AnnData(grouped_counts()).write_h5ad(path)

    with pytest.raises(ValueError, match="needs the matrix in memory"):
        LSA(N_PC, solver="arpack").fit(ad.read_h5ad(path, backed="r"))


def test_drop_depth_drops_the_component_that_follows_the_depth() -> None:
    X = grouped_counts(depth_spread=1.5)
    model = LSA(N_PC, flavor="gensim", drop_depth=True, device="cpu")

    scores = model.fit_transform(X)

    assert model.depth_correlations_ is not None
    assert model.n_dropped_components_ is not None
    dropped = np.abs(model.depth_correlations_) > 0.75
    assert model.n_dropped_components_ == int(dropped.sum()) >= 1
    assert scores.shape[1] == N_PC - model.n_dropped_components_
    depth = np.asarray(X.sum(axis=1)).ravel()
    for column in scores.T.numpy():
        assert abs(scipy.stats.spearmanr(column, depth).statistic) <= 0.75


def test_binarize_fits_the_binary_matrix() -> None:
    X = grouped_counts()
    binarized = LSA(N_PC, binarize=True, device="cpu").fit(X)
    binary = LSA(N_PC, device="cpu").fit((X > 0).astype(np.float32))

    assert binarized.components_ is not None
    assert binary.components_ is not None
    torch.testing.assert_close(binarized.components_, binary.components_)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"flavor": "tfidf"}, "flavor='tfidf' is not valid"),
        ({"solver": "lobpcg"}, "solver='lobpcg' is not valid"),
        ({"n_iter": -1}, "are not valid"),
    ],
)
def test_lsa_rejects_an_invalid_option(options: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        LSA(N_PC, **options)


def test_lsa_rejects_too_many_components_and_negative_counts() -> None:
    X = grouped_counts()

    with pytest.raises(ValueError, match="Use n_pc=59 or fewer"):
        LSA(60, device="cpu").fit(X)
    with pytest.raises(ValueError, match="negative values"):
        LSA(N_PC, device="cpu").fit(-X)


def test_transform_needs_a_fit() -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        LSA(N_PC).transform(grouped_counts())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_the_randomized_solver_on_cuda_equals_the_cpu() -> None:
    X = grouped_counts()
    cuda = LSA(N_PC, solver="randomized", device="cuda").fit(X)
    cpu = LSA(N_PC, solver="randomized", device="cpu").fit(X)

    assert cuda.components_ is not None
    assert cpu.components_ is not None
    torch.testing.assert_close(cuda.components_, cpu.components_, atol=1e-4, rtol=0)
