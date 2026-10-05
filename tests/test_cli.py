from __future__ import annotations

from importlib.metadata import version
from typing import TYPE_CHECKING, Any

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch
import typer
from glmpca.cli.glmpca import (
    AVAILABLE_PROCESSORS,
    FamilyChoice,
    app,
    log_normalized,
    nearest_neighbors,
    normalize_processors,
    umap_leiden,
)
from glmpca.ExponentialFamily import Gaussian, GLMFamily, NegativeBinomial
from glmpca.GLMPCA import GLMPCA
from scipy import sparse
from typer.testing import CliRunner

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from typer.testing import Result

N_CELLS = 40
N_FEATURES = 12
N_PC = 2

runner = CliRunner()


def poisson_counts() -> np.ndarray:
    return (
        np.random
        .default_rng(0)
        .poisson(3.0, size=(N_CELLS, N_FEATURES))
        .astype(np.float32)
    )


def write_input(
    tmp_path: Path, X: np.ndarray | sparse.csr_matrix | sparse.csc_matrix
) -> Path:
    path = tmp_path / "counts.h5ad"
    ad.AnnData(X).write_h5ad(path)
    return path


def run(input_path: Path, out_path: Path, *options: str) -> Result:
    torch.manual_seed(0)
    np.random.seed(0)
    return runner.invoke(
        app,
        [
            "-i",
            str(input_path),
            "-o",
            str(out_path),
            "-n",
            str(N_PC),
            "--maxIter",
            "2",
            "--batchSize",
            "8",
            *options,
        ],
    )


@pytest.mark.parametrize(
    "matrix_type",
    [np.asarray, sparse.csr_matrix, sparse.csc_matrix],
    ids=lambda matrix_type: matrix_type.__name__,
)
def test_glmpca_writes_the_results_to_the_output_file(
    matrix_type: Callable[[np.ndarray], Any], tmp_path: Path
) -> None:
    input_path = write_input(tmp_path, matrix_type(poisson_counts()))
    out_path = tmp_path / "out.h5ad"

    result = run(input_path, out_path)

    assert result.exit_code == 0, result.output
    adata = ad.read_h5ad(out_path)
    assert adata.obsm["X_glmPCA"].shape == (N_CELLS, N_PC)
    assert adata.varm["glmPCA_loadings"].shape == (N_FEATURES, N_PC)
    assert adata.var["glmPCA_intercept"].shape == (N_FEATURES,)
    params = adata.uns["glmPCA"]["params"]
    assert params["n_pc"] == N_PC
    assert params["family"] == "poisson"
    assert params["max_iter"] == 2
    assert params["batch_size"] == 8
    assert params["init"] == "spectral"
    assert not params["compile"]
    assert "X_umap" not in adata.obsm
    assert "leiden" not in adata.obs
    assert not list(tmp_path.glob("*.tsv"))


def test_glmpca_results_equal_the_library_results(tmp_path: Path) -> None:
    counts = poisson_counts()
    out_path = tmp_path / "out.h5ad"

    result = run(write_input(tmp_path, counts), out_path)

    assert result.exit_code == 0, result.output
    torch.manual_seed(0)
    np.random.seed(0)
    X = torch.Tensor(counts)
    model = GLMPCA(N_PC, family="poisson", max_iter=2, batch_size=8)
    model.fit(X)
    assert model.saturated_loadings_ is not None
    assert model.saturated_intercept_ is not None
    with torch.no_grad():
        scores = model.transform(X)
    adata = ad.read_h5ad(out_path)
    torch.testing.assert_close(torch.from_numpy(adata.obsm["X_glmPCA"]), scores)
    torch.testing.assert_close(
        torch.from_numpy(adata.varm["glmPCA_loadings"]),
        model.saturated_loadings_.detach(),
    )
    torch.testing.assert_close(
        torch.from_numpy(adata.var["glmPCA_intercept"].to_numpy()),
        model.saturated_intercept_.detach(),
    )


def test_glmpca_passes_every_option_to_the_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models: list[GLMPCA] = []
    original_fit = GLMPCA.fit

    def recording_fit(self: GLMPCA, X: torch.Tensor) -> bool:
        models.append(self)
        return original_fit(self, X)

    monkeypatch.setattr(GLMPCA, "fit", recording_fit)
    input_path = write_input(tmp_path, poisson_counts())

    result = runner.invoke(
        app,
        [
            "-i",
            str(input_path),
            "-o",
            str(tmp_path / "out.h5ad"),
            "-n",
            "3",
            "-gf",
            "gaussian",
            "--maxIter",
            "3",
            "--learningRate",
            "0.1",
            "--batchSize",
            "16",
            "--gamma",
            "0.25",
            "--nInit",
            "2",
            "--init",
            "random",
            "--device",
            "cpu",
            "--optimizer",
            "adam",
            "-p",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    (model,) = models
    assert model.n_pc == 3
    assert isinstance(model.exponential_family, Gaussian)
    assert model.max_iter == 3
    assert model.initial_learning_rate_ == 0.1
    assert model.batch_size == 16
    assert model.gamma == 0.25
    assert model.n_init == 2
    assert model.init == "random"
    assert model.device == "cpu"
    assert model.chunk_size is None
    assert model.optimizer == "adam"
    assert model.exponential_family.family_params["n_jobs"] == 1


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (["-gf", "gamma"], "defined only for values greater than 0"),
        (["--device", "mps"], "The mps device is not supported"),
    ],
    ids=["gamma_with_zeros", "mps"],
)
def test_glmpca_reports_errors_without_a_traceback(
    options: list[str], message: str, tmp_path: Path
) -> None:
    counts = poisson_counts()
    counts[0, 0] = 0
    out_path = tmp_path / "out.h5ad"

    result = run(write_input(tmp_path, counts), out_path, *options)

    assert result.exit_code == 1
    assert message in result.output
    assert not out_path.exists()


def test_glmpca_rejects_an_unknown_family(tmp_path: Path) -> None:
    input_path = write_input(tmp_path, poisson_counts())

    result = run(input_path, tmp_path / "out.h5ad", "-gf", "dirichlet")

    assert result.exit_code == 2


@pytest.mark.parametrize(
    ("file_format", "signature"),
    [("png", b"\x89PNG"), ("pdf", b"%PDF"), ("svg", b"<?xml")],
)
def test_glmpca_plots_the_umap_with_leiden_clusters(
    file_format: str, signature: bytes, tmp_path: Path
) -> None:
    out_path = tmp_path / "out.h5ad"
    plot_path = tmp_path / f"umap.{file_format}"

    result = run(
        write_input(tmp_path, poisson_counts()),
        out_path,
        "-op",
        str(plot_path),
        "-nk",
        "5",
        "--plotFileFormat",
        file_format,
        "--dpi",
        "50",
    )

    assert result.exit_code == 0, result.output
    assert plot_path.read_bytes().startswith(signature)
    adata = ad.read_h5ad(out_path)
    assert adata.obsm["X_umap"].shape == (N_CELLS, 2)
    table = pd.read_csv(
        plot_path.with_suffix(".tsv"),
        sep="\t",
        dtype={"Cell_ID": str, "cluster": str},
    )
    assert list(table.columns) == ["Cell_ID", "UMAP1", "UMAP2", "cluster"]
    assert table["Cell_ID"].tolist() == adata.obs_names.tolist()
    np.testing.assert_allclose(
        table[["UMAP1", "UMAP2"]].to_numpy(),
        np.asarray(adata.obsm["X_umap"], dtype=np.float64),
        rtol=1e-6,
    )
    assert table["cluster"].tolist() == adata.obs["leiden"].astype(str).tolist()


def test_umap_leiden_finds_separated_groups() -> None:
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(3), 30)
    centers = np.eye(3, 10)[labels] * 50
    coordinates = (centers + rng.normal(size=(90, 10))).astype(np.float32)

    embedding, clusters = umap_leiden(coordinates, n_neighbors=10, resolution=1.0)

    assert embedding.shape == (90, 2)
    assert len(np.unique(clusters)) == 3
    for label in range(3):
        assert len(np.unique(clusters[labels == label])) == 1


def test_umap_leiden_gives_the_same_result_at_every_run() -> None:
    rng = np.random.default_rng(0)
    coordinates = np.vstack([
        rng.normal(center, 0.1, size=(30, 2)) for center in (0.0, 5.0, 10.0)
    ])

    first = umap_leiden(coordinates, n_neighbors=10, resolution=1.0)
    second = umap_leiden(coordinates, n_neighbors=10, resolution=1.0)

    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_nearest_neighbors_are_the_brute_force_neighbors() -> None:
    coordinates = np.random.default_rng(0).normal(size=(200, 5))

    indices, distances = nearest_neighbors(coordinates, n_neighbors=15)

    pairwise = np.linalg.norm(coordinates[:, None] - coordinates[None], axis=2)
    np.testing.assert_array_equal(indices, np.argsort(pairwise, axis=1)[:, :15])
    np.testing.assert_allclose(distances, np.sort(pairwise, axis=1)[:, :15], atol=1e-6)
    np.testing.assert_array_equal(distances[:, 0], 0)


def test_nearest_neighbors_search_approximately_above_the_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("glmpca.cli.glmpca.EXACT_NEIGHBORS_LIMIT", 100)
    coordinates = np.random.default_rng(0).normal(size=(300, 5))

    indices, distances = nearest_neighbors(coordinates, n_neighbors=15)

    assert indices.shape == distances.shape == (300, 15)
    np.testing.assert_array_equal(indices[:, 0], np.arange(300))
    assert np.all(np.diff(distances, axis=1) >= 0)


def test_umap_leiden_equals_scanpy() -> None:
    sc = pytest.importorskip("scanpy")
    rng = np.random.default_rng(0)
    centers = rng.normal(0, 4, size=(4, 10))
    coordinates = (
        centers[rng.integers(0, 4, 500)] + rng.normal(size=(500, 10))
    ).astype(np.float32)
    adata = ad.AnnData(np.zeros((500, 1), dtype=np.float32))
    adata.obsm["X_glmPCA"] = coordinates

    sc.pp.neighbors(adata, use_rep="X_glmPCA", n_neighbors=15)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, flavor="igraph", directed=False, n_iterations=-1)
    embedding, clusters = umap_leiden(coordinates, n_neighbors=15, resolution=1.0)

    np.testing.assert_array_equal(embedding, adata.obsm["X_umap"])
    np.testing.assert_array_equal(clusters.astype(str), adata.obs["leiden"].to_numpy())


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("max", AVAILABLE_PROCESSORS),
        ("max/2", max(AVAILABLE_PROCESSORS // 2, 1)),
        ("1", 1),
        (str(AVAILABLE_PROCESSORS + 1), AVAILABLE_PROCESSORS),
    ],
)
def test_number_of_processors_is_parsed(value: str, expected: int) -> None:
    assert normalize_processors(value) == expected


@pytest.mark.parametrize("value", ["0", "-1", "all"])
def test_number_of_processors_rejects_invalid_values(value: str) -> None:
    with pytest.raises(typer.BadParameter):
        normalize_processors(value)


def test_glmpca_prints_the_version() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.output.strip() == f"glmpca {version('glmpca')}"


def test_glmpca_prints_the_help() -> None:
    result = runner.invoke(app, ["-h"])

    assert result.exit_code == 0
    assert "--nPrinComps" in result.output
    assert "--learningRate" in result.output
    assert "--outFileUMAP" in result.output


def test_the_family_choices_are_the_library_families() -> None:
    assert {choice.value for choice in FamilyChoice} == {
        family.value for family in GLMFamily
    }


def test_the_depth_factor_is_written_to_obs(tmp_path: Path) -> None:
    out_path = tmp_path / "out.h5ad"

    result = run(write_input(tmp_path, poisson_counts()), out_path, "--depthFactor")

    assert result.exit_code == 0, result.output
    adata = ad.read_h5ad(out_path)
    assert adata.obs["glmPCA_depth"].shape == (N_CELLS,)
    assert np.all(np.isfinite(adata.obs["glmPCA_depth"].to_numpy()))


def test_no_depth_factor_by_default_leaves_the_column_out(tmp_path: Path) -> None:
    out_path = tmp_path / "out.h5ad"

    result = run(write_input(tmp_path, poisson_counts()), out_path)

    assert result.exit_code == 0, result.output
    assert "glmPCA_depth" not in ad.read_h5ad(out_path).obs


def test_the_depth_factor_flag_turns_the_cell_offset_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models: list[GLMPCA] = []
    original_fit = GLMPCA.fit

    def recording_fit(self: GLMPCA, X: torch.Tensor) -> bool:
        models.append(self)
        return original_fit(self, X)

    monkeypatch.setattr(GLMPCA, "fit", recording_fit)
    out_path = tmp_path / "out.h5ad"

    result = run(write_input(tmp_path, poisson_counts()), out_path, "--depthFactor")

    assert result.exit_code == 0, result.output
    (model,) = models
    assert model.depth_factor
    assert model.saturated_depth_ is not None
    assert ad.read_h5ad(out_path).uns["glmPCA"]["params"]["depth_factor"]


def test_the_keep_sparse_flag_reaches_the_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models: list[GLMPCA] = []
    original_fit = GLMPCA.fit

    def recording_fit(self: GLMPCA, X: torch.Tensor) -> bool:
        models.append(self)
        return original_fit(self, X)

    monkeypatch.setattr(GLMPCA, "fit", recording_fit)
    out_path = tmp_path / "out.h5ad"

    result = run(write_input(tmp_path, poisson_counts()), out_path, "--keepSparse")

    assert result.exit_code == 0, result.output
    (model,) = models
    assert model.keep_sparse
    assert ad.read_h5ad(out_path).uns["glmPCA"]["params"]["keep_sparse"]


@pytest.mark.parametrize(
    "matrix_type",
    [np.asarray, sparse.csr_matrix, sparse.csc_matrix],
    ids=lambda matrix_type: matrix_type.__name__,
)
def test_log_normalized_scales_every_cell_to_10k_then_takes_log1p(
    matrix_type: Callable[[np.ndarray], Any],
) -> None:
    counts = poisson_counts()
    counts[0] = 0
    expected = np.zeros_like(counts)
    totals = counts[1:].sum(axis=1, keepdims=True)
    expected[1:] = np.log1p(counts[1:] / totals * 10_000)

    result = log_normalized(matrix_type(counts))

    assert sparse.issparse(result) == sparse.issparse(matrix_type(counts))
    dense = result.toarray() if sparse.issparse(result) else result
    assert dense.dtype == np.float32
    np.testing.assert_allclose(dense, expected, rtol=1e-6)


def test_the_log_normalize_flag_fits_the_transformed_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fitted: list[np.ndarray] = []
    original_fit = GLMPCA.fit

    def recording_fit(self: GLMPCA, X: ad.AnnData) -> bool:
        fitted.append(np.asarray(X.X))
        return original_fit(self, X)

    monkeypatch.setattr(GLMPCA, "fit", recording_fit)
    counts = poisson_counts()
    out_path = tmp_path / "out.h5ad"

    result = run(
        write_input(tmp_path, counts), out_path, "-gf", "gaussian", "--logNormalize"
    )

    assert result.exit_code == 0, result.output
    (X,) = fitted
    np.testing.assert_allclose(X, log_normalized(counts))
    adata = ad.read_h5ad(out_path)
    np.testing.assert_array_equal(adata.X, counts)
    assert adata.uns["glmPCA"]["params"]["log_normalize"]


def test_no_log_normalize_by_default(tmp_path: Path) -> None:
    out_path = tmp_path / "out.h5ad"

    result = run(write_input(tmp_path, poisson_counts()), out_path)

    assert result.exit_code == 0, result.output
    assert not ad.read_h5ad(out_path).uns["glmPCA"]["params"]["log_normalize"]


def test_the_compile_flag_reaches_the_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models: list[GLMPCA] = []
    original_fit = GLMPCA.fit

    def recording_fit(self: GLMPCA, X: torch.Tensor) -> bool:
        models.append(self)
        return original_fit(self, X)

    monkeypatch.setattr(GLMPCA, "fit", recording_fit)
    monkeypatch.setattr(torch, "compile", lambda function, **_: function)
    out_path = tmp_path / "out.h5ad"

    result = run(write_input(tmp_path, poisson_counts()), out_path, "--compile")

    assert result.exit_code == 0, result.output
    (model,) = models
    assert model.compile
    assert ad.read_h5ad(out_path).uns["glmPCA"]["params"]["compile"]


def test_the_negative_binomial_writes_its_mle_dispersion_to_var(
    tmp_path: Path,
) -> None:
    counts = poisson_counts()
    out_path = tmp_path / "out.h5ad"

    result = run(write_input(tmp_path, counts), out_path, "-gf", "negative_binomial")

    assert result.exit_code == 0, result.output
    family = NegativeBinomial()
    family.initialize_family_parameters(torch.from_numpy(counts))
    np.testing.assert_allclose(
        ad.read_h5ad(out_path).var["MLE_dispersion"], family.family_params["nu"]
    )


def test_other_families_write_no_dispersion(tmp_path: Path) -> None:
    out_path = tmp_path / "out.h5ad"

    result = run(write_input(tmp_path, poisson_counts()), out_path)

    assert result.exit_code == 0, result.output
    assert "MLE_dispersion" not in ad.read_h5ad(out_path).var


def test_an_lsi_depth_factor_reports_the_dropped_components_in_uns(
    tmp_path: Path,
) -> None:
    out_path = tmp_path / "out.h5ad"

    result = run(
        write_input(tmp_path, poisson_counts()),
        out_path,
        "-gf",
        "signac_lsi",
        "--depthFactor",
    )

    assert result.exit_code == 0, result.output
    assert "DEPTH:" in result.output
    report = ad.read_h5ad(out_path).uns["glmPCA"]
    assert len(report["depth_correlations"]) == N_PC
    assert 0 <= report["n_dropped_components"] <= N_PC


@pytest.mark.parametrize("family", ["signac_lsi", "gensim_lsi"])
def test_log_normalize_is_ignored_for_an_lsi_family(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, family: str
) -> None:
    models: list[GLMPCA] = []
    original_fit = GLMPCA.fit

    def recording_fit(self: GLMPCA, X: ad.AnnData) -> bool:
        models.append(self)
        return original_fit(self, X)

    counts = poisson_counts()
    out_path = tmp_path / "out.h5ad"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(GLMPCA, "fit", recording_fit)
        result = run(
            write_input(tmp_path, counts), out_path, "-gf", family, "--logNormalize"
        )

    assert result.exit_code == 0, result.output
    assert "--logNormalize is ignored" in caplog.text
    assert not ad.read_h5ad(out_path).uns["glmPCA"]["params"]["log_normalize"]
    (model,) = models
    np.testing.assert_allclose(
        model.exponential_family.family_params["idf"].numpy(),
        _lsi_idf(family, counts),
        rtol=1e-6,
    )


def test_the_binomial_stores_its_trials_and_clipped_entries_in_uns(
    tmp_path: Path,
) -> None:
    counts = poisson_counts()
    out_path = tmp_path / "out.h5ad"

    result = run(
        write_input(tmp_path, counts), out_path, "-gf", "binomial", "--nTrials", "3"
    )

    assert result.exit_code == 0, result.output
    report = ad.read_h5ad(out_path).uns["glmPCA"]
    assert report["n_trials"] == 3
    assert report["n_clipped"] == int((counts > 3).sum())


def _lsi_idf(family: str, counts: np.ndarray) -> np.ndarray:
    if family == "signac_lsi":
        return counts.shape[0] / np.maximum(counts.sum(axis=0), 1.0)
    holders = (counts > 0).sum(axis=0)
    return np.where(holders > 0, np.log2(counts.shape[0] / np.maximum(holders, 1)), 0)
