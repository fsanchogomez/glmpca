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
    normalize_processors,
    umap_leiden,
)
from glmpca.ExponentialFamily import Gaussian, GLMFamily
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
            "--chunkSize",
            "16",
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
    assert model.chunk_size == 16
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

    result = run(input_path, tmp_path / "out.h5ad", "-gf", "binomial")

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
    centres = np.eye(3, 10)[labels] * 50
    coordinates = (centres + rng.normal(size=(90, 10))).astype(np.float32)

    embedding, clusters = umap_leiden(coordinates, n_neighbors=10, resolution=1.0)

    assert embedding.shape == (90, 2)
    assert len(np.unique(clusters)) == 3
    for label in range(3):
        assert len(np.unique(clusters[labels == label])) == 1


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


def test_the_family_choices_are_the_library_families_plus_fast_poisson() -> None:
    assert {choice.value for choice in FamilyChoice} == {
        family.value for family in GLMFamily
    } | {"fast_poisson"}


def test_glmpca_runs_fast_poisson(tmp_path: Path) -> None:
    counts = (
        np.random
        .default_rng(0)
        .poisson(3.0, size=(N_CELLS, N_FEATURES))
        .astype(np.float32)
    )
    out_path = tmp_path / "out.h5ad"

    result = run(
        write_input(tmp_path, counts),
        out_path,
        "-gf",
        "fast_poisson",
        "--maxIter",
        "20",
    )

    assert result.exit_code == 0, result.output
    adata = ad.read_h5ad(out_path)
    assert adata.obsm["X_glmPCA"].shape == (N_CELLS, N_PC)
    assert adata.varm["glmPCA_loadings"].shape == (N_FEATURES, N_PC)
    assert adata.var["glmPCA_intercept"].shape == (N_FEATURES,)
    assert adata.uns["glmPCA"]["params"]["family"] == "fast_poisson"
    assert adata.uns["glmPCA"]["params"]["accelerate"]


def test_the_lsi_start_reaches_the_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models: list[GLMPCA] = []
    original_fit = GLMPCA.fit

    def recording_fit(self: GLMPCA, X: torch.Tensor) -> bool:
        models.append(self)
        return original_fit(self, X)

    monkeypatch.setattr(GLMPCA, "fit", recording_fit)

    result = run(
        write_input(tmp_path, poisson_counts()),
        tmp_path / "out.h5ad",
        "--init",
        "lsi",
    )

    assert result.exit_code == 0, result.output
    (model,) = models
    assert model.init == "lsi"


def test_the_depth_factor_is_written_to_obs(tmp_path: Path) -> None:
    out_path = tmp_path / "out.h5ad"

    result = run(write_input(tmp_path, poisson_counts()), out_path)

    assert result.exit_code == 0, result.output
    adata = ad.read_h5ad(out_path)
    assert adata.obs["glmPCA_depth"].shape == (N_CELLS,)
    assert np.all(np.isfinite(adata.obs["glmPCA_depth"].to_numpy()))


def test_fast_poisson_writes_its_size_factor_to_the_same_column(
    tmp_path: Path,
) -> None:
    out_path = tmp_path / "out.h5ad"

    result = run(
        write_input(tmp_path, poisson_counts()),
        out_path,
        "-gf",
        "fast_poisson",
    )

    assert result.exit_code == 0, result.output
    adata = ad.read_h5ad(out_path)
    assert adata.obs["glmPCA_depth"].shape == (N_CELLS,)


def test_no_depth_factor_leaves_the_column_out(tmp_path: Path) -> None:
    out_path = tmp_path / "out.h5ad"

    result = run(write_input(tmp_path, poisson_counts()), out_path, "--noDepthFactor")

    assert result.exit_code == 0, result.output
    assert "glmPCA_depth" not in ad.read_h5ad(out_path).obs


def test_no_depth_factor_turns_the_cell_offset_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models: list[GLMPCA] = []
    original_fit = GLMPCA.fit

    def recording_fit(self: GLMPCA, X: torch.Tensor) -> bool:
        models.append(self)
        return original_fit(self, X)

    monkeypatch.setattr(GLMPCA, "fit", recording_fit)
    out_path = tmp_path / "out.h5ad"

    result = run(write_input(tmp_path, poisson_counts()), out_path, "--noDepthFactor")

    assert result.exit_code == 0, result.output
    (model,) = models
    assert not model.depth_factor
    assert model.saturated_depth_ is None
    assert not ad.read_h5ad(out_path).uns["glmPCA"]["params"]["depth_factor"]


def test_the_tfidf_flag_reaches_the_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models: list[GLMPCA] = []
    original_fit = GLMPCA.fit

    def recording_fit(self: GLMPCA, X: torch.Tensor) -> bool:
        models.append(self)
        return original_fit(self, X)

    monkeypatch.setattr(GLMPCA, "fit", recording_fit)
    out_path = tmp_path / "out.h5ad"

    result = run(write_input(tmp_path, poisson_counts()), out_path, "--tfidf")

    assert result.exit_code == 0, result.output
    (model,) = models
    assert model.tfidf
    assert ad.read_h5ad(out_path).uns["glmPCA"]["params"]["tfidf"]


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


def test_the_penalty_reaches_fast_poisson(tmp_path: Path) -> None:
    out_path = tmp_path / "out.h5ad"

    result = run(
        write_input(tmp_path, poisson_counts()),
        out_path,
        "-gf",
        "fast_poisson",
        "--penalty",
        "3.5",
    )

    assert result.exit_code == 0, result.output
    assert ad.read_h5ad(out_path).uns["glmPCA"]["params"]["penalty"] == 3.5


def test_no_accelerate_turns_the_acceleration_off(tmp_path: Path) -> None:
    counts = (
        np.random
        .default_rng(0)
        .poisson(3.0, size=(N_CELLS, N_FEATURES))
        .astype(np.float32)
    )
    out_path = tmp_path / "out.h5ad"

    result = run(
        write_input(tmp_path, counts),
        out_path,
        "-gf",
        "fast_poisson",
        "--maxIter",
        "20",
        "--noDaarem",
    )

    assert result.exit_code == 0, result.output
    adata = ad.read_h5ad(out_path)
    assert adata.obsm["X_glmPCA"].shape == (N_CELLS, N_PC)
    assert not adata.uns["glmPCA"]["params"]["accelerate"]
