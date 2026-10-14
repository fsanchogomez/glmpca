"""Check the scLSA command."""

from __future__ import annotations

from importlib.metadata import version
from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from glmpca.cli.sclsa import app
from scipy import sparse
from typer.testing import CliRunner

if TYPE_CHECKING:
    from pathlib import Path

    from typer.testing import Result

N_CELLS = 60
N_FEATURES = 30
N_PC = 3

runner = CliRunner()


def counts() -> sparse.csr_matrix:
    rng = np.random.default_rng(0)
    group = np.repeat(np.arange(3), N_CELLS // 3)
    rates = np.full((3, N_FEATURES), 0.1)
    for g in range(3):
        rates[g, g * 10 : (g + 1) * 10] = 1.0
    return sparse.csr_matrix(rng.poisson(rates[group]).astype(np.float32))


def write_input(tmp_path: Path) -> Path:
    path = tmp_path / "counts.h5ad"
    ad.AnnData(counts()).write_h5ad(path)
    return path


def run(input_path: Path, out_path: Path, *options: str) -> Result:
    return runner.invoke(
        app,
        ["-i", str(input_path), "-o", str(out_path), "-n", str(N_PC), *options],
    )


@pytest.mark.parametrize("flavor", ["signac", "gensim"])
def test_sclsa_writes_the_results_to_the_output_file(
    tmp_path: Path, flavor: str
) -> None:
    out_path = tmp_path / "out.h5ad"

    result = run(write_input(tmp_path), out_path, "--flavor", flavor)

    assert result.exit_code == 0, result.output
    adata = ad.read_h5ad(out_path)
    assert adata.obsm["X_lsa"].shape == (N_CELLS, N_PC)
    assert adata.varm["LSA_loadings"].shape == (N_FEATURES, N_PC)
    params = adata.uns["LSA"]["params"]
    assert params["flavor"] == flavor
    assert params["solver"] == "auto"
    assert params["solver_used"] == "arpack"
    assert len(adata.uns["LSA"]["singular_values"]) == N_PC
    np.testing.assert_array_equal(adata.X.toarray(), counts().toarray())


def test_sclsa_defaults_to_20_components(tmp_path: Path) -> None:
    out_path = tmp_path / "out.h5ad"

    result = runner.invoke(app, ["-i", str(write_input(tmp_path)), "-o", str(out_path)])

    assert result.exit_code == 0, result.output
    assert ad.read_h5ad(out_path).uns["LSA"]["params"]["n_pc"] == 20


def test_backed_takes_the_randomized_solver(tmp_path: Path) -> None:
    out_path = tmp_path / "out.h5ad"

    result = run(write_input(tmp_path), out_path, "--backed")

    assert result.exit_code == 0, result.output
    params = ad.read_h5ad(out_path).uns["LSA"]["params"]
    assert params["backed"]
    assert params["solver_used"] == "randomized"


def test_arpack_cannot_be_used_with_backed(tmp_path: Path) -> None:
    result = run(
        write_input(tmp_path), tmp_path / "out.h5ad", "--backed", "--solver", "arpack"
    )

    assert result.exit_code != 0
    assert "--solver arpack needs the matrix in memory" in result.output


def test_drop_depth_reports_the_dropped_components_in_uns(tmp_path: Path) -> None:
    out_path = tmp_path / "out.h5ad"

    result = run(write_input(tmp_path), out_path, "--dropDepth")

    assert result.exit_code == 0, result.output
    assert "DEPTH:" in result.output
    adata = ad.read_h5ad(out_path)
    report = adata.uns["LSA"]
    assert len(report["depth_correlations"]) == N_PC
    kept = N_PC - int(report["n_dropped_components"])
    assert adata.obsm["X_lsa"].shape == (N_CELLS, kept)


def test_no_drop_depth_by_default_leaves_the_report_out(tmp_path: Path) -> None:
    out_path = tmp_path / "out.h5ad"

    result = run(write_input(tmp_path), out_path)

    assert result.exit_code == 0, result.output
    assert "depth_correlations" not in ad.read_h5ad(out_path).uns["LSA"]


def test_binarize_is_recorded_and_keeps_the_counts(tmp_path: Path) -> None:
    out_path = tmp_path / "out.h5ad"

    result = run(write_input(tmp_path), out_path, "--binarize")

    assert result.exit_code == 0, result.output
    adata = ad.read_h5ad(out_path)
    assert adata.uns["LSA"]["params"]["binarize"]
    np.testing.assert_array_equal(adata.X.toarray(), counts().toarray())


def test_the_umap_outputs_are_written(tmp_path: Path) -> None:
    out_path = tmp_path / "out.h5ad"
    plot_path = tmp_path / "umap.png"

    result = run(write_input(tmp_path), out_path, "-op", str(plot_path), "-nk", "10")

    assert result.exit_code == 0, result.output
    adata = ad.read_h5ad(out_path)
    assert adata.obsm["X_umap"].shape == (N_CELLS, 2)
    assert "leiden" in adata.obs
    assert plot_path.exists()
    table = pd.read_csv(plot_path.with_suffix(".tsv"), sep="\t")
    assert list(table.columns) == ["Cell_ID", "UMAP1", "UMAP2", "cluster"]


def test_the_version_names_the_package() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert f"glmpca {version('glmpca')}" in result.output
