"""Check the depth of a cell and the rule that drops the components that follow it."""

from __future__ import annotations

from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pytest
import torch
from glmpca.utils import DEPTH_CORRELATION_CUTOFF, cell_depth, components_to_keep
from scipy import sparse

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

COUNTS = np.random.default_rng(0).poisson(2.0, size=(30, 8)).astype(np.float32)


@pytest.mark.parametrize(
    "as_input",
    [torch.from_numpy, np.asarray, sparse.csr_matrix, ad.AnnData],
    ids=["tensor", "ndarray", "csr", "anndata"],
)
def test_cell_depth_is_the_total_count_of_every_cell(
    as_input: Callable[[np.ndarray], object],
) -> None:
    np.testing.assert_allclose(cell_depth(as_input(COUNTS)), COUNTS.sum(axis=1))


@pytest.mark.parametrize("dense", [False, True])
def test_cell_depth_reads_a_backed_anndata_by_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dense: bool
) -> None:
    monkeypatch.setattr("glmpca.utils.DEPTH_CHUNK_ROWS", 7)
    path = tmp_path / "counts.h5ad"
    ad.AnnData(COUNTS if dense else sparse.csr_matrix(COUNTS)).write_h5ad(path)

    depth = cell_depth(ad.read_h5ad(path, backed="r"))

    np.testing.assert_allclose(depth, COUNTS.sum(axis=1))


def test_components_to_keep_drops_the_scores_that_follow_the_depth(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rng = np.random.default_rng(0)
    depth = rng.uniform(10, 1000, size=200)
    scores = np.column_stack([
        rng.normal(size=200),
        np.log(depth) + rng.normal(scale=0.1, size=200),
        rng.normal(size=200),
        -depth,
    ])

    kept, correlations = components_to_keep(scores, depth)

    np.testing.assert_array_equal(kept, [0, 2])
    assert np.all(np.abs(correlations[[1, 3]]) > DEPTH_CORRELATION_CUTOFF)
    assert "DEPTH: 2 of 4 components dropped" in capsys.readouterr().out
