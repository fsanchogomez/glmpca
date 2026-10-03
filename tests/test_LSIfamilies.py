"""Check the LSI families against the TF-IDF weightings they encode."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from glmpca.LSIfamilies import TFIDF_SCALE, GensimLSI, SignacLSI
from glmpca.sparse import SparseRows
from scipy import sparse

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


def test_gensim_lsi_weighs_as_gensim_does() -> None:
    family = GensimLSI()
    X = torch.from_numpy(LSI_COUNTS)
    family.initialize_family_parameters(X)

    np.testing.assert_allclose(family.weigh(X).numpy(), GENSIM_WEIGHTS, atol=1e-6)


def test_signac_lsi_weighs_as_signac_runtfidf_does() -> None:
    family = SignacLSI()
    X = torch.from_numpy(LSI_COUNTS)
    family.initialize_family_parameters(X)

    depth = LSI_COUNTS.sum(axis=1, keepdims=True)
    idf = LSI_COUNTS.shape[0] / np.maximum(LSI_COUNTS.sum(axis=0), 1.0)
    expected = np.log1p(LSI_COUNTS / depth * idf * TFIDF_SCALE)
    np.testing.assert_allclose(family.weigh(X).numpy(), expected, rtol=1e-6)


@pytest.mark.parametrize("family_type", [SignacLSI, GensimLSI])
def test_the_lsi_families_reject_negative_values(
    family_type: type[SignacLSI | GensimLSI],
) -> None:
    X = torch.from_numpy(LSI_COUNTS.copy())
    X[0, 0] = -1.0

    with pytest.raises(ValueError, match="negative"):
        family_type().initialize_family_parameters(X)


@pytest.mark.parametrize("family_type", [SignacLSI, GensimLSI])
def test_the_lsi_weights_do_not_depend_on_sparse_storage_or_chunks(
    family_type: type[SignacLSI | GensimLSI],
) -> None:
    dense = family_type({"chunk_size": 10**9})
    dense.initialize_family_parameters(torch.from_numpy(LSI_COUNTS))
    chunked = family_type({"chunk_size": 2})
    chunked.initialize_family_parameters(SparseRows(sparse.csr_matrix(LSI_COUNTS)))

    for key in ("idf", "pivot"):
        if key in dense.family_params:
            torch.testing.assert_close(
                torch.as_tensor(chunked.family_params[key]),
                torch.as_tensor(dense.family_params[key]),
            )
