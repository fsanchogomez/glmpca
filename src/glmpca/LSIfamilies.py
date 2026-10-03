"""The latent semantic indexing families: a Gaussian on a weighting of the counts.

`WeightedGaussian` is the shared design, and `SignacLSI` and `GensimLSI` are the TF-IDF
weightings of Signac and of gensim. GLMPCA holds the intercept of these families at
zero, so a fit is the truncated SVD of the weighted matrix.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from .ExponentialFamily import Gaussian

if TYPE_CHECKING:
    from .sparse import SparseRows


TFIDF_SCALE = 1e4
"""Scale factor of Signac's `RunTFIDF(method = 1)`, applied before `log1p`."""

GENSIM_SLOPE = 0.25
"""Slope of gensim's pivoted unique normalisation, `smartirs="lfu"`."""


def _count_statistics(
    X: torch.Tensor | SparseRows, chunk: int, family_name: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """The total count and the number of holding cells of every feature, in float64.

    A pass by blocks of rows, so a sparse input is never made dense as a whole. It
    rejects a negative value: the weightings of the LSI families take counts.
    """
    totals = torch.zeros(X.shape[1], dtype=torch.float64, device=X.device)
    holders = torch.zeros(X.shape[1], dtype=torch.float64, device=X.device)
    for start in range(0, X.shape[0], chunk):
        block = X[start : start + chunk]
        if bool((block < 0).any()):
            msg = (
                f"The {family_name} family weighs counts, but the input has negative "
                f"values."
            )
            raise ValueError(msg)
        totals += block.sum(dim=0, dtype=torch.float64)
        holders += (block > 0).sum(dim=0, dtype=torch.float64)
    return totals, holders


class WeightedGaussian(Gaussian):
    r"""Gaussian with standard deviation one, on a weighting of the counts.

    The sufficient statistic of a cell is `weigh(x)`, and `eta` and `A` are those of
    the Gaussian, so the cost is `0.5 ||weigh(X) - Theta_hat||^2`. With the intercept at
    zero, which GLMPCA holds for these families, its minimiser is the truncated SVD of
    the weighted matrix: latent semantic indexing. The weights of the features are
    fitted once by initialize_family_parameters, and transform weighs a new cell with
    them. A zero stays a zero, so keep_sparse works.
    """

    def weigh(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def sufficient_statistics(self, X: torch.Tensor) -> torch.Tensor:
        return self.weigh(X)

    def log_base_measure(self, X: torch.Tensor) -> torch.Tensor:
        return -torch.square(self.weigh(X)) / 2.0 - np.log(2.0 * np.pi) / 2.0

    def base_measure(self, X: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.log_base_measure(X))

    def invert_g(self, X: torch.Tensor) -> torch.Tensor:
        return self.weigh(X)


class SignacLSI(WeightedGaussian):
    r"""Signac's `RunTFIDF(method = 1)` followed by `RunSVD`.

        weigh(x) = log1p(tf · idf · TFIDF_SCALE)
        tf       = count / depth of the cell
        idf      = number of cells / total counts of the feature

    The idf divides by the total counts of the feature, as Signac's `rowSums` does, not
    by the number of cells that hold it. A feature with no counts gets a total of 1.

    family_params of interest:
        - "idf" (torch.Tensor): the idf of every feature, set by
        initialize_family_parameters.
    """

    def __init__(
        self, family_params: dict[str, Any] | None = None, **kwargs: object
    ) -> None:
        super().__init__(family_params, **kwargs)
        self.family_name = "signac_lsi"

    def weigh(self, X: torch.Tensor) -> torch.Tensor:
        depth = X.sum(dim=1, keepdim=True).clip(min=1.0)
        return torch.log1p(X / depth * self.family_params["idf"] * TFIDF_SCALE)

    def initialize_family_parameters(self, X: torch.Tensor | SparseRows) -> None:
        totals, _ = _count_statistics(
            X, int(self.family_params.get("chunk_size", 8192)), self.family_name
        )
        self.family_params["idf"] = (X.shape[0] / totals.clip(min=1.0)).to(X.dtype)


class GensimLSI(WeightedGaussian):
    r"""gensim's `TfidfModel(normalize=True, smartirs="lfu")` followed by `LsiModel`.

    It is the call of sincei's `TOPICMODEL.runLSA`. For a count above zero,

        weigh(x) = (1 + log2(count)) · log2(cells / holders)
                   / ((1 - GENSIM_SLOPE) · pivot + GENSIM_SLOPE · kept)

    with `holders` the cells that hold the feature, `pivot` the mean number of features
    that a cell holds, and `kept` the features of this cell whose weight is above zero.
    A feature that every cell holds weighs zero and leaves the cell, as in gensim.

    family_params of interest:
        - "idf" (torch.Tensor) and "pivot" (float), set by
        initialize_family_parameters.
    """

    def __init__(
        self, family_params: dict[str, Any] | None = None, **kwargs: object
    ) -> None:
        super().__init__(family_params, **kwargs)
        self.family_name = "gensim_lsi"

    def weigh(self, X: torch.Tensor) -> torch.Tensor:
        held = X > 0
        weight = torch.where(
            held,
            (1.0 + torch.log2(torch.where(held, X, 1.0))) * self.family_params["idf"],
            0.0,
        )
        kept = (weight > 0).sum(dim=1, keepdim=True)
        norm = (1.0 - GENSIM_SLOPE) * self.family_params["pivot"] + GENSIM_SLOPE * kept
        return weight / norm

    def initialize_family_parameters(self, X: torch.Tensor | SparseRows) -> None:
        _, holders = _count_statistics(
            X, int(self.family_params.get("chunk_size", 8192)), self.family_name
        )
        idf = torch.where(
            holders > 0, torch.log2(X.shape[0] / holders.clip(min=1.0)), 0.0
        )
        self.family_params["idf"] = idf.to(X.dtype)
        self.family_params["pivot"] = float(holders.sum()) / X.shape[0]
