"""The manifold primitives that GLM-PCA needs.

The loadings live on the Grassmann manifold, since the cost sees only their span,
and the intercept and the offset of a cell in Euclidean space. The optimisers move them
through the same three operations: `egrad2rgrad`, `proju` and `retr`. `canonical_basis`
picks the representative of the fitted span that PCA would report.

The Stiefel manifold stays available: the Grassmann one is built on it, and it is the
right manifold for a cost that depends on the basis rather than on the span.
"""

from .euclidean import Euclidean
from .grassmann import Grassmann, canonical_basis
from .optimizer import (
    RiemannianAdagrad,
    RiemannianAdam,
    RiemannianConjugateGradient,
)
from .parameter import ManifoldParameter
from .stiefel import EuclideanStiefel

__all__ = [
    "Euclidean",
    "EuclideanStiefel",
    "Grassmann",
    "ManifoldParameter",
    "RiemannianAdagrad",
    "RiemannianAdam",
    "RiemannianConjugateGradient",
    "canonical_basis",
]
