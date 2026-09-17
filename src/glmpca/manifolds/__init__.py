"""The manifold primitives that GLM-PCA needs.

The loadings live on the Stiefel manifold and the intercept in Euclidean space.
`RiemannianAdagrad` optimises both through the same three operations: `egrad2rgrad`,
`proju` and `retr`.
"""

from .euclidean import Euclidean
from .optimizer import RiemannianAdagrad, RiemannianAdam
from .parameter import ManifoldParameter
from .stiefel import EuclideanStiefel

__all__ = [
    "Euclidean",
    "EuclideanStiefel",
    "ManifoldParameter",
    "RiemannianAdagrad",
    "RiemannianAdam",
]
