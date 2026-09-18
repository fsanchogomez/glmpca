from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import numpy as np
    from scipy.sparse import csr_matrix


class SparseRows:
    """A CSR matrix read by blocks of rows, each block handed out dense.

    `GLMPCA(keep_sparse=True)` holds its input this way, so that the dense matrix never
    exists as a whole: a block is densified when it is read, and dropped after. It
    answers `shape`, `device`, `dtype` and indexing by rows as a tensor does, so the
    code that reads a matrix by blocks takes either.
    """

    def __init__(self, matrix: csr_matrix) -> None:
        self.matrix = matrix

    @property
    def shape(self) -> torch.Size:
        return torch.Size(self.matrix.shape)

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        return torch.float32

    @property
    def nbytes(self) -> int:
        return int(
            self.matrix.data.nbytes
            + self.matrix.indices.nbytes
            + self.matrix.indptr.nbytes
        )

    def __getitem__(self, rows: slice | torch.Tensor | np.ndarray) -> torch.Tensor:
        if isinstance(rows, torch.Tensor):
            rows = rows.cpu().numpy()
        return torch.from_numpy(self.matrix[rows].toarray())
