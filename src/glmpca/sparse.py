from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from scipy.sparse import csr_matrix, issparse

if TYPE_CHECKING:
    import h5py
    from anndata.abc import CSRDataset


def densified(block: csr_matrix, device: torch.device) -> torch.Tensor:
    """The CSR `block` as a dense float32 tensor, built on `device` from non-zeros."""
    n_rows, n_cols = block.shape
    counts = torch.from_numpy(np.diff(block.indptr)).to(device).long()
    row_of = torch.repeat_interleave(torch.arange(n_rows, device=device), counts)
    flat = row_of * n_cols + torch.from_numpy(block.indices).to(device).long()
    values = torch.from_numpy(np.asarray(block.data, dtype=np.float32))
    dense = torch.zeros(n_rows * n_cols, device=device)
    dense.index_add_(0, flat, values.to(device))
    return dense.view(n_rows, n_cols)


def binary(block: csr_matrix | np.ndarray) -> csr_matrix | np.ndarray:
    """1 where `block` is not zero, else 0, as float32. A CSR block stays CSR.

    The CSR block keeps its indices and gets new values, so the caller's matrix is
    left as it was.
    """
    if issparse(block):
        return csr_matrix(
            ((block.data != 0).astype(np.float32), block.indices, block.indptr),
            shape=block.shape,
        )
    return (np.asarray(block) != 0).astype(np.float32)


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
        block = self._block(rows)
        return torch.from_numpy(
            np.asarray(block.toarray() if issparse(block) else block, dtype=np.float32)
        )

    def dense(
        self, rows: slice | torch.Tensor | np.ndarray, device: torch.device
    ) -> torch.Tensor:
        """The rows `rows`, dense on `device`.

        Off the CPU only the non-zeros are copied, and the block is densified on
        `device`, so a copy takes the size of the CSR block, not of the dense block.
        """
        if device.type == "cpu":
            return self[rows]
        block = self._block(rows)
        if not issparse(block):
            return torch.from_numpy(np.asarray(block, dtype=np.float32)).to(device)
        return densified(block, device)

    def _block(self, rows: slice | torch.Tensor | np.ndarray) -> csr_matrix:
        """The rows `rows` as they are stored, in the order asked."""
        if isinstance(rows, torch.Tensor):
            rows = rows.cpu().numpy()
        return self.matrix[rows]


class BackedRows(SparseRows):
    """The matrix of a backed AnnData, read from its file by blocks of rows.

    `GLMPCA` reads an AnnData opened with `backed="r"` this way, so the matrix stays in
    the file and only the rows of the block being read are in memory. The matrix is a
    CSR dataset or a dense HDF5 dataset. Rows asked in any order are read in increasing
    order, as the file needs, and handed back in the order they were asked. With
    `binarize`, every value that is not zero is read as 1.
    """

    def __init__(
        self, matrix: CSRDataset | h5py.Dataset, *, binarize: bool = False
    ) -> None:
        self.matrix = matrix
        self.binarize = binarize

    @property
    def nbytes(self) -> int:
        return 0

    def _block(
        self, rows: slice | torch.Tensor | np.ndarray
    ) -> csr_matrix | np.ndarray:
        if isinstance(rows, slice):
            block = self.matrix[rows]
        else:
            if isinstance(rows, torch.Tensor):
                rows = rows.cpu().numpy()
            rows = np.asarray(rows)
            order = np.argsort(rows, kind="stable")
            block = self.matrix[rows[order]][np.argsort(order)]
        return binary(block) if self.binarize else block
