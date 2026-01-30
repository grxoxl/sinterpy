"""Linear operators (dense/sparse), including Toeplitz-based convolution operators."""

import numpy as np
import numpy.typing as npt
from typing import Tuple, Optional
from scipy.sparse import csc_array, issparse
from scipy.linalg import toeplitz

from .constants import SPARSE, DTYPE_BASE


__all__ = ["ConvolutionOperator", "AcousticStationaryOperator", "OperatorBase"]

dtype_base = DTYPE_BASE
sparse = SPARSE



class OperatorBase:
    def __init__(self, data: npt.NDArray[dtype_base], dtype: Optional[np.dtype] = dtype_base, sparse=sparse):
        data = csc_array(data) if sparse else np.asarray(data, dtype=dtype)
        if data.ndim != 2:
            raise ValueError(f"OperatorBase expects a 2D array, got ndim={data.ndim}, shape={data.shape}.")
        m, n = data.shape
        if m <= 0 or n <= 0:
            raise ValueError(f"Shape must be positive, got {data.shape}.")
        self.data = data
        self.shape = data.shape
        self.dtype = self.data.dtype if hasattr(self.data, "dtype") else dtype
        self.sparse = issparse(self.data) or isinstance(self.data, csc_array)
    
    @property
    def m(self) -> np.int32:
        return self.shape[0]

    @property
    def n(self) -> np.int32:
        return self.shape[1]
    
    def matvec(self, x: npt.NDArray[dtype_base]) -> npt.NDArray[dtype_base]:
        x = np.asarray(x, dtype=self.dtype)
        if x.ndim != 1 or x.shape[0] != self.n:
            raise ValueError(f"matvec: expected shape ({self.n},), got {x.shape}.")
        y = self.data @ x
        return np.asarray(y, dtype=self.dtype).reshape(-1)
    
    def rmatvec(self, x: npt.NDArray[dtype_base]) -> npt.NDArray[dtype_base]:
        x = np.asarray(x, dtype=self.dtype)
        if x.ndim != 1 or x.shape[0] != self.m:
            raise ValueError(f"rmatvec: expected ({self.m},), got {x.shape}.")
        y = self.data.T @ x
        return np.asarray(y, dtype=self.dtype).reshape(-1)

    def matmat(self, X: npt.NDArray[dtype_base]) -> npt.NDArray[dtype_base]:
        X = np.asarray(X, dtype=self.dtype)
        if X.ndim != 2 or X.shape[0] != self.n:
            raise ValueError(f"matmat: expected shape ({self.n}, k), got {X.shape}.")
        Y = self.data @ X
        return np.asarray(Y, dtype=self.dtype)
    
    def transpose(self) -> "OperatorBase":
        return OperatorBase(self.data.T, dtype=self.dtype, sparse=self.sparse)

    @property
    def T(self) -> "OperatorBase":
        return self.transpose()
    
    def to_numpy(self) -> npt.NDArray[dtype_base]:
        return self.data.toarray() if self.sparse else self.data
    
    def __matmul__(self, other):
        if isinstance(other, OperatorBase):
            return OperatorBase(self.data @ other.data, dtype=self.dtype, sparse=self.sparse or other.sparse)

        other = np.asarray(other, dtype=self.dtype)
        if other.ndim == 1:
            return self.matvec(other)
        if other.ndim == 2:
            return self.matmat(other)
        raise TypeError(f"Unsupported operand type for @: {type(other)}")

    def __rmatmul__(self, other):
        other = np.asarray(other, dtype=self.dtype)
        if other.ndim == 1:
            return self.rmatvec(other)
        if other.ndim == 2:
            return other @ self.data
        raise TypeError(f"Unsupported operand type for @: {type(other)}")

    def __add__(self, other):
        if isinstance(other, OperatorBase):
            if self.shape != other.shape:
                raise ValueError(f"Operator + Operator: shape mismatch {self.shape} vs {other.shape}.")
            return OperatorBase(self.data + other.data, dtype=self.dtype, sparse=self.sparse or other.sparse)
        other = np.asarray(other, dtype=self.dtype)
        if other.ndim == 2 and other.shape == self.shape:
            return OperatorBase(self.data + other, dtype=self.dtype, sparse=self.sparse)
        raise TypeError(f"Unsupported operand type for +: {type(other)}")

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, OperatorBase):
            if self.shape != other.shape:
                raise ValueError(f"Operator - Operator: shape mismatch {self.shape} vs {other.shape}.")
            return OperatorBase(self.data - other.data, dtype=self.dtype, sparse=self.sparse or other.sparse)
        other = np.asarray(other, dtype=self.dtype)
        if other.ndim == 2 and other.shape == self.shape:
            return OperatorBase(self.data - other, dtype=self.dtype, sparse=self.sparse)
        raise TypeError(f"Unsupported operand type for -: {type(other)}")

    def __rsub__(self, other):
        other = np.asarray(other, dtype=self.dtype)
        if other.ndim == 2 and other.shape == self.shape:
            return OperatorBase(other - self.data, dtype=self.dtype, sparse=self.sparse)
        raise TypeError(f"Unsupported operand type for -: {type(other)}")

    def __mul__(self, other):
        # scalar scaling only (not matrix multiply; that's @)
        if np.isscalar(other):
            return OperatorBase(self.data * other, dtype=self.dtype, sparse=self.sparse)
        if isinstance(other, OperatorBase):
            if self.shape != other.shape:
                raise ValueError(f"Operator * Operator (elemwise): shape mismatch {self.shape} vs {other.shape}.")
            return OperatorBase(self.data * other.data, dtype=self.dtype, sparse=self.sparse or other.sparse)
        other = np.asarray(other, dtype=self.dtype)
        if other.ndim == 2 and other.shape == self.shape:
            return OperatorBase(self.data * other, dtype=self.dtype, sparse=self.sparse)
        raise TypeError(f"Unsupported operand type for *: {type(other)}")

    def __rmul__(self, other):
        if np.isscalar(other):
            return OperatorBase(other * self.data, dtype=self.dtype, sparse=self.sparse)
        return self.__mul__(other)

    def __truediv__(self, other):
        if np.isscalar(other):
            if other == 0:
                raise ZeroDivisionError("division by zero")
            return OperatorBase(self.data / other, dtype=self.dtype, sparse=self.sparse)
        if isinstance(other, OperatorBase):
            if self.shape != other.shape:
                raise ValueError(f"Operator / Operator (elemwise): shape mismatch {self.shape} vs {other.shape}.")
            return OperatorBase(self.data / other.data, dtype=self.dtype, sparse=self.sparse or other.sparse)
        other = np.asarray(other, dtype=self.dtype)
        if other.ndim == 2 and other.shape == self.shape:
            return OperatorBase(self.data / other, dtype=self.dtype, sparse=self.sparse)
        raise TypeError(f"Unsupported operand type for /: {type(other)}")

    def __rtruediv__(self, other):
        if np.isscalar(other):
            return OperatorBase(other / self.data, dtype=self.dtype, sparse=self.sparse)
        other = np.asarray(other, dtype=self.dtype)
        if other.ndim == 2 and other.shape == self.shape:
            return OperatorBase(other / self.data, dtype=self.dtype, sparse=self.sparse)
        raise TypeError(f"Unsupported operand type for /: {type(other)}")

    def __neg__(self):
        return OperatorBase(-self.data, dtype=self.dtype, sparse=self.sparse)

    def __pos__(self):
        return self

    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"data={self.data}, "
            f"shape={self.shape}, "
            f"dtype={self.dtype}, "
            f"sparse={self.sparse}"
            f")"
        )

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}\n"
            f"data={self.data}, "
            f"  shape : {self.shape}\n"
            f"  dtype : {self.dtype}\n"
            f"  sparse: {self.sparse}"
        )
    
class ConvolutionOperator(OperatorBase):
    def __init__(self,
                 kernel: npt.NDArray[dtype_base] = (-1, 1),
                 offset: np.int32 = None,
                 shape: Tuple[np.int32, np.int32] = (1, 1),
                 dtype: Optional[np.dtype] = dtype_base,
                 sparse=sparse
                 ):
        kernel = np.asarray(kernel, dtype=dtype)
        if kernel.ndim != 1:
            raise ValueError(f"ConvolutionOperator expects a 1D kernel, got ndim={kernel.ndim}, shape={kernel.shape}.")
        if shape[0] != shape[1]:
            raise ValueError(f"Shape must be equal for both dimensions, got {shape}.")

        self.kernel = kernel
        self.offset = (len(self.kernel) // 2) if offset is None else int(offset)
        self.shape = (int(shape[0]), int(shape[1]))

        data = self._from_kernel(self.shape[0], self.offset, self.kernel, dtype, sparse)
        super().__init__(data, dtype=dtype, sparse=sparse)

    @staticmethod
    def _from_kernel(N: int, offset: int, kernel: npt.NDArray[dtype_base], dtype: np.dtype, sparse: bool):
        if offset < 0 or offset >= N:
            raise ValueError(f"offset must satisfy 0 <= offset < N, got offset={offset}, N={N}.")

        first_col = np.zeros(N, dtype=dtype)
        first_row = np.zeros(N, dtype=dtype)

        for k, val in enumerate(kernel):
            d = k - offset
            if d >= 0:
                if d < N:
                    first_col[d] = val
            else:
                idx = -d
                if idx < N:
                    first_row[idx] = val

        first_row[0] = first_col[0]

        A = toeplitz(first_col, first_row)
        return csc_array(A) if sparse else A

class AcousticStationaryOperator(OperatorBase):
    def __init__(self,
                 wavelet: npt.NDArray[dtype_base],
                 nt: np.int32,
                 dtype: Optional[np.dtype] = dtype_base,
                 sparse=sparse
                 ):
        self.wavelet = np.asarray(wavelet, dtype=dtype)
        self.nt = int(nt)
        self.dtype = dtype
        self.sparse = sparse

        # Create the acoustic operator
        wavelet_op = ConvolutionOperator(kernel=self.wavelet, offset=None, shape=(self.nt, self.nt), dtype=dtype, sparse=sparse)
        first_deriv_op = ConvolutionOperator(kernel=np.array([1, -1], dtype=dtype), offset=1, shape=(self.nt, self.nt), dtype=dtype, sparse=sparse)
        # Zero out the last row to handle the boundary condition.
        # If the operator is stored in CSC, structural changes are expensive; do it in LIL and convert back.
        if first_deriv_op.sparse:
            data_lil = first_deriv_op.data.tolil()
            data_lil[-1, :] = 0
            first_deriv_op.data = data_lil.tocsc()
        else:
            first_deriv_op.data[-1, :] = 0

        acoustic_op = 0.5 * (wavelet_op @ first_deriv_op)

        super().__init__(acoustic_op.data if hasattr(acoustic_op, "data") else acoustic_op,
                         dtype=self.dtype,
                         sparse=self.sparse)
        
