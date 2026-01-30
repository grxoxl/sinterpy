from __future__ import annotations

"""Loss functions for residual vectors."""

import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps

from .constants import ArrayF32, DTYPE_BASE

__all__ = ["L2Loss", "L1Loss", "HuberLoss", "LossFunctionBase"]

# Public aliases
array_like = ArrayF32
dtype_base = DTYPE_BASE

def ensure_array_like(arg_name: str = "r", dtype=dtype_base):
    """
    Variant B: "almost free" sanitizer.
    Converts the specified argument to np.ndarray with the requested dtype
    ONLY if it is not already np.ndarray of that dtype.

    Designed for instance methods with signature like:
        func(self, r, ...)
    Supports both positional and keyword passing of `r`.
    """

    dtype_np = np.dtype(dtype)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Keyword argument case
            if arg_name in kwargs:
                r = kwargs[arg_name]
                if not (isinstance(r, np.ndarray) and r.dtype == dtype_np):
                    kwargs[arg_name] = np.asarray(r, dtype=dtype_np)
                return func(*args, **kwargs)

            # Positional argument case: expect (self, r, ...)
            if len(args) < 2:
                raise TypeError(
                    f"{func.__name__} expects argument '{arg_name}' as the 2nd positional argument"
                )

            args = list(args)
            r = args[1]
            if not (isinstance(r, np.ndarray) and r.dtype == dtype_np):
                args[1] = np.asarray(r, dtype=dtype_np)
            return func(*args, **kwargs)

        return wrapper

    return decorator


class LossFunctionBase(ABC):
    @abstractmethod
    def value(self, r: array_like) -> dtype_base:
        raise NotImplementedError

    @abstractmethod
    def grad_r(self, r: array_like) -> npt.NDArray[np.float32]:
        raise NotImplementedError


class L2Loss(LossFunctionBase):
    """
    rho(r) = 0.5 * r^2
    grad: rho'(r) = r
    """

    @ensure_array_like("r", dtype=dtype_base)
    def value(self, r: array_like) -> dtype_base:
        # r is guaranteed np.ndarray(dtype_base) here
        return 0.5 * dtype_base(np.dot(r, r))

    @ensure_array_like("r", dtype=dtype_base)
    def grad_r(self, r: array_like) -> npt.NDArray[dtype_base]:
        # r is guaranteed np.ndarray(dtype_base) here
        return r


@dataclass(frozen=True)
class L1Loss(LossFunctionBase):
    """
    Smooth approximation of L1:
        rho(r) = sqrt(r^2 + eps^2) - eps
    grad: rho'(r) = r / sqrt(r^2 + eps^2)

    eps controls smoothness near 0.
    """
    eps: dtype_base = 1e-3

    @ensure_array_like("r", dtype=dtype_base)
    def value(self, r: array_like) -> dtype_base:
        # r is guaranteed np.ndarray(dtype_base) here
        return dtype_base(np.sum(np.sqrt(r * r + self.eps * self.eps) - self.eps))

    @ensure_array_like("r", dtype=dtype_base)
    def grad_r(self, r: array_like) -> npt.NDArray[dtype_base]:
        # r is guaranteed np.ndarray(dtype_base) here
        return r / np.sqrt(r * r + self.eps * self.eps)


@dataclass(frozen=True)
class HuberLoss(LossFunctionBase):
    """
    Huber loss:
        rho(r) = 0.5 * r^2                      if |r| <= delta
               = delta * (|r| - 0.5 * delta)    otherwise

    grad:
        rho'(r) = r                             if |r| <= delta
                = delta * sign(r)               otherwise
    """
    delta: dtype_base = 1.345

    @ensure_array_like("r", dtype=dtype_base)
    def value(self, r: array_like) -> dtype_base:
        # r is guaranteed np.ndarray(dtype_base) here
        a = np.abs(r)
        quad = a <= self.delta

        out = np.empty_like(r, dtype=dtype_base)
        out[quad] = 0.5 * r[quad] * r[quad]
        out[~quad] = self.delta * (a[~quad] - 0.5 * self.delta)
        return dtype_base(np.sum(out))

    @ensure_array_like("r", dtype=dtype_base)
    def grad_r(self, r: array_like) -> npt.NDArray[dtype_base]:
        # r is guaranteed np.ndarray(dtype_base) here
        a = np.abs(r)
        quad = a <= self.delta

        g = np.empty_like(r, dtype=dtype_base)
        g[quad] = r[quad]
        g[~quad] = self.delta * np.sign(r[~quad])
        return g
