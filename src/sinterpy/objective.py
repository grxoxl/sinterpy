from __future__ import annotations

"""Objective functions."""

from . import losses
import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod

from .constants import Array, DTYPE_BASE

__all__ = ["ModelBasedObjective", "ObjectiveBase"]

array_like = Array
dtype_base = DTYPE_BASE
Loss = losses.LossFunctionBase


class ObjectiveBase(ABC):

    @abstractmethod
    def __call__(self, x: Array) -> dtype_base:
        ...

    @abstractmethod
    def gradient(self, x: Array) -> Array:
        ...



class ModelBasedObjective(ObjectiveBase):

    def __init__(
        self,
        predict,                 # callable: x -> y_pred
        predict_grad,            # callable: x -> J (or function that applies J^T to a vector; see note below)
        real: array_like,        # observed y
        prior: array_like,       # x0 (prior model vector)
        loss: Loss,
        regloss: Loss = None,
        lam: dtype_base = 0.0,        # prior weight
    ):
        self.predict = predict
        self.predict_grad = predict_grad
        self.real = np.asarray(real, dtype=dtype_base).ravel()
        self.prior = np.asarray(prior, dtype=dtype_base).ravel()
        self.loss = loss
        self.regloss = regloss
        self.lam = lam

    def __call__(self, x: npt.NDArray) -> dtype_base:
        x = np.asarray(x, dtype=dtype_base).ravel()

        r_data = self.predict(x) - self.real
        f = self.loss.value(r_data)

        if self.regloss and self.lam != 0.0:
            r_prior = x - self.prior
            f = dtype_base(f + self.lam * self.regloss.value(r_prior))

        return dtype_base(f)

    def gradient(self, x: npt.NDArray) -> npt.NDArray:
        x = np.asarray(x, dtype=dtype_base).ravel()

        r_data = self.predict(x) - self.real
        g_r = self.loss.grad_r(r_data)

        J = self.predict_grad(x)          # expected shape (m, n)
        g = J.T @ g_r                     # shape (n,)

        if self.regloss and self.lam != 0.0:
            r_prior = x - self.prior
            g += self.lam * self.regloss.grad_r(r_prior)

        return np.asarray(g, dtype=dtype_base)

