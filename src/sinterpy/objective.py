from __future__ import annotations

"""Objective functions."""

from . import losses
import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod

from .constants import ArrayF32, DTYPE_BASE, SPARSE
from .operators import ConvolutionOperator

__all__ = ["ModelBasedObjective", "SparseSpikeObjective", "ObjectiveBase"]

array_like = ArrayF32
dtype_base = DTYPE_BASE
Loss = losses.LossFunctionBase


class ObjectiveBase(ABC):

    @abstractmethod
    def __call__(self, x: ArrayF32) -> dtype_base:
        ...

    @abstractmethod
    def gradient(self, x: ArrayF32) -> ArrayF32:
        ...



class ModelBasedObjective(ObjectiveBase):

    def __init__(
        self,
        predict,                 # callable: x -> y_pred
        predict_grad,            # callable: x -> J (or applies J^T to a vector)
        real: array_like,        # observed y
        prior: array_like,       # x0 (prior model vector)
        misfit_loss: Loss,       # data misfit loss
        mbi_loss: Loss = None,   # prior/regularization loss
        lam: dtype_base = 0.0,   # prior weight
    ):
        self.predict = predict
        self.predict_grad = predict_grad
        self.real = np.asarray(real, dtype=dtype_base).ravel()
        self.prior = np.asarray(prior, dtype=dtype_base).ravel()
        self.misfit_loss = misfit_loss
        self.mbi_loss = mbi_loss
        self.lam = lam

    def _as_x(self, x: npt.NDArray) -> npt.NDArray:
        return np.asarray(x, dtype=dtype_base).ravel()

    def __call__(self, x: npt.NDArray) -> dtype_base:
        x = self._as_x(x)

        r_data = self.predict(x) - self.real
        f = self.misfit_loss.value(r_data)

        if self.mbi_loss and self.lam != 0.0:
            r_prior = x - self.prior
            f = dtype_base(f + self.lam * self.mbi_loss.value(r_prior))

        return dtype_base(f)

    def gradient(self, x: npt.NDArray) -> npt.NDArray:
        x = self._as_x(x)

        r_data = self.predict(x) - self.real
        g_r = self.misfit_loss.grad_r(r_data)

        J = self.predict_grad(x)          # expected shape (m, n)
        g = J.T @ g_r                     # shape (n,)

        if self.mbi_loss and self.lam != 0.0:
            r_prior = x - self.prior
            g += self.lam * self.mbi_loss.grad_r(r_prior)

        return np.asarray(g, dtype=dtype_base)


class SparseSpikeObjective(ModelBasedObjective):
    """
    Model-based objective with L1 penalty on reflection coefficients: D * lnX.

    The L1 term uses a (smooth) L1 loss to keep the objective differentiable.
    """

    def __init__(
        self,
        predict,                 # callable: x -> y_pred
        predict_grad,            # callable: x -> J (or applies J^T to a vector)
        real: array_like,        # observed y
        prior: array_like,       # x0 (prior model vector)
        misfit_loss: Loss,       # model-based inversion loss
        mbi_loss: Loss = None,   # prior/regularization loss
        mbi_lam: dtype_base = 0.0,  # prior weight
        ssi_loss: Loss = None,   # sparse-spike (L1) loss on D @ x
        ssi_lam: dtype_base = 0.0,  # sparse-spike weight
    ):
        super().__init__(
            predict=predict,
            predict_grad=predict_grad,
            real=real,
            prior=prior,
            misfit_loss=misfit_loss,
            mbi_loss=mbi_loss,
            lam=mbi_lam,
        )
        n = int(np.asarray(prior, dtype=dtype_base).size)
        self.deriv_op = ConvolutionOperator(
            kernel=np.array([1, -1], dtype=dtype_base),
            offset=1,
            shape=(n, n),
            dtype=dtype_base,
            sparse=SPARSE,
        )
        self.ssi_loss = ssi_loss if ssi_loss is not None else losses.L1Loss()
        self.ssi_lam = ssi_lam

    def __call__(self, x: npt.NDArray) -> dtype_base:
        x = self._as_x(x)
        f = super().__call__(x)

        if self.ssi_lam != 0.0:
            r_refl = self.deriv_op @ x
            f = dtype_base(f + self.ssi_lam * self.ssi_loss.value(r_refl))

        return dtype_base(f)

    def gradient(self, x: npt.NDArray) -> npt.NDArray:
        x = self._as_x(x)
        g = super().gradient(x)

        if self.ssi_lam != 0.0:
            r_refl = self.deriv_op @ x
            g_refl = self.ssi_loss.grad_r(r_refl)
            g = g + self.ssi_lam * (self.deriv_op.T @ g_refl)

        return np.asarray(g, dtype=dtype_base)

