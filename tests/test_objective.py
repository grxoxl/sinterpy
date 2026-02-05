import unittest

import numpy as np

from sinterpy.constants import DTYPE_BASE
from sinterpy.losses import L2Loss
from sinterpy.objective import ModelBasedObjective, SparseSpikeObjective


class TestObjective(unittest.TestCase):
    def test_model_based_objective_matches_manual(self) -> None:
        A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=DTYPE_BASE)
        b = np.array([1.0, -1.0], dtype=DTYPE_BASE)
        x0 = np.array([0.25, -0.5], dtype=DTYPE_BASE)
        lam = DTYPE_BASE(0.5)

        def predict(x: np.ndarray) -> np.ndarray:
            return (A @ np.asarray(x, dtype=DTYPE_BASE).ravel()).astype(DTYPE_BASE)

        def predict_grad(x: np.ndarray):
            return A

        obj = ModelBasedObjective(
            predict=predict,
            predict_grad=predict_grad,
            real=b,
            prior=x0,
            misfit_loss=L2Loss(),
            mbi_loss=L2Loss(),
            lam=lam,
        )

        x = np.array([0.5, -0.25], dtype=DTYPE_BASE)
        r = (A @ x - b).astype(DTYPE_BASE)
        r_prior = (x - x0).astype(DTYPE_BASE)

        expected_f = 0.5 * float(np.dot(r, r)) + float(lam) * 0.5 * float(np.dot(r_prior, r_prior))
        self.assertAlmostEqual(float(obj(x)), expected_f, places=6)

        expected_g = (A.T @ r + lam * r_prior).astype(DTYPE_BASE)
        g = obj.gradient(x)
        self.assertEqual(g.dtype, DTYPE_BASE)
        self.assertTrue(np.allclose(g, expected_g, rtol=1e-6, atol=1e-6))

        # Finite-difference gradient sanity check (single component).
        idx = 1
        eps = DTYPE_BASE(1e-4)
        e = np.zeros_like(x, dtype=DTYPE_BASE)
        e[idx] = DTYPE_BASE(1.0)
        fp = float(obj((x + eps * e).astype(DTYPE_BASE)))
        fm = float(obj((x - eps * e).astype(DTYPE_BASE)))
        fd = (fp - fm) / float(2.0 * eps)
        self.assertAlmostEqual(float(g[idx]), fd, delta=1e-2)

    def test_sparse_spike_objective_l2_penalty(self) -> None:
        rng = np.random.default_rng(1)
        n = 8
        x = rng.normal(size=n).astype(DTYPE_BASE)
        zeros = np.zeros(n, dtype=DTYPE_BASE)

        def predict(x: np.ndarray) -> np.ndarray:
            return np.asarray(x, dtype=DTYPE_BASE).ravel()

        def predict_grad(x: np.ndarray):
            return np.eye(n, dtype=DTYPE_BASE)

        obj = SparseSpikeObjective(
            predict=predict,
            predict_grad=predict_grad,
            real=zeros,
            prior=zeros,
            misfit_loss=L2Loss(),
            mbi_loss=None,
            mbi_lam=DTYPE_BASE(0.0),
            ssi_loss=L2Loss(),
            ssi_lam=DTYPE_BASE(0.1),
        )

        Dx = obj.deriv_op @ x
        expected_f = 0.5 * float(np.dot(x, x)) + float(obj.ssi_lam) * 0.5 * float(np.dot(Dx, Dx))
        self.assertAlmostEqual(float(obj(x)), expected_f, places=6)

        expected_g = (x + obj.ssi_lam * (obj.deriv_op.T @ Dx)).astype(DTYPE_BASE)
        g = obj.gradient(x)
        self.assertEqual(g.dtype, DTYPE_BASE)
        self.assertTrue(np.allclose(g, expected_g, rtol=1e-6, atol=1e-6))

        # Finite-difference gradient sanity check (single component).
        idx = 2
        eps = DTYPE_BASE(1e-4)
        e = np.zeros_like(x, dtype=DTYPE_BASE)
        e[idx] = DTYPE_BASE(1.0)
        fp = float(obj((x + eps * e).astype(DTYPE_BASE)))
        fm = float(obj((x - eps * e).astype(DTYPE_BASE)))
        fd = (fp - fm) / float(2.0 * eps)
        self.assertAlmostEqual(float(g[idx]), fd, delta=1e-2)

