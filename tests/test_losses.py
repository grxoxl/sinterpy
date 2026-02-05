import unittest

import numpy as np

from sinterpy.constants import DTYPE_BASE
from sinterpy.losses import HuberLoss, L1Loss, L2Loss


class TestLosses(unittest.TestCase):
    def test_l2_value_and_grad(self) -> None:
        loss = L2Loss()
        r = np.array([1.0, -2.0, 3.0], dtype=DTYPE_BASE)

        val = loss.value(r)
        self.assertEqual(np.asarray(val).dtype, DTYPE_BASE)
        self.assertAlmostEqual(float(val), 7.0, places=6)

        grad = loss.grad_r(r)
        self.assertEqual(grad.dtype, DTYPE_BASE)
        self.assertTrue(np.array_equal(grad, r))

        # also accepts non-array input and non-f32 arrays
        self.assertAlmostEqual(float(loss.value([1.0, -2.0, 3.0])), 7.0, places=6)
        grad64 = loss.grad_r(np.array([1.0, -2.0, 3.0], dtype=np.float64))
        self.assertEqual(grad64.dtype, DTYPE_BASE)

    def test_huber_piecewise_value_and_grad(self) -> None:
        loss = HuberLoss(delta=DTYPE_BASE(1.0))
        r = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=DTYPE_BASE)

        val = loss.value(r)
        self.assertEqual(np.asarray(val).dtype, DTYPE_BASE)
        self.assertAlmostEqual(float(val), 4.0, places=6)

        grad = loss.grad_r(r)
        expected = np.array([-1.0, -1.0, 0.0, 1.0, 1.0], dtype=DTYPE_BASE)
        self.assertTrue(np.array_equal(grad, expected))

    def test_gradients_match_finite_differences(self) -> None:
        rng = np.random.default_rng(0)
        r = rng.normal(size=11).astype(DTYPE_BASE)
        eps = DTYPE_BASE(1e-4)
        idxs = (0, 3, 7)

        for loss in (L1Loss(eps=DTYPE_BASE(1e-3)), HuberLoss(delta=DTYPE_BASE(1.345))):
            g = loss.grad_r(r)
            for idx in idxs:
                e = np.zeros_like(r, dtype=DTYPE_BASE)
                e[idx] = DTYPE_BASE(1.0)
                fp = float(loss.value((r + eps * e).astype(DTYPE_BASE)))
                fm = float(loss.value((r - eps * e).astype(DTYPE_BASE)))
                fd = (fp - fm) / float(2.0 * eps)
                self.assertAlmostEqual(float(g[idx]), fd, delta=2e-2)

