import unittest

import numpy as np

from sinterpy.constants import DTYPE_BASE
from sinterpy.operators import AcousticStationaryOperator, ConvolutionOperator, OperatorBase
from sinterpy.utils import ricker_wavelet


class TestOperators(unittest.TestCase):
    def test_operator_base_matvec_and_rmatvec(self) -> None:
        A = np.array([[1.0, 2.0], [3.0, 4.0], [-1.0, 0.0]], dtype=DTYPE_BASE)
        op = OperatorBase(A, dtype=DTYPE_BASE, sparse=False)

        x = np.array([0.5, -2.0], dtype=DTYPE_BASE)
        y = op.matvec(x)
        self.assertTrue(np.allclose(y, A @ x))

        z = np.array([1.0, 2.0, 3.0], dtype=DTYPE_BASE)
        w = op.rmatvec(z)
        self.assertTrue(np.allclose(w, A.T @ z))

        with self.assertRaises(ValueError):
            _ = op.matvec(np.zeros(3, dtype=DTYPE_BASE))
        with self.assertRaises(ValueError):
            _ = op.rmatvec(np.zeros(2, dtype=DTYPE_BASE))

    def test_operator_operator_matmul(self) -> None:
        A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=DTYPE_BASE)
        B = np.array([[2.0, 0.0], [1.0, -1.0]], dtype=DTYPE_BASE)

        opA = OperatorBase(A, dtype=DTYPE_BASE, sparse=False)
        opB = OperatorBase(B, dtype=DTYPE_BASE, sparse=False)

        opC = opA @ opB
        C = opC.to_numpy()
        self.assertTrue(np.allclose(C, A @ B))

    def test_convolution_operator_matches_expected_matrix(self) -> None:
        n = 4
        op = ConvolutionOperator(
            kernel=np.array([1.0, -1.0], dtype=DTYPE_BASE),
            offset=np.int32(1),
            shape=(np.int32(n), np.int32(n)),
            dtype=DTYPE_BASE,
            sparse=False,
        )

        expected = np.array(
            [
                [-1.0, 1.0, 0.0, 0.0],
                [0.0, -1.0, 1.0, 0.0],
                [0.0, 0.0, -1.0, 1.0],
                [0.0, 0.0, 0.0, -1.0],
            ],
            dtype=DTYPE_BASE,
        )
        self.assertTrue(np.array_equal(op.to_numpy(), expected))

        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=DTYPE_BASE)
        y = op @ x
        self.assertTrue(np.allclose(y, np.array([1.0, 1.0, 1.0, -4.0], dtype=DTYPE_BASE)))

        with self.assertRaises(ValueError):
            _ = ConvolutionOperator(
                kernel=np.array([1.0, -1.0], dtype=DTYPE_BASE),
                offset=np.int32(n),
                shape=(np.int32(n), np.int32(n)),
                dtype=DTYPE_BASE,
                sparse=False,
            )

    def test_acoustic_stationary_operator_constant_model_zero(self) -> None:
        nt = np.int32(16)
        wavelet = ricker_wavelet(
            f_hz=DTYPE_BASE(25.0),
            dt=DTYPE_BASE(0.002),
            nt=np.int32(9),
        ).astype(DTYPE_BASE)

        A = AcousticStationaryOperator(wavelet=wavelet, nt=nt, dtype=DTYPE_BASE, sparse=False)
        x = np.full(int(nt), 2.0, dtype=DTYPE_BASE)
        y = A @ x

        self.assertEqual(y.dtype, DTYPE_BASE)
        self.assertTrue(np.allclose(y, 0.0, rtol=0.0, atol=1e-6))

