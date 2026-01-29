import unittest
from pathlib import Path
import sys


# Allow running tests without installing the package.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from sinterpy.losses import HuberLoss, L2Loss
from sinterpy.objective import ModelBasedObjective
from sinterpy.operators import AcousticStationaryOperator
from sinterpy.utils import butter_lowpass_filter, ricker_wavelet


class TestCheckpointExample(unittest.TestCase):
    def test_minimal_inversion_pipeline(self) -> None:
        np.random.seed(42)

        nt = 64
        z = np.linspace(0.0, 1.0, nt, dtype=np.float32)

        trend = (5000.0 + 10000.0 * z).astype(np.float32)
        noise = np.random.normal(0.0, 1.0, nt).astype(np.float32)
        noise = np.convolve(noise, np.ones(7, dtype=np.float32) / 7.0, mode="same").astype(
            np.float32
        )

        imp = trend * (1.0 + 0.10 * noise)
        imp = np.clip(imp, 1.0, None).astype(np.float32)
        x_true = np.log(imp).astype(np.float32)

        wavelet = ricker_wavelet(f_hz=np.float32(25.0), dt=np.float32(0.002), nt=np.int32(32)).astype(
            np.float32
        )
        A = AcousticStationaryOperator(
            wavelet=wavelet,
            nt=np.int32(nt),
            dtype=np.float32,
            sparse=False,
        )

        def predict(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=np.float32).ravel()
            return (A @ x).astype(np.float32)

        def predict_grad(x: np.ndarray):
            return A

        b_clean = predict(x_true)
        b_noisy = (b_clean + np.random.normal(0.0, 0.005, size=b_clean.shape).astype(np.float32)).astype(
            np.float32
        )

        x_prior = butter_lowpass_filter(x_true, cutoff=2, fs=100, order=2).astype(np.float32)

        obj = ModelBasedObjective(
            predict=predict,
            predict_grad=predict_grad,
            real=b_noisy,
            prior=x_prior,
            loss=L2Loss(),
            regloss=HuberLoss(delta=np.float32(1.345)),
            lam=np.float32(1e-2),
        )

        f0 = float(obj(x_prior))
        g0 = obj.gradient(x_prior).astype(np.float32)
        self.assertEqual(g0.shape, x_prior.shape)

        step = np.float32(0.1) / (np.linalg.norm(g0).astype(np.float32) + np.float32(1e-6))
        x1 = (x_prior - step * g0).astype(np.float32)
        f1 = float(obj(x1))

        self.assertLess(f1, f0)

        # Basic finite-difference gradient sanity check (single component).
        idx = 7
        eps = np.float32(1e-3)
        e = np.zeros_like(x_prior, dtype=np.float32)
        e[idx] = np.float32(1.0)
        fp = float(obj((x_prior + eps * e).astype(np.float32)))
        fm = float(obj((x_prior - eps * e).astype(np.float32)))
        fd = (fp - fm) / float(2.0 * eps)

        self.assertAlmostEqual(float(g0[idx]), fd, delta=1e-1)

