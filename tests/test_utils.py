import unittest

import numpy as np

from sinterpy.constants import DTYPE_BASE
from sinterpy.utils import butter_lowpass_filter, ricker_wavelet


class TestUtils(unittest.TestCase):
    def test_ricker_wavelet_symmetric_and_peaks_at_center(self) -> None:
        nt = np.int32(33)
        w = ricker_wavelet(f_hz=DTYPE_BASE(25.0), dt=DTYPE_BASE(0.002), nt=nt)

        self.assertEqual(w.shape, (int(nt),))
        center = int(nt) // 2
        self.assertEqual(int(np.argmax(w)), center)
        self.assertTrue(np.allclose(w, w[::-1], rtol=0.0, atol=1e-12))
        self.assertAlmostEqual(float(w[center]), 1.0, places=12)

    def test_butter_lowpass_filter_preserves_dc_and_attenuates_hf(self) -> None:
        fs = 100.0
        cutoff = 2.0
        order = 2

        t = np.arange(400, dtype=np.float64) / fs
        x_dc = np.full_like(t, 3.0)
        y_dc = butter_lowpass_filter(x_dc, cutoff=cutoff, fs=fs, order=order)
        self.assertEqual(y_dc.shape, x_dc.shape)
        self.assertTrue(np.allclose(y_dc, x_dc, rtol=0.0, atol=1e-6))

        x_hf = np.sin(2.0 * np.pi * 20.0 * t)
        y_hf = butter_lowpass_filter(x_hf, cutoff=cutoff, fs=fs, order=order)
        self.assertEqual(y_hf.shape, x_hf.shape)
        self.assertLess(float(np.std(y_hf)), 0.1 * float(np.std(x_hf)))

