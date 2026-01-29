"""Small DSP utilities."""

import numpy as np
from scipy.signal import butter, filtfilt

from .constants import Array, DTYPE_BASE

dtype_base = DTYPE_BASE

__all__ = ["ricker_wavelet", "butter_lowpass_filter"]

def ricker_wavelet(f_hz: DTYPE_BASE, dt: DTYPE_BASE, nt: np.int32) -> Array:
    t0 = (nt - 1) / 2 * dt
    t = np.arange(nt) * dt - t0
    a = (np.pi * f_hz * t) ** 2
    return (1 - 2 * a) * np.exp(-a)

def butter_lowpass_filter(data, cutoff, fs, order):
    """
    Lowpass filter
    fs - sample rate, Hz
    cutoff - desired cutoff frequency, Hz
    order - sin wave can be approx represented as quadratic
    """
    normal_cutoff = 2*cutoff / fs
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y
