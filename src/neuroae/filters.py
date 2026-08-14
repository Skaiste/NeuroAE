"""Signal filters used by NeuroAE datasets."""

from __future__ import annotations

import numpy as np


class NilearnBandPassFilter:
    """Nilearn band-pass filter with NeuroAE's ``(regions, timepoints)`` API."""

    def __init__(self, tr, high_pass=0.008, low_pass=0.08, detrend=False, ensure_finite=True):
        self.tr = float(tr)
        self.high_pass = float(high_pass)
        self.low_pass = float(low_pass)
        self.detrend = bool(detrend)
        self.ensure_finite = bool(ensure_finite)
        self._validate_band()

    def get_params(self, deep=True):
        return vars(self).copy()

    def filter(self, signal):
        """Filter a two-dimensional ``(regions, timepoints)`` signal."""
        from nilearn.signal import clean

        signal = np.asarray(signal)
        if signal.ndim != 2:
            raise ValueError("NilearnBandPassFilter requires shape (regions, timepoints).")
        filtered = clean(
            signal.T, detrend=self.detrend, standardize=False,
            high_pass=self.high_pass, low_pass=self.low_pass, t_r=self.tr,
            ensure_finite=self.ensure_finite,
        )
        return np.asarray(filtered, dtype=np.float32).T

    def _validate_band(self):
        if self.tr <= 0:
            raise ValueError("tr must be a positive number of seconds.")
        nyquist_hz = 1.0 / (2.0 * self.tr)
        if not 0 < self.high_pass < self.low_pass < nyquist_hz:
            raise ValueError(
                f"Invalid band [{self.high_pass}, {self.low_pass}] Hz for TR={self.tr} s "
                f"(Nyquist={nyquist_hz} Hz)."
            )


__all__ = ["NilearnBandPassFilter"]
