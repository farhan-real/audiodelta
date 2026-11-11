"""
Unit tests for ITU-R BS.1770-4 polyphase oversampling true-peak calculations.
"""

import numpy as np
import pytest

from core.peak import compute_peak_and_true_peak


def test_compute_peak_empty():
    sp, tp = compute_peak_and_true_peak(np.array([], dtype=np.float32), 44100)
    assert sp == 0.0
    assert tp == 0.0


def test_compute_peak_silence():
    y = np.zeros(4096, dtype=np.float32)
    sp, tp = compute_peak_and_true_peak(y, 44100)
    assert sp == 0.0
    assert tp == 0.0


def test_compute_peak_dc_offset():
    y = np.full(4096, 0.5, dtype=np.float32)
    sp, tp = compute_peak_and_true_peak(y, 44100)
    assert pytest.approx(sp, abs=1e-4) == 0.5
    assert tp >= sp - 1e-3


def test_compute_peak_standard_sine():
    sr = 44100
    t = np.linspace(0.0, 1.0, sr, endpoint=False)
    y = 0.8 * np.sin(2 * np.pi * 1000 * t)
    sp, tp = compute_peak_and_true_peak(y, sr)
    assert pytest.approx(sp, abs=1e-3) == 0.8
    assert tp >= sp - 1e-4


def test_compute_peak_intersample_peak():
    sr = 48000
    freq = 12000
    n_samples = 4800
    t = np.arange(n_samples) / sr
    phase = np.pi / 4.0
    y = np.sin(2 * np.pi * freq * t + phase)
    sp, tp = compute_peak_and_true_peak(y, sr)

    assert pytest.approx(sp, abs=0.02) == np.sin(np.pi / 4.0)
    assert tp > sp + 0.15
    assert pytest.approx(tp, abs=0.08) == 1.0


def test_compute_peak_high_sample_rate():
    sr = 192000
    t = np.linspace(0.0, 0.5, int(sr * 0.5), endpoint=False)
    y = 0.75 * np.cos(2 * np.pi * 5000 * t)
    sp, tp = compute_peak_and_true_peak(y, sr)
    assert pytest.approx(sp, abs=1e-3) == 0.75
    assert tp >= sp - 1e-3


def test_compute_peak_stereo():
    sr = 44100
    ch1 = np.full(2048, 0.3, dtype=np.float32)
    ch2 = np.full(2048, 0.6, dtype=np.float32)
    y = np.vstack([ch1, ch2])
    sp, tp = compute_peak_and_true_peak(y, sr)
    assert pytest.approx(sp, abs=1e-4) == 0.6
    assert tp >= sp - 1e-3


def test_compute_peak_large_array_chunking_with_callback():
    sr = 44100
    n = 280000
    y = np.zeros(n, dtype=np.float32)
    y[100] = 0.4
    y[270000] = 0.95

    calls = []

    def callback(current, total):
        calls.append((current, total))

    sp, tp = compute_peak_and_true_peak(y, sr, progress_callback=callback)
    assert pytest.approx(sp, abs=1e-4) == 0.95
    assert tp >= 0.95
    assert len(calls) >= 2
    assert calls[-1][0] == calls[-1][1]