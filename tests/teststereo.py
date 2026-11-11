"""
Unit tests for stereo field evaluation, phase correlation, and mono fallback handling.
"""

import numpy as np
import pytest

from core.stereo import evaluate_stereo_field


def test_evaluate_stereo_mono_1d():
    sr = 44100
    y = np.sin(np.linspace(0, 10, 4410)).astype(np.float32)
    audit, mono = evaluate_stereo_field(y, sr)

    assert audit.channel_count == 1
    assert audit.channel_layout_desc == "Mono (1 ch)"
    assert audit.global_correlation == 1.0
    assert audit.min_window_correlation == 1.0
    assert audit.mono_loss_db == 0.0
    assert not audit.mono_fallback_used
    assert mono.ndim == 1
    assert mono.shape[-1] == y.shape[-1]


def test_evaluate_stereo_mono_2d_single_channel():
    sr = 44100
    y = np.sin(np.linspace(0, 10, 4410))[np.newaxis, :].astype(np.float32)
    audit, mono = evaluate_stereo_field(y, sr)

    assert audit.channel_count == 1
    assert audit.channel_layout_desc == "Mono (1 ch)"
    assert not audit.mono_fallback_used


def test_evaluate_stereo_silence():
    sr = 44100
    y = np.zeros((2, 4410), dtype=np.float32)
    audit, mono = evaluate_stereo_field(y, sr)

    assert audit.channel_count == 2
    assert audit.global_correlation == 1.0
    assert audit.mono_loss_db == 0.0
    assert not audit.mono_fallback_used


def test_evaluate_stereo_dual_mono():
    sr = 44100
    signal = np.sin(np.linspace(0, 50, 44100)).astype(np.float64)
    y = np.vstack([signal, signal])
    audit, mono = evaluate_stereo_field(y, sr)

    assert audit.channel_count == 2
    assert audit.channel_layout_desc == "Stereo (2 ch)"
    assert pytest.approx(audit.global_correlation, abs=1e-4) == 1.0
    assert pytest.approx(audit.min_window_correlation, abs=1e-4) == 1.0
    assert audit.pct_negative_windows == 0.0
    assert pytest.approx(audit.mono_loss_db, abs=1e-3) == 0.0
    assert not audit.mono_fallback_used


def test_evaluate_stereo_antiphase_fallback():
    sr = 44100
    signal = np.sin(np.linspace(0, 50, 44100)).astype(np.float64)
    y = np.vstack([signal, -signal])
    audit, mono = evaluate_stereo_field(y, sr)

    assert pytest.approx(audit.global_correlation, abs=1e-4) == -1.0
    assert audit.pct_negative_windows == 100.0
    assert audit.mono_fallback_used
    assert np.allclose(mono, signal, atol=1e-4)


def test_evaluate_stereo_decorrelated():
    sr = 44100
    t = np.linspace(0, 1, sr, endpoint=False)
    left = np.sin(2 * np.pi * 440 * t)
    right = np.cos(2 * np.pi * 440 * t)
    y = np.vstack([left, right])
    audit, _ = evaluate_stereo_field(y, sr)

    assert pytest.approx(audit.global_correlation, abs=0.05) == 0.0


def test_evaluate_stereo_multichannel():
    sr = 44100
    ch = np.zeros(4410)
    y = np.vstack([ch, ch, ch, ch])
    audit, _ = evaluate_stereo_field(y, sr)

    assert audit.channel_count == 4
    assert "Multichannel (4 ch" in audit.channel_layout_desc