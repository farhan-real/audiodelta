"""
Unit tests for key estimation, spectral balance, feature extraction, and track analysis.
"""

from pathlib import Path
import tempfile
import numpy as np
import pytest
import soundfile as sf

from core.constants import MAJOR_PROFILE, MINOR_PROFILE
from core.features import (
    analyze_track,
    compute_spectral_balance,
    estimate_key,
    extract_features,
)


def test_estimate_key_undetermined_cases():
    assert estimate_key(None) == "Undetermined"
    assert estimate_key(np.array([])) == "Undetermined"
    assert estimate_key(np.zeros((12, 0))) == "Undetermined"
    assert estimate_key(np.ones(12)) == "Undetermined"
    assert estimate_key(np.zeros((12, 10))) == "Undetermined"


def test_estimate_key_c_major():
    chroma = np.repeat(MAJOR_PROFILE[:, np.newaxis], 20, axis=1)
    key = estimate_key(chroma)
    assert key == "C maj"


def test_estimate_key_a_minor():
    rotated_minor = np.roll(MINOR_PROFILE, 9)
    chroma = np.repeat(rotated_minor[:, np.newaxis], 20, axis=1)
    key = estimate_key(chroma)
    assert key == "A min"


def test_compute_spectral_balance_too_short():
    short_signal = np.zeros(32, dtype=np.float32)
    assert compute_spectral_balance(short_signal, 44100) == {}


def test_compute_spectral_balance_low_vs_high():
    sr = 44100
    t = np.linspace(0, 1, sr, endpoint=False)

    bass_sig = np.sin(2 * np.pi * 100 * t)
    bass_bal = compute_spectral_balance(bass_sig, sr)
    assert bass_bal["Sub/Bass (20-250 Hz)"] > bass_bal["High-Mid (1-5 kHz)"]

    air_sig = np.sin(2 * np.pi * 10000 * t)
    air_bal = compute_spectral_balance(air_sig, sr)
    assert air_bal["Air (5-20 kHz)"] > air_bal["Sub/Bass (20-250 Hz)"]


def test_extract_features_empty():
    c, m, comb = extract_features(np.array([], dtype=np.float32))
    assert c.shape == (12, 0)
    assert m.shape == (13, 0)
    assert comb.shape == (25, 0)


def test_extract_features_sine_normalizations():
    sr = 22050
    t = np.linspace(0, 1.5, int(sr * 1.5), endpoint=False)
    y = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    c, m, comb = extract_features(y)
    assert c.shape[0] == 12
    assert m.shape[0] == 13
    assert comb.shape[0] == 25
    assert c.shape[1] > 0
    assert c.shape[1] == m.shape[1] == comb.shape[1]

    # Verify column Euclidean norms are approximately 1.0
    c_norms = np.linalg.norm(c, axis=0)
    assert np.allclose(c_norms, 1.0, atol=1e-3)

    m_norms = np.linalg.norm(m, axis=0)
    assert np.allclose(m_norms, 1.0, atol=1e-3)

    comb_norms = np.linalg.norm(comb, axis=0)
    assert np.allclose(comb_norms, 1.0, atol=1e-3)


def test_analyze_track_missing_file():
    with pytest.raises(FileNotFoundError):
        analyze_track(Path("non_existent_audio_file.wav"))


def test_analyze_track_synthetic_wav_with_and_without_progress(capsys):
    sr = 44100
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    ch1 = 0.5 * np.sin(2 * np.pi * 440 * t)
    ch2 = 0.5 * np.sin(2 * np.pi * 440 * t)
    stereo_wave = np.vstack([ch1, ch2]).T.astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp_dir:
        wav_path = Path(tmp_dir) / "synth.wav"
        sf.write(str(wav_path), stereo_wave, sr)

        analysis_prog = analyze_track(wav_path, show_progress=True)
        out = capsys.readouterr().out
        assert "[1/6]" in out
        assert "[6/6]" in out
        assert analysis_prog.duration == pytest.approx(duration, abs=0.05)

        analysis_quiet = analyze_track(wav_path, show_progress=False)
        out_quiet = capsys.readouterr().out
        assert out_quiet == ""
        assert analysis_quiet.sample_rate == sr