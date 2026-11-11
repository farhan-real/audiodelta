"""
Key estimation, 4-band spectral balance, MFCC/chroma extraction, and track analysis pipeline.
"""

import math
from pathlib import Path
import sys
from typing import Dict, Tuple
import warnings
import librosa
import numpy as np
import pyloudnorm as pyln
import scipy.signal

from core.constants import (
    ANALYSIS_SR,
    EPS,
    HOP_LENGTH,
    MAJOR_PROFILE,
    MINOR_PROFILE,
    PITCH_NAMES,
)
from core.models import TrackAnalysis
from core.peak import compute_peak_and_true_peak
from core.stereo import evaluate_stereo_field
from core.utils import ProgressBar, suppress_c_stderr


def estimate_key(chroma: np.ndarray) -> str:
    """Estimates the musical key using Krumhansl-Schmuckler key profile correlation."""
    if chroma is None or chroma.size == 0 or chroma.ndim != 2 or chroma.shape[1] == 0:
        return "Undetermined"

    mean_chroma = np.mean(chroma, axis=1)
    if np.std(mean_chroma) < EPS or not np.all(np.isfinite(mean_chroma)):
        return "Undetermined"

    scores = []
    for tonic, pitch in enumerate(PITCH_NAMES):
        for mode, profile in [("maj", MAJOR_PROFILE), ("min", MINOR_PROFILE)]:
            rotated = np.roll(profile, tonic)
            corr = np.corrcoef(mean_chroma, rotated)[0, 1]
            scores.append((corr if math.isfinite(corr) else -1.0, f"{pitch} {mode}"))

    if not scores:
        return "Undetermined"
    scores.sort(reverse=True)
    return scores[0][1]


def compute_spectral_balance(y_raw: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Computes four-band relative spectral power distribution using averaged Welch PSD
    across available audio channels. Independent of mono sum phase cancellations.
    """
    n_channels = 1 if y_raw.ndim == 1 else y_raw.shape[0]
    nperseg = min(4096, y_raw.shape[-1])
    if nperseg < 64:
        return {}

    psd_list = []
    channels_to_read = min(n_channels, 2)
    for ch in range(channels_to_read):
        sig = y_raw[ch] if y_raw.ndim == 2 else y_raw
        freqs, psd = scipy.signal.welch(
            sig.astype(np.float64),
            fs=sr,
            window="hann",
            nperseg=nperseg,
            scaling="density"
        )
        psd_list.append(psd)

    psd_avg = np.mean(psd_list, axis=0)

    bands = {
        "Sub/Bass (20-250 Hz)": (20.0, 250.0),
        "Low-Mid (250-1000 Hz)": (250.0, 1000.0),
        "High-Mid (1-5 kHz)":    (1000.0, 5000.0),
        "Air (5-20 kHz)":        (5000.0, 20000.0),
    }

    powers = {}
    for name, (low, high) in bands.items():
        idx = np.where((freqs >= low) & (freqs < high))[0]
        powers[name] = float(np.sum(psd_avg[idx])) if len(idx) > 0 else EPS

    total = sum(powers.values()) + EPS
    return {k: 10 * math.log10(max(v, EPS) / total) for k, v in powers.items()}


def extract_features(y_mono: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts independent harmonic (chroma) and timbral (CMVN MFCC) features.
    Normalizes each feature group to unit Euclidean length independently before
    applying the calibrated 70/30 weighting.
    """
    try:
        chroma = librosa.feature.chroma_cqt(
            y=y_mono,
            sr=ANALYSIS_SR,
            hop_length=HOP_LENGTH,
            fmin=librosa.note_to_hz("C1"),
            n_octaves=7
        )
    except Exception:
        chroma = librosa.feature.chroma_stft(
            y=y_mono,
            sr=ANALYSIS_SR,
            hop_length=HOP_LENGTH
        )

    mfcc = librosa.feature.mfcc(
        y=y_mono,
        sr=ANALYSIS_SR,
        n_mfcc=14,
        hop_length=HOP_LENGTH
    )[1:]

    n_frames = min(chroma.shape[1], mfcc.shape[1])
    if n_frames == 0:
        return (
            np.empty((12, 0), dtype=np.float32),
            np.empty((13, 0), dtype=np.float32),
            np.empty((25, 0), dtype=np.float32)
        )

    chroma = chroma[:, :n_frames]
    mfcc = mfcc[:, :n_frames]

    # Normalize pure chroma per frame
    chroma_norm = chroma / np.maximum(np.linalg.norm(chroma, axis=0, keepdims=True), EPS)

    # CMVN on MFCC across time to ensure timbre comparisons remain invariant to EQ changes
    mfcc_mean = np.mean(mfcc, axis=1, keepdims=True)
    mfcc_std = np.std(mfcc, axis=1, keepdims=True)
    mfcc_cmvn = (mfcc - mfcc_mean) / np.maximum(mfcc_std, EPS)
    mfcc_norm = mfcc_cmvn / np.maximum(np.linalg.norm(mfcc_cmvn, axis=0, keepdims=True), EPS)

    # Combine with calibrated 70/30 energy weighting
    combined = np.vstack([chroma_norm * 0.70, mfcc_norm * 0.30])
    combined /= np.maximum(np.linalg.norm(combined, axis=0, keepdims=True), EPS)

    return chroma_norm.astype(np.float32), mfcc_norm.astype(np.float32), combined.astype(np.float32)


def analyze_track(path: Path, show_progress: bool = True) -> TrackAnalysis:
    """
    Analyzes a complete audio track given its file system path with real-time progress updates.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    track_name = path.name

    def notify_step(step_num: int, message: str) -> None:
        if show_progress:
            sys.stdout.write(f"  [{step_num}/6] {track_name}: {message}\n")
            sys.stdout.flush()

    notify_step(1, "Loading and decoding audio stream...")
    with suppress_c_stderr(), warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        y_raw, sr = librosa.load(str(path), sr=None, mono=False)
    if y_raw.size == 0:
        raise ValueError(f"Empty audio file: {path}")

    duration = y_raw.shape[-1] / sr

    notify_step(2, "Auditing stereo correlation and phase stability...")
    stereo_audit, y_analysis_mono = evaluate_stereo_field(y_raw, sr)

    notify_step(3, "Calculating ITU-R BS.1770-4 polyphase true peak...")
    bar = ProgressBar(total=1, prefix=f"        Filtering", length=24) if show_progress else None

    def peak_cb(current_chunk: int, total_chunks: int) -> None:
        if bar is not None:
            bar.total = total_chunks
            bar.update(current_chunk, f"({current_chunk}/{total_chunks} chunks)")

    sample_peak_lin, true_peak_lin = compute_peak_and_true_peak(
        y_raw,
        sr,
        progress_callback=peak_cb
    )
    if bar is not None:
        bar.finish()

    sample_peak_dbfs = 20 * math.log10(max(sample_peak_lin, EPS))
    true_peak_dbtp = 20 * math.log10(max(true_peak_lin, EPS))

    rms_linear = float(np.sqrt(np.mean(y_raw.astype(np.float64) ** 2)))
    crest_factor = 20 * math.log10(max(true_peak_lin / max(rms_linear, EPS), EPS))

    notify_step(4, "Measuring integrated LUFS and loudness range...")
    if y_raw.ndim == 2:
        channels_to_use = min(y_raw.shape[0], 5)
        audio_for_ln = np.ascontiguousarray(y_raw[:channels_to_use].T, dtype=np.float64)
    else:
        audio_for_ln = np.ascontiguousarray(y_analysis_mono, dtype=np.float64)

    lufs = None
    lra = None
    try:
        meter = pyln.Meter(sr)
        val = float(meter.integrated_loudness(audio_for_ln))
        if math.isfinite(val):
            lufs = val
    except Exception:
        pass

    try:
        meter = pyln.Meter(sr)
        val = float(meter.loudness_range(audio_for_ln))
        if math.isfinite(val):
            lra = val
    except Exception:
        pass

    notify_step(5, "Resampling and extracting harmonic/timbral features...")
    if sr == ANALYSIS_SR:
        y_resamp = np.ascontiguousarray(y_analysis_mono, dtype=np.float32)
    else:
        y_resamp = librosa.resample(
            np.ascontiguousarray(y_analysis_mono, dtype=np.float32),
            orig_sr=sr,
            target_sr=ANALYSIS_SR
        )

    try:
        tempo, _ = librosa.beat.beat_track(y=y_resamp, sr=ANALYSIS_SR)
        bpm = float(np.asarray(tempo).reshape(-1)[0]) if np.size(tempo) else 0.0
        if not math.isfinite(bpm):
            bpm = 0.0
    except Exception:
        bpm = 0.0

    chroma_features, timbral_features, combined_features = extract_features(y_resamp)

    notify_step(6, "Evaluating musical key and 4-band spectral distribution...")
    key_est = estimate_key(chroma_features)
    spectral_balance = compute_spectral_balance(y_raw, sr)

    return TrackAnalysis(
        path=path,
        duration=duration,
        sample_rate=sr,
        bpm=bpm,
        key=key_est,
        lufs=lufs,
        lra=lra,
        sample_peak_dbfs=sample_peak_dbfs,
        true_peak_dbtp=true_peak_dbtp,
        crest_factor_db=crest_factor,
        stereo=stereo_audit,
        spectral_balance_db=spectral_balance,
        chroma_features=chroma_features,
        timbral_features=timbral_features,
        combined_features=combined_features
    )
