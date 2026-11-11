#!/usr/bin/env python3
"""
Audio Mix, Edit and Structural Comparison Tool

Engineered for:
- Mix vs. Master QA (LUFS, LRA, Crest Factor, True Peak, 4-Band Spectral Balance)
- Multi-Window Stereo Phase, Correlation Stability, and Mono Fold-Down Loss
- Dual-Domain Feature Extraction (Independent Harmonic and Timbral Similarities)
- Speed-Ratio Aware (Slope-Preserving) Snippet and Sample Discovery
- Bipartite Interval Coverage (Zero False Attribution Across Edits)

Dependencies:
    pip install librosa numpy scipy soundfile pyloudnorm
"""

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import pyloudnorm as pyln
import scipy.signal
from scipy.ndimage import uniform_filter1d

# ============================================================
# CONSTANTS AND FILTER COEFFICIENTS
# ============================================================

ANALYSIS_SR = 22050
HOP_LENGTH = 2048  # ~92.9 ms per frame (~10.8 fps)
EPS = 1e-9

PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# ITU-R BS.1770-4 Table 2: 48-tap interpolation FIR filter (4 phases x 12 taps)
BS1770_PHASE_0 = np.array([
    0.001708984375,  0.010986328125, -0.0196533203125,  0.033203125,
   -0.0594482421875,  0.1373291015625,  0.97216796875,   -0.102294921875,
    0.047607421875, -0.026611328125,  0.014892578125, -0.00830078125
], dtype=np.float64)

BS1770_PHASE_1 = np.array([
   -0.0291748046875,  0.029296875,    -0.0517578125,    0.089111328125,
   -0.16650390625,   0.465087890625,   0.77978515625,  -0.2003173828125,
    0.1015625,      -0.0582275390625,  0.0330810546875, -0.0189208984375
], dtype=np.float64)

BS1770_PHASE_2 = np.array([
   -0.0189208984375,  0.0330810546875, -0.0582275390625,  0.1015625,
   -0.2003173828125,  0.77978515625,    0.465087890625, -0.16650390625,
    0.089111328125,  -0.0517578125,    0.029296875,    -0.0291748046875
], dtype=np.float64)

BS1770_PHASE_3 = np.array([
   -0.00830078125,   0.014892578125,  -0.026611328125,  0.047607421875,
   -0.102294921875,   0.97216796875,    0.1373291015625, -0.0594482421875,
    0.033203125,    -0.0196533203125,  0.010986328125,   0.001708984375
], dtype=np.float64)

# Interleaved 48-tap polyphase filter
BS1770_FILTER_48 = np.zeros(48, dtype=np.float64)
BS1770_FILTER_48[0::4] = BS1770_PHASE_0
BS1770_FILTER_48[1::4] = BS1770_PHASE_1
BS1770_FILTER_48[2::4] = BS1770_PHASE_2
BS1770_FILTER_48[3::4] = BS1770_PHASE_3

# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class MatchedSnippet:
    a_start: float
    a_end: float
    b_start: float
    b_end: float
    duration_a: float
    duration_b: float
    speed_ratio: float
    composite_similarity: float
    harmonic_similarity: float
    timbral_similarity: float

@dataclass
class StereoAudit:
    channel_count: int
    channel_layout_desc: str
    global_correlation: float
    min_window_correlation: float
    pct_negative_windows: float
    mono_loss_db: float
    mid_side_ratio_db: float
    mono_fallback_used: bool

@dataclass
class TrackAnalysis:
    path: Path
    duration: float
    sample_rate: int
    bpm: float
    key: str
    lufs: Optional[float]
    lra: Optional[float]
    sample_peak_dbfs: float
    true_peak_dbtp: float
    crest_factor_db: float
    stereo: StereoAudit
    spectral_balance_db: Dict[str, float]
    chroma_features: np.ndarray  # Shape: (12, frames)
    timbral_features: np.ndarray  # Shape: (13, frames)
    combined_features: np.ndarray  # Shape: (25, frames)

# ============================================================
# UTILITIES
# ============================================================

def fmt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    total_hundredths = int(round(seconds * 100))
    minutes = total_hundredths // 6000
    secs = (total_hundredths % 6000) / 100.0
    return f"{minutes:02d}:{secs:05.2f}"

def subtract_intervals(
    interval: Tuple[float, float],
    claimed: List[Tuple[float, float]]
) -> List[Tuple[float, float]]:
    """
    Subtracts a list of claimed disjoint intervals from a single candidate interval.
    Returns the remaining disjoint sub-intervals.
    """
    remaining = [interval]
    for cs, ce in claimed:
        next_rem = []
        for rs, re in remaining:
            if ce <= rs or cs >= re:
                next_rem.append((rs, re))
            else:
                if cs > rs:
                    next_rem.append((rs, cs))
                if ce < re:
                    next_rem.append((ce, re))
        remaining = next_rem
        if not remaining:
            break
    return [iv for iv in remaining if (iv[1] - iv[0]) > 0.05]

# ============================================================
# VERIFIED TRUE PEAK ENGINE
# ============================================================

def compute_peak_and_true_peak(y: np.ndarray, sr: int) -> Tuple[float, float]:
    """
    Computes Sample Peak (dBFS) and True Peak (dBTP).
    Uses ITU-R BS.1770-4 48-tap 4x polyphase FIR filtering with exact group-delay
    alignment (tau = 23.5 samples at 4x rate). For sr > 96k, applies 2x oversampling.
    """
    if y.size == 0:
        return 0.0, 0.0

    y_2d = np.atleast_2d(y.astype(np.float64))
    sample_peak = float(np.max(np.abs(y_2d)))

    chunk_size = 262144
    pad_len = 32
    n_samples = y_2d.shape[-1]
    true_peak = sample_peak

    if sr <= 96000:
        # Standard ITU-R BS.1770-4 4x oversampling filter
        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            left_pad = min(start, pad_len)
            right_pad = min(n_samples - end, pad_len)

            chunk = y_2d[:, start - left_pad : end + right_pad]
            pre_pad = pad_len - left_pad
            post_pad = pad_len - right_pad

            if pre_pad > 0 or post_pad > 0:
                pad_mode = "reflect" if chunk.shape[-1] > 1 else "constant"
                chunk = np.pad(chunk, ((0, 0), (pre_pad, post_pad)), mode=pad_mode)

            y_up = scipy.signal.upfirdn(BS1770_FILTER_48, chunk, up=4, down=1, axis=-1)

            # Account for 48-tap filter group delay: tau = (48 - 1) / 2 = 23.5 samples
            # Base pad_len * 4 is sample index 0; offset by 22 samples to encompass peak center
            core_start = pad_len * 4 + 22
            core_end = core_start + (end - start) * 4
            core = y_up[:, core_start:core_end]

            if core.size > 0:
                true_peak = max(true_peak, float(np.max(np.abs(core))))
    else:
        # High-rate (> 96 kHz) 2x oversampling filter with reflection margins
        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            left_pad = min(start, pad_len)
            right_pad = min(n_samples - end, pad_len)

            chunk = y_2d[:, start - left_pad : end + right_pad]
            pre_pad = pad_len - left_pad
            post_pad = pad_len - right_pad

            if pre_pad > 0 or post_pad > 0:
                pad_mode = "reflect" if chunk.shape[-1] > 1 else "constant"
                chunk = np.pad(chunk, ((0, 0), (pre_pad, post_pad)), mode=pad_mode)

            y_up = scipy.signal.resample_poly(chunk, 2, 1, axis=-1)
            core_start = pad_len * 2
            core_end = core_start + (end - start) * 2
            core = y_up[:, core_start:core_end]

            if core.size > 0:
                true_peak = max(true_peak, float(np.max(np.abs(core))))

    return sample_peak, true_peak

# ============================================================
# STEREO AND MONO COMPATIBILITY AUDIT
# ============================================================

def evaluate_stereo_field(y_raw: np.ndarray, sr: int) -> Tuple[StereoAudit, np.ndarray]:
    """
    Computes global correlation, windowed sliding correlation, mono fold-down power loss,
    and returns a safe mono signal for feature extraction.
    """
    n_channels = 1 if y_raw.ndim == 1 else y_raw.shape[0]

    if n_channels == 1:
        mono_signal = (y_raw if y_raw.ndim == 1 else y_raw[0]).astype(np.float32)
        audit = StereoAudit(
            channel_count=1,
            channel_layout_desc="Mono (1 ch)",
            global_correlation=1.0,
            min_window_correlation=1.0,
            pct_negative_windows=0.0,
            mono_loss_db=0.0,
            mid_side_ratio_db=-99.9,
            mono_fallback_used=False
        )
        return audit, mono_signal

    layout_desc = "Stereo (2 ch)" if n_channels == 2 else f"Multichannel ({n_channels} ch, Front L/R audited)"

    left = y_raw[0].astype(np.float64)
    right = y_raw[1].astype(np.float64)
    n_samples = left.shape[-1]

    e_left = float(np.mean(left ** 2))
    e_right = float(np.mean(right ** 2))

    if e_left < 1e-12 and e_right < 1e-12:
        audit = StereoAudit(n_channels, layout_desc, 1.0, 1.0, 0.0, 0.0, -99.9, False)
        return audit, left.astype(np.float32)

    denom = (np.sqrt(np.sum(left ** 2)) * np.sqrt(np.sum(right ** 2))) + EPS
    global_corr = float(np.clip(np.sum(left * right) / denom, -1.0, 1.0))

    mid = (left + right) / 2.0
    side = (left - right) / 2.0
    e_mid = float(np.mean(mid ** 2))
    e_side = float(np.mean(side ** 2))
    e_stereo_avg = (e_left + e_right) / 2.0

    ms_ratio = float(10 * math.log10(max(e_side, EPS) / max(e_mid, EPS)))
    mono_loss = float(10 * math.log10(max(e_mid, EPS) / max(e_stereo_avg, EPS)))

    # Windowed correlation tracking (100 ms sliding window, 50% overlap)
    win_size = max(512, int(sr * 0.100))
    hop = win_size // 2
    window_corrs = []

    for start in range(0, n_samples - win_size + 1, hop):
        lw = left[start : start + win_size]
        rw = right[start : start + win_size]
        p_lw = np.sum(lw ** 2)
        p_rw = np.sum(rw ** 2)

        if (p_lw / win_size) > 1e-6 and (p_rw / win_size) > 1e-6:
            r = np.sum(lw * rw) / (np.sqrt(p_lw * p_rw) + EPS)
            window_corrs.append(float(r))

    if window_corrs:
        min_win_corr = float(np.min(window_corrs))
        pct_negative = float(np.mean(np.array(window_corrs) < 0.0) * 100.0)
    else:
        min_win_corr = global_corr
        pct_negative = 0.0

    # Protected mono downmix for feature extraction
    if e_mid < 0.01 * (e_left + EPS):
        y_mono = left.astype(np.float32)
        fallback_used = True
    else:
        y_mono = mid.astype(np.float32)
        fallback_used = False

    audit = StereoAudit(
        channel_count=n_channels,
        channel_layout_desc=layout_desc,
        global_correlation=global_corr,
        min_window_correlation=min_win_corr,
        pct_negative_windows=pct_negative,
        mono_loss_db=mono_loss,
        mid_side_ratio_db=ms_ratio,
        mono_fallback_used=fallback_used
    )
    return audit, y_mono

# ============================================================
# SPECTRAL AND FEATURE ANALYSIS
# ============================================================

def estimate_key(chroma: np.ndarray) -> str:
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
    applying the nominal 70/30 weighting.
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

    # CMVN on MFCC across time
    mfcc_mean = np.mean(mfcc, axis=1, keepdims=True)
    mfcc_std = np.std(mfcc, axis=1, keepdims=True)
    mfcc_cmvn = (mfcc - mfcc_mean) / np.maximum(mfcc_std, EPS)

    # Normalize MFCC per frame so expected group norm equals 1.0
    mfcc_norm = mfcc_cmvn / np.maximum(np.linalg.norm(mfcc_cmvn, axis=0, keepdims=True), EPS)

    # Combine with calibrated 70/30 energy weighting
    combined = np.vstack([chroma_norm * 0.70, mfcc_norm * 0.30])
    combined /= np.maximum(np.linalg.norm(combined, axis=0, keepdims=True), EPS)

    return chroma_norm.astype(np.float32), mfcc_norm.astype(np.float32), combined.astype(np.float32)

def analyze_track(path: Path) -> TrackAnalysis:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    y_raw, sr = librosa.load(str(path), sr=None, mono=False)
    if y_raw.size == 0:
        raise ValueError(f"Empty audio file: {path}")

    duration = y_raw.shape[-1] / sr
    stereo_audit, y_analysis_mono = evaluate_stereo_field(y_raw, sr)

    sample_peak_lin, true_peak_lin = compute_peak_and_true_peak(y_raw, sr)
    sample_peak_dbfs = 20 * math.log10(max(sample_peak_lin, EPS))
    true_peak_dbtp = 20 * math.log10(max(true_peak_lin, EPS))

    rms_linear = float(np.sqrt(np.mean(y_raw.astype(np.float64) ** 2)))
    crest_factor = 20 * math.log10(max(true_peak_lin / max(rms_linear, EPS), EPS))

    # pyloudnorm channel configuration
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

# ============================================================
# TEMPO-SCALED SNIPPET DETECTOR
# ============================================================

def resample_feature_matrix(features: np.ndarray, ratio: float) -> np.ndarray:
    if features.size == 0 or abs(ratio - 1.0) < 0.005:
        return features

    n_feats, n_frames = features.shape
    new_n_frames = max(1, int(round(n_frames * ratio)))
    orig_times = np.linspace(0.0, 1.0, n_frames)
    new_times = np.linspace(0.0, 1.0, new_n_frames)

    resampled = np.empty((n_feats, new_n_frames), dtype=np.float32)
    for i in range(n_feats):
        resampled[i, :] = np.interp(new_times, orig_times, features[i, :])

    return resampled / np.maximum(np.linalg.norm(resampled, axis=0, keepdims=True), EPS)

def find_snippets(
    a: TrackAnalysis,
    b: TrackAnalysis,
    min_duration: float = 3.0,
    sim_threshold: float = 0.72,
    auto_tempo_scale: bool = True
) -> List[MatchedSnippet]:
    if a.combined_features.size == 0 or b.combined_features.size == 0:
        return []

    hop_sec = HOP_LENGTH / ANALYSIS_SR
    min_frames = max(1, int(round(min_duration / hop_sec)))
    smooth_frames = max(1, min(int(round(1.0 / hop_sec)), min_frames))

    speed_ratio = 1.0
    if auto_tempo_scale and a.bpm > 30.0 and b.bpm > 30.0:
        raw_ratio = b.bpm / a.bpm
        for target in [1.0, 0.5, 2.0]:
            if abs(raw_ratio - target) < 0.08:
                speed_ratio = raw_ratio
                break

    b_features = resample_feature_matrix(b.combined_features, 1.0 / speed_ratio) if speed_ratio != 1.0 else b.combined_features

    S = np.clip(a.combined_features.T @ b_features, 0.0, 1.0)
    n_a, n_b = S.shape

    candidates: List[MatchedSnippet] = []

    for d in range(-(n_a - 1), n_b):
        diag = np.diagonal(S, offset=d)
        if len(diag) < min_frames or np.max(diag) < sim_threshold:
            continue

        smooth_diag = uniform_filter1d(diag, size=smooth_frames, mode="nearest")
        matched_mask = smooth_diag >= sim_threshold

        padded = np.r_[False, matched_mask, False]
        diff = np.diff(padded.astype(np.int8))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        for s, e in zip(starts, ends):
            run_len = e - s
            if run_len >= min_frames:
                score = float(np.mean(diag[s:e]))
                if score >= sim_threshold:
                    if d >= 0:
                        ia_start, ia_end = s, e
                        ib_start, ib_end = s + d, e + d
                    else:
                        ia_start, ia_end = -d + s, -d + e
                        ib_start, ib_end = s, e

                    a_start = min(a.duration, ia_start * hop_sec)
                    a_end = min(a.duration, ia_end * hop_sec)
                    b_start = min(b.duration, ib_start * hop_sec * speed_ratio)
                    b_end = min(b.duration, ib_end * hop_sec * speed_ratio)

                    dur_a = a_end - a_start
                    dur_b = b_end - b_start

                    if min(dur_a, dur_b) >= max(0.5, min_duration - hop_sec):
                        # Measure independent harmonic and timbral similarities
                        frames_a = slice(ia_start, min(a.chroma_features.shape[1], ia_end))
                        frames_b = slice(
                            int(round(ib_start * speed_ratio)),
                            min(b.chroma_features.shape[1], int(round(ib_end * speed_ratio)))
                        )

                        eval_len = min(
                            frames_a.stop - frames_a.start,
                            frames_b.stop - frames_b.start
                        )

                        if eval_len > 0:
                            ca = a.chroma_features[:, frames_a.start : frames_a.start + eval_len]
                            cb = b.chroma_features[:, frames_b.start : frames_b.start + eval_len]
                            harm_score = float(np.mean(np.sum(ca * cb, axis=0)))

                            ta = a.timbral_features[:, frames_a.start : frames_a.start + eval_len]
                            tb = b.timbral_features[:, frames_b.start : frames_b.start + eval_len]
                            timb_score = float(np.mean(np.sum(ta * tb, axis=0)))
                        else:
                            harm_score = score
                            timb_score = score

                        candidates.append(MatchedSnippet(
                            a_start=a_start,
                            a_end=a_end,
                            b_start=b_start,
                            b_end=b_end,
                            duration_a=dur_a,
                            duration_b=dur_b,
                            speed_ratio=speed_ratio,
                            composite_similarity=score,
                            harmonic_similarity=float(np.clip(harm_score, 0.0, 1.0)),
                            timbral_similarity=float(np.clip(timb_score, 0.0, 1.0))
                        ))

    # Slope-aware deduplication using alignment offset c = t_B - r * t_A
    candidates.sort(key=lambda x: (x.composite_similarity * math.sqrt(x.duration_a)), reverse=True)
    filtered: List[MatchedSnippet] = []

    for cand in candidates:
        overlap = False
        cand_c = cand.b_start - cand.speed_ratio * cand.a_start
        for kept in filtered:
            kept_c = kept.b_start - kept.speed_ratio * kept.a_start
            if abs(cand_c - kept_c) <= max(2 * hop_sec, 0.35):
                ov_a = max(0.0, min(cand.a_end, kept.a_end) - max(cand.a_start, kept.a_start))
                if ov_a / (min(cand.duration_a, kept.duration_a) + EPS) > 0.40:
                    overlap = True
                    break
            else:
                ov_a = max(0.0, min(cand.a_end, kept.a_end) - max(cand.a_start, kept.a_start))
                ov_b = max(0.0, min(cand.b_end, kept.b_end) - max(cand.b_start, kept.b_start))
                if (ov_a / (cand.duration_a + EPS) > 0.8) and (ov_b / (cand.duration_b + EPS) > 0.8):
                    overlap = True
                    break
        if not overlap:
            filtered.append(cand)

    filtered.sort(key=lambda x: x.a_start)
    return filtered

# ============================================================
# BIPARTITE COVERAGE ALLOCATOR
# ============================================================

def compute_verified_coverage(
    a_dur: float,
    b_dur: float,
    matches: List[MatchedSnippet]
) -> Tuple[float, float, float, float, List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    Allocates structural coverage through analytic interval subtraction.
    Guarantees that mapped intervals in A and B are strictly mutual, non-overlapping,
    and verified without interpolating across gaps.
    """
    if not matches or a_dur <= 0 or b_dur <= 0:
        return 0.0, 0.0, 0.0, 0.0, [(0.0, a_dur)], [(0.0, b_dur)]

    claimed_a: List[Tuple[float, float]] = []
    claimed_b: List[Tuple[float, float]] = []

    # Sort candidates by structural significance
    sorted_matches = sorted(matches, key=lambda x: (x.composite_similarity * x.duration_a), reverse=True)

    for m in sorted_matches:
        r = m.speed_ratio
        c = m.b_start - r * m.a_start

        # Subdivide candidate interval against previously claimed intervals in Track A
        unclaimed_a = subtract_intervals((m.a_start, m.a_end), claimed_a)

        for ua_s, ua_e in unclaimed_a:
            # Map candidate sub-interval to Track B coordinates
            ub_s = r * ua_s + c
            ub_e = r * ua_e + c

            # Subdivide mapped interval against previously claimed intervals in Track B
            valid_b = subtract_intervals((ub_s, ub_e), claimed_b)

            for vb_s, vb_e in valid_b:
                # Map back to Track A coordinates
                va_s = (vb_s - c) / r
                va_e = (vb_e - c) / r

                if (va_e - va_s) >= 0.5:
                    claimed_a.append((va_s, va_e))
                    claimed_b.append((vb_s, vb_e))

                    # Consolidate internal interval lists
                    claimed_a = sorted(claimed_a, key=lambda x: x[0])
                    merged_a = [claimed_a[0]]
                    for s, e in claimed_a[1:]:
                        if s <= merged_a[-1][1]:
                            merged_a[-1] = (merged_a[-1][0], max(merged_a[-1][1], e))
                        else:
                            merged_a.append((s, e))
                    claimed_a = merged_a

                    claimed_b = sorted(claimed_b, key=lambda x: x[0])
                    merged_b = [claimed_b[0]]
                    for s, e in claimed_b[1:]:
                        if s <= merged_b[-1][1]:
                            merged_b[-1] = (merged_b[-1][0], max(merged_b[-1][1], e))
                        else:
                            merged_b.append((s, e))
                    claimed_b = merged_b

    covered_sec_a = sum(e - s for s, e in claimed_a)
    covered_sec_b = sum(e - s for s, e in claimed_b)

    pct_a = min(100.0, (covered_sec_a / max(a_dur, EPS)) * 100.0)
    pct_b = min(100.0, (covered_sec_b / max(b_dur, EPS)) * 100.0)

    gaps_a = [g for g in subtract_intervals((0.0, a_dur), claimed_a) if (g[1] - g[0]) >= 1.5]
    gaps_b = [g for g in subtract_intervals((0.0, b_dur), claimed_b) if (g[1] - g[0]) >= 1.5]

    return covered_sec_a, covered_sec_b, pct_a, pct_b, gaps_a, gaps_b

# ============================================================
# AUDIT REPORTING
# ============================================================

def print_track_card(label: str, t: TrackAnalysis):
    print(f"\n[{label}] {t.path.name}")
    print(f"  Duration:         {fmt_time(t.duration)} ({t.duration:.2f} s)")
    print(f"  Channel Format:   {t.stereo.channel_layout_desc}")
    print(f"  Estimated BPM:    {t.bpm:.1f}")
    print(f"  Estimated Key:    {t.key}")
    lufs_str = f"{t.lufs:.1f} LUFS" if t.lufs is not None else "N/A"
    lra_str = f"{t.lra:.1f} LU" if t.lra is not None else "N/A"
    print(f"  Loudness (LUFS):  {lufs_str:<10} | LRA: {lra_str}")
    print(f"  Sample Peak:      {t.sample_peak_dbfs:.2f} dBFS   | True Peak: {t.true_peak_dbtp:.2f} dBTP")
    print(f"  Dynamic Punch:    {t.crest_factor_db:.2f} dB Crest Factor (True Peak / RMS)")

    st = t.stereo
    ms_str = "Mono" if st.mid_side_ratio_db <= -90.0 else f"{st.mid_side_ratio_db:.1f} dB"
    print(f"  Stereo Field:     Global Corr: {st.global_correlation:+.2f} | M/S Energy Ratio: {ms_str}")
    print(f"  Phase Stability:  Min Window:  {st.min_window_correlation:+.2f} | Negative Windows: {st.pct_negative_windows:.1f}%")
    if st.mono_loss_db < -0.2:
        print(f"  Mono Fold-Down:   {st.mono_loss_db:.2f} dB Power Loss upon mono sum")
    if st.mono_fallback_used:
        print("  [NOTICE] Severe out-of-phase mono cancellation; Left channel was used for feature extraction.")

def print_mix_deltas(a: TrackAnalysis, b: TrackAnalysis):
    print("\n" + "=" * 65)
    print("MIX AND MASTERING AUDIT (Song B vs. Song A)")
    print("=" * 65)

    if a.lufs is not None and b.lufs is not None:
        delta_lufs = b.lufs - a.lufs
        indicator = "Louder" if delta_lufs > 0.2 else "Quieter" if delta_lufs < -0.2 else "Matched loudness"
        print(f"  Loudness:         Song B is {delta_lufs:+.1f} LU ({indicator})")
    else:
        print("  Loudness:         N/A (LUFS not available)")

    if a.lra is not None and b.lra is not None:
        delta_lra = b.lra - a.lra
        indicator = "More dynamic variation" if delta_lra > 0.5 else "More compressed" if delta_lra < -0.5 else "Consistent range"
        print(f"  Loudness Range:   Song B is {delta_lra:+.1f} LU ({indicator})")

    delta_crest = b.crest_factor_db - a.crest_factor_db
    crest_indicator = "More dynamic punch" if delta_crest > 0.5 else "More limited/compressed" if delta_crest < -0.5 else "Preserved dynamics"
    print(f"  Dynamic Punch:    {delta_crest:+.1f} dB Crest Factor ({crest_indicator})")

    delta_sample = b.sample_peak_dbfs - a.sample_peak_dbfs
    delta_true = b.true_peak_dbtp - a.true_peak_dbtp
    print(f"  Sample Peak:      Song B is {b.sample_peak_dbfs:.2f} dBFS (Delta: {delta_sample:+.2f} dBFS)")
    if b.sample_peak_dbfs > 0.0:
        print("  [ALERT] Decoded audio file contains floating-point samples exceeding 0.0 dBFS.")

    print(f"  True Peak:        Song B is {b.true_peak_dbtp:.2f} dBTP (Delta: {delta_true:+.2f} dBTP)")
    if b.true_peak_dbtp > 0.0 and b.sample_peak_dbfs <= 0.0:
        print("  [WARNING] Inter-sample true-peak overshoot detected (> 0.0 dBTP). Risk of DAC/lossy clipping.")
    elif b.true_peak_dbtp > -1.0 and b.true_peak_dbtp <= 0.0:
        print("  [NOTICE] True peak exceeds standard streaming ceiling (-1.0 dBTP target margin).")

    st_b = b.stereo
    print(f"\n  Stereo Correlation: {st_b.global_correlation:+.2f} (Worst Window: {st_b.min_window_correlation:+.2f})")
    if st_b.pct_negative_windows > 15.0:
        print(f"  [CRITICAL] Severe mono cancellation: {st_b.pct_negative_windows:.1f}% of active windows have negative correlation.")
    elif st_b.pct_negative_windows > 5.0:
        print(f"  [WARNING] Out-of-phase material: {st_b.pct_negative_windows:.1f}% negative correlation windows.")

    if st_b.mono_loss_db < -3.0:
        print(f"  [WARNING] Severe mono fold-down attenuation: {st_b.mono_loss_db:.1f} dB loss when collapsed to mono.")

    if a.stereo.mid_side_ratio_db > -90.0 and st_b.mid_side_ratio_db > -90.0:
        delta_ms = st_b.mid_side_ratio_db - a.stereo.mid_side_ratio_db
        width_indicator = "Wider" if delta_ms > 0.5 else "Narrower" if delta_ms < -0.5 else "Unchanged"
        print(f"  Stereo Width:     Side/Mid energy changed by {delta_ms:+.1f} dB ({width_indicator})")

    if a.spectral_balance_db and b.spectral_balance_db:
        print("\n  4-Band Spectral Balance Shift (Relative Share):")
        for band_name in a.spectral_balance_db:
            if band_name in b.spectral_balance_db:
                delta_p = b.spectral_balance_db[band_name] - a.spectral_balance_db[band_name]
                indicator = "Boosted" if delta_p > 0.5 else "Cut" if delta_p < -0.5 else "Even"
                print(f"    * {band_name:<24}: {delta_p:+.1f} dB [{indicator}]")

def print_snippet_report(a: TrackAnalysis, b: TrackAnalysis, matches: List[MatchedSnippet]):
    print("\n" + "=" * 65)
    print("SHARED SNIPPETS AND SAMPLES")
    print("=" * 65)

    if not matches:
        print("  No shared musical snippets detected above threshold.")
    else:
        print(f"{'Song A Location':<18}{'Song B Location':<18}{'Duration':<10}{'Composite':<11}{'Harmonic':<10}{'Timbral'}")
        print("-" * 75)
        for m in matches:
            loc_a = f"{fmt_time(m.a_start)} - {fmt_time(m.a_end)}"
            loc_b = f"{fmt_time(m.b_start)} - {fmt_time(m.b_end)}"
            dur = f"{m.duration_a:.1f} s"
            comp = f"{m.composite_similarity * 100:.1f}%"
            harm = f"{m.harmonic_similarity * 100:.1f}%"
            timb = f"{m.timbral_similarity * 100:.1f}%"
            print(f"{loc_a:<18}{loc_b:<18}{dur:<10}{comp:<11}{harm:<10}{timb}")

    sec_a, sec_b, pct_a, pct_b, gaps_a, gaps_b = compute_verified_coverage(a.duration, b.duration, matches)

    print("\n" + "-" * 65)
    print("STRUCTURAL EDIT AUDIT (Verified Bipartite Allocation)")
    print("-" * 65)
    print(f"  Song A verified shared duration:   {sec_a:.1f} s ({pct_a:.1f}%)")
    print(f"  Song B verified shared duration:   {sec_b:.1f} s ({pct_b:.1f}%)")

    if gaps_a and (not matches or pct_a < 99.0):
        print("\n  [CUT] Unique to Song A (Cut or missing in Song B):")
        for s, e in gaps_a:
            print(f"     * {fmt_time(s)} - {fmt_time(e)} ({e - s:.1f} s)")

    if gaps_b and (not matches or pct_b < 99.0):
        print("\n  [ADDED] Unique to Song B (New intro/outro, extended cuts, or distinct material):")
        for s, e in gaps_b:
            print(f"     * {fmt_time(s)} - {fmt_time(e)} ({e - s:.1f} s)")

# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Audio Mix, Edit and Structural Comparison Tool")
    parser.add_argument("song_a", type=Path, help="Reference track (e.g. Mix v1, Original, Radio Edit)")
    parser.add_argument("song_b", type=Path, help="Target track (e.g. Master, Extended Mix, Club VIP)")
    parser.add_argument("--min-len", type=float, default=3.0, help="Minimum snippet length in seconds (default: 3.0)")
    parser.add_argument("--threshold", type=float, default=0.72, help="Similarity threshold 0.0-1.0 (default: 0.72)")
    parser.add_argument("--no-tempo-rescale", action="store_true", help="Disable automatic tempo ratio scaling for snippet matching")

    args = parser.parse_args()

    if args.min_len <= 0.0:
        parser.error("--min-len must be greater than 0")
    if not (0.0 <= args.threshold <= 1.0):
        parser.error("--threshold must be between 0.0 and 1.0")

    try:
        print("Analyzing audio files...")
        track_a = analyze_track(args.song_a)
        track_b = analyze_track(args.song_b)

        print_track_card("SONG A", track_a)
        print_track_card("SONG B", track_b)

        print_mix_deltas(track_a, track_b)

        snippets = find_snippets(
            track_a,
            track_b,
            min_duration=args.min_len,
            sim_threshold=args.threshold,
            auto_tempo_scale=(not args.no_tempo_rescale)
        )
        print_snippet_report(track_a, track_b, snippets)

    except Exception as err:
        print(f"\nExecution error: {err}", file=sys.stderr)
        return 1

    return 0

if __name__ == "__main__":
    raise SystemExit(main())