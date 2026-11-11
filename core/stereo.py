"""
Stereo phase correlation, Mid/Side ratio, and mono fold-down compatibility audit.
"""

import math
from typing import Tuple
import numpy as np

from core.constants import EPS
from core.models import StereoAudit


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