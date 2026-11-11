"""
Verified Sample Peak and True Peak calculation engine compliant with ITU-R BS.1770-4.
"""

from typing import Callable, Optional, Tuple
import numpy as np
import scipy.signal

from core.constants import BS1770_FILTER_48


def compute_peak_and_true_peak(
    y: np.ndarray,
    sr: int,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> Tuple[float, float]:
    """
    Computes Sample Peak (dBFS linear) and True Peak (dBTP linear).
    Uses ITU-R BS.1770-4 48-tap 4x polyphase FIR filtering with exact group-delay
    alignment (tau = 23.5 samples at 4x rate). For sr > 96k, applies 2x oversampling.
    Supports optional progress callbacks reporting (current_chunk, total_chunks).
    """
    if y.size == 0:
        return 0.0, 0.0

    y_2d = np.atleast_2d(y.astype(np.float64))
    sample_peak = float(np.max(np.abs(y_2d)))

    chunk_size = 262144
    pad_len = 32
    n_samples = y_2d.shape[-1]
    true_peak = sample_peak

    total_chunks = max(1, (n_samples + chunk_size - 1) // chunk_size)
    chunk_index = 0

    if sr <= 96000:
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

            core_start = pad_len * 4 + 22
            core_end = core_start + (end - start) * 4
            core = y_up[:, core_start:core_end]

            if core.size > 0:
                true_peak = max(true_peak, float(np.max(np.abs(core))))

            chunk_index += 1
            if progress_callback is not None:
                progress_callback(chunk_index, total_chunks)
    else:
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

            chunk_index += 1
            if progress_callback is not None:
                progress_callback(chunk_index, total_chunks)

    return sample_peak, true_peak