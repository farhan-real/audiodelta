"""
Cross-correlation and slope-aware matched snippet detection with tempo scaling.
"""

import math
from typing import List
import numpy as np
from scipy.ndimage import uniform_filter1d

from core.constants import ANALYSIS_SR, EPS, HOP_LENGTH
from core.models import MatchedSnippet, TrackAnalysis


def resample_feature_matrix(features: np.ndarray, ratio: float) -> np.ndarray:
    """Resamples feature matrix along the time axis according to ratio."""
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
    """Finds matching shared musical snippets between two analyzed tracks."""
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