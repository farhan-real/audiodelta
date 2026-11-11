"""
Unit tests for feature resampling, diagonal cross-correlation, and snippet consolidation.
"""

from pathlib import Path
import numpy as np
import pytest

from core.models import MatchedSnippet, StereoAudit, TrackAnalysis
from core.snippets import consolidate_snippets, find_snippets, resample_feature_matrix


def test_resample_feature_matrix_empty():
    feats = np.empty((25, 0), dtype=np.float32)
    res = resample_feature_matrix(feats, 1.5)
    assert res.size == 0


def test_resample_feature_matrix_identity():
    feats = np.ones((25, 20), dtype=np.float32)
    feats /= np.linalg.norm(feats, axis=0, keepdims=True)
    res = resample_feature_matrix(feats, 1.0)
    assert res.shape == feats.shape
    assert np.allclose(res, feats)


def test_resample_feature_matrix_ratios():
    feats = np.random.RandomState(42).randn(25, 100).astype(np.float32)
    feats /= np.linalg.norm(feats, axis=0, keepdims=True)

    res_2x = resample_feature_matrix(feats, 2.0)
    assert res_2x.shape == (25, 200)
    assert np.allclose(np.linalg.norm(res_2x, axis=0), 1.0, atol=1e-3)

    res_half = resample_feature_matrix(feats, 0.5)
    assert res_half.shape == (25, 50)
    assert np.allclose(np.linalg.norm(res_half, axis=0), 1.0, atol=1e-3)


def test_consolidate_snippets_merges_contiguous():
    s1 = MatchedSnippet(0.0, 10.0, 5.0, 15.0, 10.0, 10.0, 1.0, 0.90, 0.92, 0.88)
    s2 = MatchedSnippet(10.0, 20.0, 15.0, 25.0, 10.0, 10.0, 1.0, 0.80, 0.82, 0.78)
    s3 = MatchedSnippet(35.0, 45.0, 40.0, 50.0, 10.0, 10.0, 1.0, 0.95, 0.95, 0.95)

    merged = consolidate_snippets([s1, s2, s3], tolerance=0.5)
    # s1 and s2 should be merged into one 20s block; s3 remains separate
    assert len(merged) == 2
    assert merged[0].a_start == 0.0
    assert merged[0].a_end == 20.0
    assert merged[0].b_start == 5.0
    assert merged[0].b_end == 25.0
    assert merged[0].duration_a == 20.0
    assert pytest.approx(merged[0].composite_similarity, abs=1e-3) == 0.85
    assert merged[1].a_start == 35.0


def test_consolidate_snippets_single_or_empty():
    assert consolidate_snippets([]) == []
    s = MatchedSnippet(0.0, 5.0, 0.0, 5.0, 5.0, 5.0, 1.0, 0.9, 0.9, 0.9)
    assert consolidate_snippets([s]) == [s]


def build_mock_track(duration: float, features: np.ndarray, bpm: float = 120.0) -> TrackAnalysis:
    stereo = StereoAudit(2, "Stereo (2 ch)", 1.0, 1.0, 0.0, 0.0, -99.9, False)
    chroma = features[:12, :]
    timbre = features[12:25, :]
    return TrackAnalysis(
        path=Path("mock.wav"),
        duration=duration,
        sample_rate=22050,
        bpm=bpm,
        key="C maj",
        lufs=-14.0,
        lra=6.0,
        sample_peak_dbfs=-0.5,
        true_peak_dbtp=-0.2,
        crest_factor_db=12.0,
        stereo=stereo,
        spectral_balance_db={},
        chroma_features=chroma,
        timbral_features=timbre,
        combined_features=features
    )


def test_find_snippets_empty():
    t1 = build_mock_track(10.0, np.empty((25, 0), dtype=np.float32))
    t2 = build_mock_track(10.0, np.empty((25, 0), dtype=np.float32))
    assert find_snippets(t1, t2) == []


def test_find_snippets_identical_tracks():
    np.random.seed(42)
    frames = 150
    features = np.random.randn(25, frames).astype(np.float32)
    features /= np.linalg.norm(features, axis=0, keepdims=True)

    t1 = build_mock_track(14.0, features, bpm=120.0)
    t2 = build_mock_track(14.0, features, bpm=120.0)

    matches_prog = find_snippets(t1, t2, min_duration=3.0, sim_threshold=0.72, show_progress=True)
    matches_quiet = find_snippets(t1, t2, min_duration=3.0, sim_threshold=0.72, show_progress=False)

    assert len(matches_prog) == len(matches_quiet) >= 1
    best_match = matches_prog[0]
    assert pytest.approx(best_match.composite_similarity, abs=0.05) == 1.0
    assert best_match.duration_a >= 10.0
    assert pytest.approx(best_match.speed_ratio, abs=1e-3) == 1.0


def test_find_snippets_shifted_offset():
    np.random.seed(123)
    core = np.random.randn(25, 100).astype(np.float32)
    prep = np.random.randn(25, 30).astype(np.float32)

    feat_a = core / np.linalg.norm(core, axis=0, keepdims=True)
    feat_b = np.hstack([prep, core])
    feat_b /= np.linalg.norm(feat_b, axis=0, keepdims=True)

    hop_sec = 2048 / 22050
    t1 = build_mock_track(100 * hop_sec, feat_a, bpm=120.0)
    t2 = build_mock_track(130 * hop_sec, feat_b, bpm=120.0)

    matches = find_snippets(t1, t2, min_duration=3.0, sim_threshold=0.72, show_progress=False)
    assert len(matches) >= 1
    assert matches[0].b_start > matches[0].a_start
    assert pytest.approx(matches[0].b_start - matches[0].a_start, abs=0.3) == 30 * hop_sec