"""
Unit tests for data structures and model representations.
"""

from pathlib import Path
import numpy as np

from core.models import MatchedSnippet, StereoAudit, TrackAnalysis


def test_matched_snippet_initialization():
    snippet = MatchedSnippet(
        a_start=10.0,
        a_end=20.0,
        b_start=15.0,
        b_end=25.0,
        duration_a=10.0,
        duration_b=10.0,
        speed_ratio=1.0,
        composite_similarity=0.88,
        harmonic_similarity=0.91,
        timbral_similarity=0.82
    )
    assert snippet.a_start == 10.0
    assert snippet.a_end == 20.0
    assert snippet.b_start == 15.0
    assert snippet.b_end == 25.0
    assert snippet.duration_a == 10.0
    assert snippet.duration_b == 10.0
    assert snippet.speed_ratio == 1.0
    assert snippet.composite_similarity == 0.88
    assert snippet.harmonic_similarity == 0.91
    assert snippet.timbral_similarity == 0.82


def test_stereo_audit_initialization():
    audit = StereoAudit(
        channel_count=2,
        channel_layout_desc="Stereo (2 ch)",
        global_correlation=0.95,
        min_window_correlation=0.72,
        pct_negative_windows=0.0,
        mono_loss_db=-0.1,
        mid_side_ratio_db=-12.5,
        mono_fallback_used=False
    )
    assert audit.channel_count == 2
    assert audit.global_correlation == 0.95
    assert not audit.mono_fallback_used


def test_track_analysis_initialization():
    audit = StereoAudit(
        channel_count=2,
        channel_layout_desc="Stereo (2 ch)",
        global_correlation=1.0,
        min_window_correlation=1.0,
        pct_negative_windows=0.0,
        mono_loss_db=0.0,
        mid_side_ratio_db=-99.9,
        mono_fallback_used=False
    )
    chroma = np.zeros((12, 5), dtype=np.float32)
    timbre = np.zeros((13, 5), dtype=np.float32)
    comb = np.zeros((25, 5), dtype=np.float32)

    analysis = TrackAnalysis(
        path=Path("test.wav"),
        duration=10.0,
        sample_rate=44100,
        bpm=120.0,
        key="C maj",
        lufs=-14.0,
        lra=6.0,
        sample_peak_dbfs=-0.5,
        true_peak_dbtp=-0.2,
        crest_factor_db=12.0,
        stereo=audit,
        spectral_balance_db={"Sub/Bass (20-250 Hz)": -6.0},
        chroma_features=chroma,
        timbral_features=timbre,
        combined_features=comb
    )

    assert analysis.duration == 10.0
    assert analysis.bpm == 120.0
    assert analysis.key == "C maj"
    assert analysis.chroma_features.shape == (12, 5)
    assert analysis.timbral_features.shape == (13, 5)
    assert analysis.combined_features.shape == (25, 5)