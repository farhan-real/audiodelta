"""
Unit tests for audiodelta dual-mode report generation, color formatting, and overhauled wording.
"""

from pathlib import Path
import numpy as np

from core.models import MatchedSnippet, StereoAudit, TrackAnalysis
from core.report import print_mix_deltas, print_snippet_report, print_track_card


def build_sample_analysis(name: str = "track.wav", lufs: float = -14.0) -> TrackAnalysis:
    stereo = StereoAudit(
        channel_count=2,
        channel_layout_desc="Stereo (2 ch)",
        global_correlation=0.92,
        min_window_correlation=0.65,
        pct_negative_windows=2.5,
        mono_loss_db=-0.45,
        mid_side_ratio_db=-14.2,
        mono_fallback_used=False
    )
    spectral = {
        "Sub/Bass (20-250 Hz)": -6.2,
        "Low-Mid (250-1000 Hz)": -5.1,
        "High-Mid (1-5 kHz)": -8.4,
        "Air (5-20 kHz)": -12.0
    }
    return TrackAnalysis(
        path=Path(name),
        duration=120.0,
        sample_rate=44100,
        bpm=124.0,
        key="C maj",
        lufs=lufs,
        lra=6.4,
        sample_peak_dbfs=-0.52,
        true_peak_dbtp=-0.18,
        crest_factor_db=12.4,
        stereo=stereo,
        spectral_balance_db=spectral,
        chroma_features=np.zeros((12, 10), dtype=np.float32),
        timbral_features=np.zeros((13, 10), dtype=np.float32),
        combined_features=np.zeros((25, 10), dtype=np.float32)
    )


def test_track_card_simple_mode(capsys):
    track = build_sample_analysis("simple_track.wav")
    print_track_card("SONG A", track, mode="simple", use_color=False)
    out = capsys.readouterr().out

    assert "simple_track.wav" in out
    assert "Length:" in out
    assert "02:00.00" in out
    assert "Channels:" in out
    assert "Tempo:" in out
    assert "124.0 BPM" in out
    assert "Key:" in out
    assert "C Major" in out
    assert "Loudness:" in out
    assert "-14.0 LUFS" in out
    assert "Dynamic Range:" in out
    assert "6.4 LU" in out
    assert "Punch:" in out
    assert "12.4 dB crest" in out
    assert "Peak Level:" in out
    assert "Stereo Field:" in out
    assert "Phase Health:" in out
    assert "Mono Drop:" in out
    assert "Side vs Center:" in out
    assert "Frequency Energy:" in out
    assert "Bass" in out

    assert "True Peak / RMS" not in out
    assert "ITU-R" not in out


def test_track_card_color_formatting(capsys):
    track = build_sample_analysis("color_track.wav")
    print_track_card("SONG A", track, mode="precise", use_color=True)
    out = capsys.readouterr().out
    assert "\033[" in out


def test_mix_deltas_simple_mode_avoids_explanations(capsys):
    t_a = build_sample_analysis("mix.wav", lufs=-16.0)
    t_b = build_sample_analysis("master.wav", lufs=-13.5)
    t_b.true_peak_dbtp = 0.45
    t_b.sample_peak_dbfs = 0.0

    print_mix_deltas(t_a, t_b, mode="simple", use_color=False)
    out = capsys.readouterr().out

    assert "Loudness:" in out
    assert "+2.5 LU" in out
    assert "Dynamic Range:" in out
    assert "Punch:" in out
    assert "Peak Level:" in out
    assert "True Peak:" in out
    assert "Stereo Field:" in out
    assert "Phase Health:" in out
    assert "Mono Drop:" in out
    assert "Stereo Image:" in out
    assert "Frequency Shifts:" in out

    assert "[WARNING]" not in out
    assert "Risk of DAC/lossy clipping" not in out
    assert "[CRITICAL]" not in out


def test_mix_deltas_precise_mode_includes_technical_alerts(capsys):
    t_a = build_sample_analysis("mix.wav", lufs=-16.0)
    t_b = build_sample_analysis("master.wav", lufs=-13.5)
    t_b.true_peak_dbtp = 0.45
    t_b.sample_peak_dbfs = 0.0

    print_mix_deltas(t_a, t_b, mode="precise", use_color=True)
    out = capsys.readouterr().out

    assert "[WARNING] Inter-sample true-peak overshoot detected" in out
    assert "MIX AND MASTERING AUDIT" in out
    assert "\033[" in out


def test_snippet_report_overhauled_wording(capsys):
    t_a = build_sample_analysis("a.wav")
    t_b = build_sample_analysis("b.wav")
    snippet = MatchedSnippet(
        a_start=10.0,
        a_end=40.0,
        b_start=20.0,
        b_end=50.0,
        duration_a=30.0,
        duration_b=30.0,
        speed_ratio=1.0,
        composite_similarity=0.92,
        harmonic_similarity=0.95,
        timbral_similarity=0.88
    )

    # Simple mode with overhauled wording
    print_snippet_report(t_a, t_b, [snippet], mode="simple", use_color=False)
    out_simple = capsys.readouterr().out
    assert "SHARED SECTIONS" in out_simple
    assert "ARRANGEMENT BREAKDOWN" in out_simple
    assert "Song A in Song B:  30.0 s" in out_simple
    assert "Song B in Song A:  30.0 s" in out_simple
    assert "Cut from Song A (missing in Song B):" in out_simple
    assert "New in Song B (added / extended):" in out_simple
    assert "Bipartite Allocation" not in out_simple

    # Precise mode
    print_snippet_report(t_a, t_b, [snippet], mode="precise", use_color=False)
    out_precise = capsys.readouterr().out
    assert "SHARED SNIPPETS AND SAMPLES" in out_precise
    assert "STRUCTURAL EDIT AUDIT (Verified Bipartite Allocation)" in out_precise
    assert "Song A in Song B (shared duration):" in out_precise
    assert "Song B in Song A (shared duration):" in out_precise
    assert "[CUT]" in out_precise
    assert "[ADDED]" in out_precise


def test_snippet_report_collapsible_behavior(capsys):
    t_a = build_sample_analysis("a.wav")
    t_b = build_sample_analysis("b.wav")

    snippets = [
        MatchedSnippet(float(i * 10), float(i * 10 + 4), float(i * 10), float(i * 10 + 4), 4.0, 4.0, 1.0, 0.90, 0.90, 0.90)
        for i in range(15)
    ]

    print_snippet_report(t_a, t_b, snippets, mode="simple", use_color=False, max_snippets=10, all_snippets=False)
    out_default = capsys.readouterr().out
    assert "[+] Showing 10 of 15 sections (5 additional sections collapsed)." in out_default
    assert "Run with '--all-snippets' to display all" in out_default

    print_snippet_report(t_a, t_b, snippets, mode="simple", use_color=False, max_snippets=10, all_snippets=True)
    out_all = capsys.readouterr().out
    assert "additional sections collapsed" not in out_all

    print_snippet_report(t_a, t_b, snippets, mode="simple", use_color=False, collapse_snippets=True)
    out_collapsed = capsys.readouterr().out
    assert "[v] Snippet table collapsed (15 sections detected)." in out_collapsed