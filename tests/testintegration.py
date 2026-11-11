"""
Integration tests executing full track comparison, console report generation, and CLI commands.
"""

from pathlib import Path
import tempfile
import numpy as np
import pytest
import soundfile as sf

from core.cli import build_parser, run
from core.features import analyze_track
from core.report import print_full_report
from core.snippets import find_snippets


def create_synthetic_wav(path: Path, duration: float, freq: float, sr: int = 22050):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    ch1 = 0.4 * np.sin(2 * np.pi * freq * t)
    ch2 = 0.4 * np.sin(2 * np.pi * freq * 1.5 * t)
    stereo = np.vstack([ch1, ch2]).T.astype(np.float32)
    sf.write(str(path), stereo, sr)


def test_full_analysis_and_reporting_pipeline_both_modes(capsys):
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        wav_a = tmp_path / "song_a.wav"
        wav_b = tmp_path / "song_b.wav"

        create_synthetic_wav(wav_a, duration=4.0, freq=440.0)
        create_synthetic_wav(wav_b, duration=4.0, freq=440.0)

        track_a = analyze_track(wav_a, show_progress=False)
        track_b = analyze_track(wav_b, show_progress=False)
        snippets = find_snippets(track_a, track_b, min_duration=2.0, sim_threshold=0.60, show_progress=False)

        print_full_report(track_a, track_b, snippets, mode="precise", use_color=False)
        out_precise = capsys.readouterr().out
        assert "[SONG A] song_a.wav" in out_precise
        assert "MIX AND MASTERING AUDIT" in out_precise
        assert "SHARED SNIPPETS AND SAMPLES" in out_precise
        assert "STRUCTURAL EDIT AUDIT" in out_precise
        assert "Song A in Song B (shared duration):" in out_precise

        print_full_report(track_a, track_b, snippets, mode="simple", use_color=False)
        out_simple = capsys.readouterr().out
        assert "[SONG A] song_a.wav" in out_simple
        assert "MIX COMPARISON" in out_simple
        assert "SHARED SECTIONS" in out_simple
        assert "ARRANGEMENT BREAKDOWN" in out_simple
        assert "Song A in Song B:" in out_simple


def test_cli_execution_flags_and_html_generation(capsys):
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        wav_a = tmp_path / "track1.wav"
        wav_b = tmp_path / "track2.wav"
        html_target = tmp_path / "output_report.html"

        create_synthetic_wav(wav_a, duration=3.5, freq=300.0)
        create_synthetic_wav(wav_b, duration=3.5, freq=300.0)

        code = run([
            str(wav_a),
            str(wav_b),
            "--no-color",
            "--no-progress",
            "--html",
            str(html_target),
            "--min-len",
            "2.0"
        ])
        assert code == 0
        out = capsys.readouterr().out
        assert "MIX AND MASTERING AUDIT" in out
        assert html_target.exists()
        assert "<details" in html_target.read_text(encoding="utf-8")


def test_cli_execution_invalid_arguments(capsys):
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        wav = tmp_path / "track.wav"
        create_synthetic_wav(wav, duration=1.0, freq=440.0)

        code_len = run([str(wav), str(wav), "--min-len", "-1.0"])
        assert code_len == 1
        assert "--min-len must be greater than 0" in capsys.readouterr().err

        code_th = run([str(wav), str(wav), "--threshold", "1.5"])
        assert code_th == 1
        assert "--threshold must be between 0.0 and 1.0" in capsys.readouterr().err


def test_cli_execution_missing_file(capsys):
    code = run(["missing1.wav", "missing2.wav"])
    assert code == 1
    assert "Execution error" in capsys.readouterr().err


def test_parser_defaults():
    parser = build_parser()
    args = parser.parse_args(["song_a.wav", "song_b.wav"])
    assert args.song_a == Path("song_a.wav")
    assert args.song_b == Path("song_b.wav")
    assert args.mode == "precise"
    assert args.min_len == 3.0
    assert args.threshold == 0.72
    assert not args.no_tempo_rescale
    assert not args.no_progress
    assert not args.no_color
    assert args.html is None
    assert not args.open_html