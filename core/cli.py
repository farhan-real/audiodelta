"""
Command line interface argument parsing and execution flow for audiodelta.
"""

import argparse
from pathlib import Path
import sys
from typing import List, Optional

from core.features import analyze_track
from core.report import print_full_report
from core.snippets import find_snippets


def build_parser() -> argparse.ArgumentParser:
    """Constructs the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="audiodelta",
        description="audiodelta: Audio Mix, Edit and Structural Comparison Tool"
    )
    parser.add_argument("song_a", type=Path, help="Reference track (e.g. Mix v1, Original, Radio Edit)")
    parser.add_argument("song_b", type=Path, help="Target track (e.g. Master, Extended Mix, Club VIP)")
    parser.add_argument(
        "--mode", "-m",
        choices=["simple", "precise"],
        default="precise",
        help="Output wording mode: 'simple' (concise, direct terms, no explanations) or 'precise' (detailed audio engineering & mastering terminology, default: precise)"
    )
    parser.add_argument("--min-len", type=float, default=3.0, help="Minimum snippet length in seconds (default: 3.0)")
    parser.add_argument("--threshold", type=float, default=0.72, help="Similarity threshold 0.0-1.0 (default: 0.72)")
    parser.add_argument("--no-tempo-rescale", action="store_true", help="Disable automatic tempo ratio scaling for snippet matching")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars and status updates")
    return parser


def run(argv: Optional[List[str]] = None) -> int:
    """Executes the audiodelta CLI command pipeline with progress reporting."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.min_len <= 0.0:
        sys.stderr.write("Execution error: --min-len must be greater than 0\n")
        return 1
    if not (0.0 <= args.threshold <= 1.0):
        sys.stderr.write("Execution error: --threshold must be between 0.0 and 1.0\n")
        return 1

    show_progress = not args.no_progress

    try:
        if show_progress:
            print(f"audiodelta analyzing tracks [{args.mode} mode]...")

        track_a = analyze_track(args.song_a, show_progress=show_progress)
        track_b = analyze_track(args.song_b, show_progress=show_progress)

        snippets = find_snippets(
            track_a,
            track_b,
            min_duration=args.min_len,
            sim_threshold=args.threshold,
            auto_tempo_scale=(not args.no_tempo_rescale),
            show_progress=show_progress
        )

        print_full_report(track_a, track_b, snippets, mode=args.mode)

    except Exception as err:
        sys.stderr.write(f"\nExecution error: {err}\n")
        return 1

    return 0


def main() -> None:
    """Entry point for command-line execution."""
    sys.exit(run())