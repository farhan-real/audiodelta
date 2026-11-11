"""
audiodelta core package.
"""

from core.coverage import compute_verified_coverage
from core.features import (
    analyze_track,
    compute_spectral_balance,
    estimate_key,
    extract_features,
)
from core.models import MatchedSnippet, StereoAudit, TrackAnalysis
from core.peak import compute_peak_and_true_peak
from core.report import (
    generate_html_report,
    print_full_report,
    print_mix_deltas,
    print_snippet_report,
    print_track_card,
)
from core.snippets import (
    consolidate_snippets,
    find_snippets,
    resample_feature_matrix,
)
from core.stereo import evaluate_stereo_field
from core.utils import (
    Colors,
    ProgressBar,
    colorize,
    fmt_time,
    should_use_color,
    suppress_c_stderr,
    subtract_intervals,
)

__all__ = [
    "MatchedSnippet",
    "StereoAudit",
    "TrackAnalysis",
    "Colors",
    "ProgressBar",
    "colorize",
    "should_use_color",
    "suppress_c_stderr",
    "fmt_time",
    "subtract_intervals",
    "compute_peak_and_true_peak",
    "evaluate_stereo_field",
    "estimate_key",
    "compute_spectral_balance",
    "extract_features",
    "analyze_track",
    "resample_feature_matrix",
    "consolidate_snippets",
    "find_snippets",
    "compute_verified_coverage",
    "print_track_card",
    "print_mix_deltas",
    "print_snippet_report",
    "print_full_report",
    "generate_html_report",
]
