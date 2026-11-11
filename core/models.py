"""
Data structures representing audio analysis results and comparison records.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import numpy as np


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