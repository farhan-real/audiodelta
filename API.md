# Audiodelta Python API Reference

The `audiodelta` Python library provides modular functions and data structures for audio analysis, loudness metering, inter-sample true-peak estimation, multi-window stereo phase correlation, dual-domain snippet matching, bipartite interval allocation, and interactive reporting.

All primary symbols can be imported directly from the top-level `core` package:

```python
from core import (
    analyze_track,
    find_snippets,
    consolidate_snippets,
    compute_verified_coverage,
    print_full_report,
    generate_html_report,
    TrackAnalysis,
    StereoAudit,
    MatchedSnippet
)
```

---

## Table of Contents
1. [Data Models](#data-models)
2. [Audio Analysis Pipeline (`core.features`)](#audio-analysis-pipeline-corefeatures)
3. [Peak & True Peak Engine (`core.peak`)](#peak--true-peak-engine-corepeak)
4. [Stereo Field & Phase Audit (`core.stereo`)](#stereo-field--phase-audit-corestereo)
5. [Snippet Matching & Consolidation (`core.snippets`)](#snippet-matching--consolidation-coresnippets)
6. [Bipartite Interval Coverage Allocation (`core.coverage`)](#bipartite-interval-coverage-allocation-corecoverage)
7. [Reporting & HTML Export (`core.report`)](#reporting--html-export-corereport)
8. [Utilities & Progress Tracking (`core.utils`)](#utilities--progress-tracking-coreutils)

---

## Data Models

### `TrackAnalysis`
Represents the comprehensive analysis record for an individual audio file.

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import numpy as np
from core.models import StereoAudit

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
```

#### Fields
- `path` (*Path*): Absolute or relative filesystem path to the analyzed audio file.
- `duration` (*float*): Total length of the track in seconds.
- `sample_rate` (*int*): Native sampling frequency of the decoded audio file in Hertz (Hz).
- `bpm` (*float*): Estimated tempo in Beats Per Minute (BPM) derived via beat tracking.
- `key` (*str*): Estimated musical key and modality (e.g. `"C maj"`, `"D min"`, `"Undetermined"`).
- `lufs` (*Optional[float]*): Integrated loudness in Loudness Units Full Scale (`LUFS`) per ITU-R BS.1770-4. Returns `None` if audio length is insufficient for gating.
- `lra` (*Optional[float]*): Loudness Range in Loudness Units (`LU`) measuring dynamic spread.
- `sample_peak_dbfs` (*float*): Maximum absolute sample amplitude expressed in `dBFS`.
- `true_peak_dbtp` (*float*): Maximum reconstructed inter-sample true peak expressed in `dBTP`.
- `crest_factor_db` (*float*): Peak-to-RMS energy ratio in decibels ($20 \log_{10}(\text{TruePeak} / \text{RMS})$).
- `stereo` (*StereoAudit*): Stereo phase correlation and mono fold-down compatibility metrics.
- `spectral_balance_db` (*Dict[str, float]*): Welch PSD relative power distribution across 4 frequency bands.
- `chroma_features` (*np.ndarray*): 12-bin unit-normalized CQT chromagram matrix of shape `(12, frames)`.
- `timbral_features` (*np.ndarray*): 13-bin CMVN-normalized MFCC matrix of shape `(13, frames)`.
- `combined_features` (*np.ndarray*): Calibrated 70/30 composite feature matrix of shape `(25, frames)`.

---

### `StereoAudit`
Represents phase stability, spatial distribution, and mono compatibility metrics.

```python
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
```

#### Fields
- `channel_count` (*int*): Number of audio channels present (1 for mono, 2 for stereo, >2 for multichannel).
- `channel_layout_desc` (*str*): Human-readable channel configuration descriptor.
- `global_correlation` (*float*): Full-track Pearson correlation coefficient between Left and Right channels $[-1.0, +1.0]$.
- `min_window_correlation` (*float*): Minimum correlation recorded across any active $100\text{ ms}$ sliding window.
- `pct_negative_windows` (*float*): Percentage of active $100\text{ ms}$ windows with negative phase correlation ($\rho < 0.0$).
- `mono_loss_db` (*float*): Power attenuation in decibels when collapsing Left and Right to mono ($M = \frac{L+R}{2}$).
- `mid_side_ratio_db` (*float*): Side-to-Mid logarithmic energy ratio ($10 \log_{10}(E_{\text{side}} / E_{\text{mid}})$).
- `mono_fallback_used` (*bool*): True if severe out-of-phase cancellation triggered isolated channel fallback for feature analysis.

---

### `MatchedSnippet`
Represents an individual musical snippet or sample match between two tracks.

```python
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
```

#### Fields
- `a_start` (*float*): Start timestamp in Song A (in seconds).
- `a_end` (*float*): End timestamp in Song A (in seconds).
- `b_start` (*float*): Start timestamp in Song B (in seconds).
- `b_end` (*float*): End timestamp in Song B (in seconds).
- `duration_a` (*float*): Duration of the matched snippet in Song A ($a_{\text{end}} - a_{\text{start}}$).
- `duration_b` (*float*): Duration of the matched snippet in Song B ($b_{\text{end}} - b_{\text{start}}$).
- `speed_ratio` (*float*): Playback speed ratio between Song B and Song A ($r = \text{BPM}_B / \text{BPM}_A$).
- `composite_similarity` (*float*): Weighted cosine similarity across the combined 25-bin feature descriptor $[0.0, 1.0]$.
- `harmonic_similarity` (*float*): Pure chromagram cosine similarity $[0.0, 1.0]$.
- `timbral_similarity` (*float*): Pure CMVN MFCC cosine similarity $[0.0, 1.0]$.

---

## Audio Analysis Pipeline (`core.features`)

### `analyze_track()`
Decodes and analyzes an audio track from disk through the full 6-stage evaluation pipeline.

```python
def analyze_track(path: Path, show_progress: bool = True) -> TrackAnalysis
```

#### Parameters
- `path` (*Path*): Path to the target audio file (`.wav`, `.mp3`, `.flac`, `.ogg`, `.aiff`, etc.).
- `show_progress` (*bool*, optional): If `True`, writes terminal progress updates for each of the 6 pipeline stages. Defaults to `True`.

#### Returns
- `TrackAnalysis`: Fully populated audio analysis data structure.

#### Raises
- `FileNotFoundError`: If the file path does not exist.
- `ValueError`: If the decoded audio stream is empty.

---

### `extract_features()`
Extracts independent normalized harmonic (CQT chroma) and timbral (CMVN MFCC) descriptor matrices.

```python
def extract_features(y_mono: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
```

#### Parameters
- `y_mono` (*np.ndarray*): 1D single-channel audio sample array resampled to $22,050\text{ Hz}$.

#### Returns
- `Tuple[np.ndarray, np.ndarray, np.ndarray]`:
  1. `chroma_norm` (*np.ndarray*): Matrix of shape `(12, frames)` with unit-normalized columns.
  2. `mfcc_norm` (*np.ndarray*): Matrix of shape `(13, frames)` with CMVN-normalized and unit-norm columns.
  3. `combined` (*np.ndarray*): Matrix of shape `(25, frames)` containing the calibrated 70% chroma and 30% MFCC blend.

---

### `estimate_key()`
Estimates the musical key using Krumhansl-Schmuckler key profile correlation.

```python
def estimate_key(chroma: np.ndarray) -> str
```

#### Parameters
- `chroma` (*np.ndarray*): Chromagram feature matrix of shape `(12, frames)`.

#### Returns
- `str`: Estimated key (e.g. `"C maj"`, `"A min"`), or `"Undetermined"`.

---

### `compute_spectral_balance()`
Computes four-band relative spectral power distribution via multi-channel averaged Welch PSD.

```python
def compute_spectral_balance(y_raw: np.ndarray, sr: int) -> Dict[str, float]
```

#### Parameters
- `y_raw` (*np.ndarray*): Audio array of shape `(channels, samples)` or `(samples,)`.
- `sr` (*int*): Sampling rate in Hz.

#### Returns
- `Dict[str, float]`: Logarithmic relative energy shares (in dB) across:
  - `"Sub/Bass (20-250 Hz)"`
  - `"Low-Mid (250-1000 Hz)"`
  - `"High-Mid (1-5 kHz)"`
  - `"Air (5-20 kHz)"`

---

## Peak & True Peak Engine (`core.peak`)

### `compute_peak_and_true_peak()`
Calculates linear sample peak and ITU-R BS.1770-4 compliant polyphase oversampled true peak.

```python
def compute_peak_and_true_peak(
    y: np.ndarray,
    sr: int,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> Tuple[float, float]
```

#### Parameters
- `y` (*np.ndarray*): Raw audio data array.
- `sr` (*int*): Audio sampling rate in Hz.
- `progress_callback` (*Optional[Callable[[int, int], None]]*, optional): Optional callback invoked per chunk: `callback(current_chunk, total_chunks)`.

#### Returns
- `Tuple[float, float]`: `(sample_peak_linear, true_peak_linear)`.

---

## Stereo Field & Phase Audit (`core.stereo`)

### `evaluate_stereo_field()`
Evaluates Left/Right phase correlation, sliding window stability, and mono fold-down power loss.

```python
def evaluate_stereo_field(y_raw: np.ndarray, sr: int) -> Tuple[StereoAudit, np.ndarray]
```

#### Parameters
- `y_raw` (*np.ndarray*): Raw audio array of shape `(channels, samples)` or `(samples,)`.
- `sr` (*int*): Sampling rate in Hz.

#### Returns
- `Tuple[StereoAudit, np.ndarray]`:
  1. `StereoAudit`: Stereo field evaluation dataclass.
  2. `y_mono` (*np.ndarray*): Phase-safe mono audio signal for downstream feature extraction.

---

## Snippet Matching & Consolidation (`core.snippets`)

### `find_snippets()`
Discovers matching musical passages between two tracks using tempo-scaled diagonal cross-correlation.

```python
def find_snippets(
    a: TrackAnalysis,
    b: TrackAnalysis,
    min_duration: float = 3.0,
    sim_threshold: float = 0.72,
    auto_tempo_scale: bool = True,
    show_progress: bool = True
) -> List[MatchedSnippet]
```

#### Parameters
- `a` (*TrackAnalysis*): Analyzed reference track.
- `b` (*TrackAnalysis*): Analyzed target track.
- `min_duration` (*float*, optional): Minimum snippet length in seconds. Defaults to `3.0`.
- `sim_threshold` (*float*, optional): Minimum similarity score threshold $[0.0, 1.0]$. Defaults to `0.72`.
- `auto_tempo_scale` (*bool*, optional): When `True`, automatically resamples feature matrices if a valid half-time, double-time, or tempo ratio is detected. Defaults to `True`.
- `show_progress` (*bool*, optional): Enables terminal progress bar for diagonal scanning. Defaults to `True`.

#### Returns
- `List[MatchedSnippet]`: Sorted list of non-redundant matched snippets.

---

### `consolidate_snippets()`
Merges contiguous or adjacent snippet fragments sharing alignment continuity into continuous arrangement sections.

```python
def consolidate_snippets(
    snippets: List[MatchedSnippet],
    tolerance: float = 1.5
) -> List[MatchedSnippet]
```

#### Parameters
- `snippets` (*List[MatchedSnippet]*): List of candidate matched snippets.
- `tolerance` (*float*, optional): Maximum allowable boundary gap (in seconds) between contiguous blocks. Defaults to `1.5`.

#### Returns
- `List[MatchedSnippet]`: Consolidated snippet list with unified timestamps and duration-weighted average similarity scores.

---

### `resample_feature_matrix()`
Resamples a feature matrix along the time dimension according to a temporal speed ratio.

```python
def resample_feature_matrix(features: np.ndarray, ratio: float) -> np.ndarray
```

#### Parameters
- `features` (*np.ndarray*): Feature matrix of shape `(n_features, n_frames)`.
- `ratio` (*float*): Resampling factor ($N_{\text{new}} = N_{\text{orig}} \cdot \text{ratio}$).

#### Returns
- `np.ndarray`: Resampled feature matrix with unit Euclidean norm per frame.

---

## Bipartite Interval Coverage Allocation (`core.coverage`)

### `compute_verified_coverage()`
Performs exact 1D interval subtraction across matched snippets to allocate mutual, non-overlapping coverage.

```python
def compute_verified_coverage(
    a_dur: float,
    b_dur: float,
    matches: List[MatchedSnippet]
) -> Tuple[float, float, float, float, List[Tuple[float, float]], List[Tuple[float, float]]]
```

#### Parameters
- `a_dur` (*float*): Total duration of Song A in seconds.
- `b_dur` (*float*): Total duration of Song B in seconds.
- `matches` (*List[MatchedSnippet]*): Candidate matches between Song A and Song B.

#### Returns
- `Tuple`:
  1. `covered_sec_a` (*float*): Unique verified shared seconds in Track A.
  2. `covered_sec_b` (*float*): Unique verified shared seconds in Track B.
  3. `pct_a` (*float*): Verified shared percentage of Track A $[0.0, 100.0]$.
  4. `pct_b` (*float*): Verified shared percentage of Track B $[0.0, 100.0]$.
  5. `gaps_a` (*List[Tuple[float, float]]*): Disjoint intervals unique to Track A (cut or removed in B) lasting $\ge 1.5\text{ s}$.
  6. `gaps_b` (*List[Tuple[float, float]]*): Disjoint intervals unique to Track B (new intro, outro, or extended sections) lasting $\ge 1.5\text{ s}$.

---

## Reporting & HTML Export (`core.report`)

### `print_full_report()`
Prints the complete comparative audit report to standard output.

```python
def print_full_report(
    a: TrackAnalysis,
    b: TrackAnalysis,
    matches: List[MatchedSnippet],
    mode: str = "precise",
    use_color: bool = True
) -> None
```

#### Parameters
- `a` (*TrackAnalysis*): Reference track analysis.
- `b` (*TrackAnalysis*): Target track analysis.
- `matches` (*List[MatchedSnippet]*): List of matched snippets.
- `mode` (*str*, optional): Output wording style (`"simple"` or `"precise"`). Defaults to `"precise"`.
- `use_color` (*bool*, optional): If `True`, formats terminal output with ANSI color styling. Defaults to `True`.

---

### `generate_html_report()`
Exports an interactive HTML comparison report with native clickable `<details><summary>` accordions.

```python
def generate_html_report(
    a: TrackAnalysis,
    b: TrackAnalysis,
    matches: List[MatchedSnippet],
    output_path: Path,
    mode: str = "precise",
    auto_open: bool = False
) -> None
```

#### Parameters
- `a` (*TrackAnalysis*): Reference track analysis.
- `b` (*TrackAnalysis*): Target track analysis.
- `matches` (*List[MatchedSnippet]*): List of matched snippets.
- `output_path` (*Path*): Output `.html` file destination.
- `mode` (*str*, optional): Wording style (`"simple"` or `"precise"`). Defaults to `"precise"`.
- `auto_open` (*bool*, optional): If `True`, automatically opens the generated file in the default browser. Defaults to `False`.

---

## Utilities & Progress Tracking (`core.utils`)

### `fmt_time()`
Formats a duration in seconds into a standard `MM:SS.ss` string representation.

```python
def fmt_time(seconds: float) -> str
```

---

### `subtract_intervals()`
Subtracts a collection of claimed intervals from a single candidate interval.

```python
def subtract_intervals(
    interval: Tuple[float, float],
    claimed: List[Tuple[float, float]]
) -> List[Tuple[float, float]]
```

---

### `suppress_c_stderr()`
Context manager that redirects C-level `stderr` (descriptor 2 and Windows `STD_ERROR_HANDLE`) to `os.devnull`.

```python
from contextlib import contextmanager
from typing import Generator

@contextmanager
def suppress_c_stderr() -> Generator[None, None, None]
```

#### Example
```python
with suppress_c_stderr():
    # Audio decoding routines that link to C libraries (e.g. libmpg123)
    y, sr = librosa.load("track.mp3")
```

---

### `ProgressBar`
Terminal progress bar utility with TTY auto-detection.

```python
class ProgressBar:
    def __init__(self, total: int, prefix: str = "", length: int = 26, file=sys.stdout) -> None: ...
    def update(self, current: int, suffix: str = "") -> None: ...
    def finish(self, message: str = "") -> None: ...
```