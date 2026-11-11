# Audiodelta

**Audiodelta** is a modular audio engineering quality assurance, structural arrangement diffing, and sample comparison engine. Engineered for mix engineers, mastering houses, audio software developers, and music supervisors, it analyzes and compares two audio tracks across loudness standards, dynamic punch, inter-sample true peaks, multi-window stereo phase fields, and harmonic/timbral snippet alignments.

---

## Key Capabilities

### 1. Mix vs. Master QA
- **Loudness & Dynamics:** Integrated Loudness (`LUFS` under ITU-R BS.1770-4) and Loudness Range (`LRA`) computed via `pyloudnorm`.
- **Dynamic Punch (Crest Factor):** Evaluates true dynamic punch by tracking the logarithmic ratio between true peak amplitude and linear RMS energy.
- **ITU-R BS.1770-4 Compliant True Peak:** 48-tap polyphase FIR filter (4 phases x 12 taps) running 4x oversampling with group-delay compensation ($\tau = 23.5$ samples) to flag inter-sample overshoots ($> 0.0\text{ dBTP}$) and verify streaming ceiling targets ($-1.0\text{ dBTP}$). Tracks at sample rates exceeding $96\text{ kHz}$ automatically transition to optimized 2x polynomial polyphase resampling.
- **4-Band Spectral Balance:** Welch Power Spectral Density (PSD) analysis across Sub/Bass ($20\text{–}250\text{ Hz}$), Low-Mid ($250\text{–}1000\text{ Hz}$), High-Mid ($1\text{–}5\text{ kHz}$), and Air ($5\text{–}20\text{ kHz}$). Calculated across discrete channels to avoid mono sum phase cancellations.

### 2. Multi-Window Stereo Phase & Mono Compatibility
- **Global & Sliding Correlation:** Measures full-track Pearson phase correlation alongside a $100\text{ ms}$ sliding window ($50\%$ overlap) to track instantaneous out-of-phase passages.
- **Phase Inversion Tracking:** Computes the exact percentage of active playback windows exhibiting negative correlation ($\rho < 0.0$).
- **Mono Fold-Down Attenuation:** Quantifies power loss (in dB) when collapsing stereo to mono ($M = \frac{L + R}{2}$).
- **Mid/Side Energy Ratio:** Evaluates stereo image width changes via logarithmic side-to-mid energy quotients.
- **Phase-Protected Mono Downmix:** Automatically detects destructive mono cancellation ($E_{\text{mid}} < 0.01 \cdot E_{\text{left}}$) and falls back to isolated single-channel routing for feature analysis.

### 3. Dual-Domain Feature Extraction
- **Harmonic & Timbral Separation:** Extracts harmonic pitch content via 12-bin Constant-Q Chromagrams (`C1` to `B7`) and timbral envelope textures via 13 Mel-Frequency Cepstral Coefficients (MFCCs).
- **Cepstral Mean and Variance Normalization (CMVN):** Normalizes MFCCs across time to ensure timbre comparisons remain invariant to static EQ balance shifts.
- **Independent L2-Normalization & Weighting:** Combines normalized harmonic ($70\%$) and timbral ($30\%$) vectors into a 25-dimensional unit-norm descriptor per analysis frame.

### 4. Speed-Ratio Aware Snippet Discovery
- **Tempo-Preserving Cross-Correlation:** Uses estimated BPM ratios to apply temporal feature matrix resampling, matching identical musical phrases across speed variations (such as half-time or double-time edits).
- **Slope-Aware Deduplication:** Merges redundant overlapping candidate runs along time-delay diagonals using alignment offsets ($c = t_B - r \cdot t_A$).

### 5. Verified Bipartite Interval Coverage Allocation
- **Zero False Attribution Across Edits:** Analytic 1D interval subtraction isolates verified shared sections without bridging silent gaps or arrangement modifications.
- **Arrangement Diffing:** Pinpoints cut material present only in Song A and newly added material present only in Song B (e.g. VIP intros, radio edit cuts, extended breakdowns).

### 6. Dual Output Reporting Modes
- **`precise` (Default):** Full engineering specifications, decibel notations (`dBFS`, `dBTP`), exact units (`LUFS`, `LU`), ITU-R standards, and mastering warnings (DAC clipping risks, streaming ceiling target overshoots, mono cancellation hazards).
- **`simple`:** Direct, practical production terms showing all metrics concisely while strictly omitting educational paragraphs and alert texts.

---

## Project Structure

```text
audiodelta/
├── core/
│   ├── __init__.py      # Public exports and package definition
│   ├── constants.py     # Filter taps, frequency limits, pitch profiles
│   ├── models.py        # Dataclasses (MatchedSnippet, StereoAudit, TrackAnalysis)
│   ├── utils.py         # Time string formatting and interval subtraction arithmetic
│   ├── peak.py          # Sample peak and polyphase FIR true-peak engine
│   ├── stereo.py        # Stereo field, sliding correlation, and mid/side analysis
│   ├── features.py      # Welch PSD, key detection, MFCC/chroma, track analysis
│   ├── snippets.py      # Diagonal cross-correlation and snippet deduplication
│   ├── coverage.py      # Bipartite interval subtraction allocator
│   ├── report.py        # Dual-mode console report formatter (simple & precise)
│   └── cli.py           # Argument parser and CLI execution entry point
├── tests/
│   ├── __init__.py
│   ├── testmodels.py    # Dataclass initialization and field validation tests
│   ├── testutils.py     # Time formatting and interval subtraction unit tests
│   ├── testpeak.py      # Sample peak and true-peak DSP oversampling tests
│   ├── teststereo.py    # Mono, stereo, anti-phase, and multichannel tests
│   ├── testfeatures.py  # Key estimation, spectral balance, and feature extraction tests
│   ├── testsnippets.py  # Feature resampling, diagonal matching, and deduplication tests
│   ├── testcoverage.py  # Bipartite non-overlapping allocation and edit gap tests
│   ├── testreport.py    # Output format validation for simple and precise modes
│   └── testintegration.py # End-to-end integration and CLI pipeline tests
├── main.py              # Top-level executable runner
└── pytest.ini           # Pytest test discovery configuration
```

---

## Installation

### Prerequisites
- Python 3.8 or higher
- `libsndfile` installed on your system (standard on macOS and Linux; bundled on Windows)

### Dependencies
Install the required dependencies via `pip`:

```bash
pip install numpy scipy librosa soundfile pyloudnorm pytest
```

---

## CLI Usage

Run comparisons directly from the command line using `main.py`:

```bash
python main.py <song_a> <song_b> [options]
```

### Positional Arguments
- `song_a`: Path to the reference audio track (e.g., Mix v1, Unmastered bounce, Radio Edit).
- `song_b`: Path to the target audio track (e.g., Master, Club VIP, Extended Mix).

### Optional Arguments
| Option | Short | Type | Default | Description |
|---|---|---|---|---|
| `--mode` | `-m` | `simple` \| `precise` | `precise` | Wording style for report outputs. |
| `--min-len` | | `float` | `3.0` | Minimum matched snippet duration in seconds. |
| `--threshold` | | `float` | `0.72` | Similarity threshold for frame alignment ($0.0\text{–}1.0$). |
| `--no-tempo-rescale` | | `flag` | `False` | Disables automatic tempo-ratio scaling during matching. |
| `--help` | `-h` | | | Displays argument documentation and exits. |

---

## CLI Examples

### 1. Standard Mastering QA (Precise Mode)
Compare an unmastered mix bounce against a mastered release:

```bash
python main.py mixes/track_mix_v1.wav masters/track_master_v1.wav --mode precise
```

#### Example Output (`precise` mode):
```text
audiodelta analyzing audio files [precise mode]...

[SONG A] track_mix_v1.wav
  Duration:         03:30.00 (210.00 s)
  Channel Format:   Stereo (2 ch)
  Estimated BPM:    126.0
  Estimated Key:    D min
  Loudness (LUFS):  -18.4 LUFS   | LRA: 8.2 LU
  Sample Peak:      -2.10 dBFS   | True Peak: -1.85 dBTP
  Dynamic Punch:    14.20 dB Crest Factor (True Peak / RMS)
  Stereo Field:     Global Corr: +0.94 | M/S Energy Ratio: -15.1 dB
  Phase Stability:  Min Window:  +0.71 | Negative Windows: 0.0%
  4-Band Relative Power:
    * Sub/Bass (20-250 Hz)    : -5.8 dB
    * Low-Mid (250-1000 Hz)   : -4.9 dB
    * High-Mid (1-5 kHz)      : -7.6 dB
    * Air (5-20 kHz)          : -11.2 dB

[SONG B] track_master_v1.wav
  Duration:         03:30.00 (210.00 s)
  Channel Format:   Stereo (2 ch)
  Estimated BPM:    126.0
  Estimated Key:    D min
  Loudness (LUFS):  -9.1 LUFS    | LRA: 4.8 LU
  Sample Peak:      0.00 dBFS    | True Peak: +0.42 dBTP
  Dynamic Punch:    8.90 dB Crest Factor (True Peak / RMS)
  Stereo Field:     Global Corr: +0.89 | M/S Energy Ratio: -12.4 dB
  Phase Stability:  Min Window:  +0.55 | Negative Windows: 0.0%
  4-Band Relative Power:
    * Sub/Bass (20-250 Hz)    : -5.2 dB
    * Low-Mid (250-1000 Hz)   : -5.1 dB
    * High-Mid (1-5 kHz)      : -7.0 dB
    * Air (5-20 kHz)          : -10.1 dB

=================================================================
MIX AND MASTERING AUDIT (Song B vs. Song A)
=================================================================
  Loudness:         Song B is +9.3 LU (Louder)
  Loudness Range:   Song B is -3.4 LU (More compressed)
  Dynamic Punch:    -5.3 dB Crest Factor (More limited/compressed)
  Sample Peak:      Song B is 0.00 dBFS (Delta: +2.10 dBFS)
  True Peak:        Song B is +0.42 dBTP (Delta: +2.27 dBTP)
  [WARNING] Inter-sample true-peak overshoot detected (> 0.0 dBTP). Risk of DAC/lossy clipping.

  Stereo Correlation: +0.89 (Worst Window: +0.55)
  Stereo Width:     Side/Mid energy changed by +2.7 dB (Wider)

  4-Band Spectral Balance Shift (Relative Share):
    * Sub/Bass (20-250 Hz)    : +0.6 dB [Boosted]
    * Low-Mid (250-1000 Hz)   : -0.2 dB [Even]
    * High-Mid (1-5 kHz)      : +0.6 dB [Boosted]
    * Air (5-20 kHz)          : +1.1 dB [Boosted]

=================================================================
SHARED SNIPPETS AND SAMPLES
=================================================================
Song A Location   Song B Location   Duration  Composite  Harmonic  Timbral
---------------------------------------------------------------------------
00:00.00-03:30.00 00:00.00-03:30.00 210.0 s   98.2%      99.1%     95.4%

-----------------------------------------------------------------
STRUCTURAL EDIT AUDIT (Verified Bipartite Allocation)
-----------------------------------------------------------------
  Song A verified shared duration:   210.0 s (100.0%)
  Song B verified shared duration:   210.0 s (100.0%)
```

---

### 2. Arrangement & Edit Comparison (Simple Mode)
Compare an Original Track against a Radio Edit using concise production wording:

```bash
python main.py original.wav radio_edit.wav --mode simple
```

#### Example Output (`simple` mode):
```text
audiodelta analyzing audio files [simple mode]...

[SONG A] original.wav
  Duration:        05:12.00 (312.00 s)
  Channels:        Stereo (2 channels)
  Tempo:           128.0 BPM
  Key:             F# Minor
  Loudness:        -14.0 LUFS
  Dynamic Range:   7.1 LU
  Peak Level:      -0.30 dB
  True Peak:       -0.10 dB
  Punch:           11.80 dB
  Stereo Width:    +0.91 (Worst spot: +0.60)
  Phase Inversion: 0.0% inverted
  Mono Drop:       -0.20 dB
  Center vs Sides: -13.5 dB
  Frequencies:
    * Bass (20-250 Hz)    : -6.0 dB
    * Lows (250-1000 Hz)  : -5.0 dB
    * Highs (1-5 kHz)     : -8.0 dB
    * Air (5-20 kHz)      : -11.5 dB

[SONG B] radio_edit.wav
  Duration:        03:15.00 (195.00 s)
  Channels:        Stereo (2 channels)
  Tempo:           128.0 BPM
  Key:             F# Minor
  Loudness:        -13.8 LUFS
  Dynamic Range:   6.9 LU
  Peak Level:      -0.30 dB
  True Peak:       -0.10 dB
  Punch:           11.70 dB
  Stereo Width:    +0.91 (Worst spot: +0.60)
  Phase Inversion: 0.0% inverted
  Mono Drop:       -0.20 dB
  Center vs Sides: -13.5 dB
  Frequencies:
    * Bass (20-250 Hz)    : -6.0 dB
    * Lows (250-1000 Hz)  : -5.0 dB
    * Highs (1-5 kHz)     : -8.0 dB
    * Air (5-20 kHz)      : -11.5 dB

=================================================================
MIX COMPARISON (Song B vs. Song A)
=================================================================
  Loudness:         +0.2 LU (Matched)
  Dynamic Range:    -0.2 LU (Consistent)
  Punch:            -0.1 dB (Preserved)
  Peak Level:       +0.00 dB (-0.30 dB)
  True Peak:        +0.00 dB (-0.10 dB)
  Stereo Width:     +0.91 (+0.60 worst)
  Phase Inversion:  0.0% inverted
  Mono Drop:        -0.20 dB
  Width Change:     +0.0 dB (Unchanged)
  Frequency Changes:
    * Bass            : +0.0 dB (Same)
    * Lows            : +0.0 dB (Same)
    * Highs           : +0.0 dB (Same)
    * Air             : +0.0 dB (Same)

=================================================================
SHARED SECTIONS
=================================================================
Song A            Song B            Length    Match      Melody    Tone
---------------------------------------------------------------------------
00:45.00-02:15.00 00:15.00-01:45.00 90.0 s    99.1%      99.8%     97.5%
03:00.00-04:45.00 01:45.00-03:30.00 105.0 s   98.9%      99.7%     97.1%

-----------------------------------------------------------------
ARRANGEMENT CHANGES
-----------------------------------------------------------------
  Song A shared:    195.0 s (62.5%)
  Song B shared:    195.0 s (100.0%)

  Removed from Song A:
     * 00:00.00 - 00:45.00 (45.0 s)
     * 02:15.00 - 03:00.00 (45.0 s)
     * 04:45.00 - 05:12.00 (27.0 s)
```

---

### 3. Sample Detection with Adjusted Length and Threshold
Find short sampled snippets ($\ge 1.5\text{ seconds}$) with a custom match threshold ($0.65$):

```bash
python main.py original_sample.wav remix_stem.wav --min-len 1.5 --threshold 0.65
```

---

## Programmatic Python API

You can import and use `audiodelta` directly within Python scripts or testing harnesses:

```python
from pathlib import Path
from core.features import analyze_track
from core.snippets import find_snippets
from core.coverage import compute_verified_coverage
from core.report import print_full_report

# 1. Analyze both tracks
track_a = analyze_track(Path("reference.wav"))
track_b = analyze_track(Path("target.wav"))

# Access individual metrics
print(f"Track A Integrated LUFS: {track_a.lufs}")
print(f"Track A True Peak:        {track_a.true_peak_dbtp:.2f} dBTP")
print(f"Track B Crest Factor:     {track_b.crest_factor_db:.2f} dB")
print(f"Track B Phase Inversion:  {track_b.stereo.pct_negative_windows:.1f}%")

# 2. Find matching musical snippets
snippets = find_snippets(
    track_a,
    track_b,
    min_duration=3.0,
    sim_threshold=0.72,
    auto_tempo_scale=True
)

for match in snippets:
    print(f"Match: {match.a_start:.1f}s-{match.a_end:.1f}s -> {match.b_start:.1f}s-{match.b_end:.1f}s")
    print(f"Composite: {match.composite_similarity * 100:.1f}% | Harmonic: {match.harmonic_similarity * 100:.1f}%")

# 3. Compute verified structural coverage
sec_a, sec_b, pct_a, pct_b, cuts_a, additions_b = compute_verified_coverage(
    track_a.duration,
    track_b.duration,
    snippets
)

print(f"Verified Shared Duration in Track A: {sec_a:.1f}s ({pct_a:.1f}%)")
print(f"Cuts unique to Track A: {cuts_a}")
print(f"Additions unique to Track B: {additions_b}")

# 4. Print reports in 'simple' or 'precise' format
print_full_report(track_a, track_b, snippets, mode="precise")
```

---

## Running Unit Tests

The test suite provides complete coverage across individual math routines, edge cases (silence, mono fallback, phase cancellations, clipping overshoots, chunking boundaries), and end-to-end integration workflows.

Run all tests using `pytest`:

```bash
pytest
```

To run with verbose output:

```bash
pytest -v
```

### Test Coverage Highlights
- **`tests/testmodels.py`:** Dataclass structures, default types, and property initializations.
- **`tests/testutils.py`:** Centisecond time formatting, edge cases ($0.0\text{s}$, negative values), and 1D interval subtraction edge cases (non-overlap, head/tail cuts, internal splits, multi-interval claims).
- **`tests/testpeak.py`:** Polyphase 48-tap FIR interpolation validation, inter-sample true-peak overshoot detection ($> 0.0\text{ dBTP}$), high sample rate fallback ($> 96\text{ kHz}$), and memory chunking boundaries.
- **`tests/teststereo.py`:** Mono (1D and 2D arrays), perfect dual-mono ($+1.0$), complete anti-phase ($-1.0$), decorrelated orthogonal stereo ($0.0$), and phase cancellation fallback to single-channel routing.
- **`tests/testfeatures.py`:** Key profile estimation across all 12 major and minor modes, Welch PSD spectral power distributions, and MFCC CMVN zero-mean normalizations.
- **`tests/testsnippets.py`:** Feature matrix time-stretching, diagonal cross-correlation run extraction, offset alignment deduplication, and tempo scaling ($0.5\times, 1.0\times, 2.0\times$).
- **`tests/testcoverage.py`:** Bipartite non-overlapping allocation, interval subtraction verification, and structural edit discovery.
- **`tests/testreport.py`:** Output verification confirming both simple and precise modes print all metrics, with simple mode omitting explanatory warnings.
- **`tests/testintegration.py`:** End-to-end execution of synthetic audio analysis, report generation, and CLI exit codes for valid and invalid argument sets.

---

## License

MIT License. Engineered for open source audio quality control, production, and mastering research.