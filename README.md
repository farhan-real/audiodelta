# audiodelta

**audiodelta** is a modular audio engineering quality assurance, structural arrangement diffing, and sample comparison engine. Engineered for mix engineers, mastering houses, audio software developers, and music supervisors, it analyzes and compares two audio tracks across loudness standards, dynamic punch, inter-sample true peaks, multi-window stereo phase fields, and harmonic/timbral snippet alignments.

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

### 4. Speed-Ratio Aware Snippet Discovery & Consolidation
- **Tempo-Preserving Cross-Correlation:** Uses estimated BPM ratios to apply temporal feature matrix resampling, matching identical musical phrases across speed variations (such as half-time or double-time edits).
- **Competitive Non-Maximum Suppression:** Discards secondary chord cross-talk where a brief chord in Song A matches multiple unrelated timestamps across Song B.
- **Contiguous Arrangement Consolidation:** Automatically merges adjacent snippet fragments into cohesive arrangement blocks, transforming fragmented matches into continuous musical sections.

### 5. Verified Bipartite Interval Coverage Allocation
- **Zero False Attribution Across Edits:** Analytic 1D interval subtraction isolates verified shared sections without bridging silent gaps or arrangement modifications.
- **Arrangement Diffing:** Pinpoints cut material present only in Song A and newly added material present only in Song B (e.g. VIP intros, radio edit cuts, extended breakdowns).

### 6. Dual Output Reporting & Interactive HTML Export
- **`simple` Mode:** Natural, intuitive producer/musician terminology (`Song A in Song B`, `Cut from Song A`, `New in Song B`, `Punch`, `Stereo Field`) showing all audio metrics concisely without educational warning paragraphs.
- **`precise` Mode (Default):** Complete audio engineering specifications, decibel notations (`dBFS`, `dBTP`), exact units (`LUFS`, `LU`), ITU-R BS.1770-4 standards, and mastering warnings (DAC clipping risks, streaming ceiling target overshoots, mono cancellation hazards).
- **Interactive Clickable HTML Report:** Generates standalone HTML comparison reports featuring native `<details><summary>` accordions that can be clicked to expand and collapse shared sections and edit breakdowns.
- **ANSI Terminal Colors:** Full color-coded output with automatic TTY and `NO_COLOR` environment detection.

### 7. Real-Time Terminal Progress Feedback
- Step-by-step progress tracking across the 6-stage audio analysis pipeline for both reference and target tracks.
- Real-time progress bar for chunked ITU-R BS.1770-4 true-peak oversampling.
- Progress bar for diagonal cross-correlation matrix scanning.

---

## Installation & Virtual Environment Setup

### 1. Prerequisites
- **Python:** Python 3.8 or higher is required.
- **System Audio Library (`libsndfile`):** Required by `soundfile` to encode and decode audio.
  - **macOS (Homebrew):** `brew install libsndfile`
  - **Ubuntu / Debian:** `sudo apt update && sudo apt install -y libsndfile1`
  - **Windows:** Bundled automatically with binary wheels.

---

### 2. Create the Virtual Environment

Create an isolated virtual environment targeting Python 3.8:

- **macOS & Linux:**
  ```bash
  python3.8 -m venv venv
  ```
  *(If running on a minimal Linux distribution where `ensurepip` is absent, install virtualenv and run `virtualenv -p python3.8 venv`)*

- **Windows (Python Launcher):**
  ```cmd
  py -3.8 -m venv venv
  ```

---

### 3. Activate the Environment

- **macOS & Linux (bash / zsh):**
  ```bash
  source venv/bin/activate
  ```

- **macOS & Linux (fish):**
  ```fish
  source venv/bin/activate.fish
  ```

- **Windows (Command Prompt):**
  ```cmd
  venv\Scripts\activate.bat
  ```

- **Windows (PowerShell):**
  ```powershell
  venv\Scripts\Activate.ps1
  ```

---

### 4. Install Dependencies

Once the environment is active (indicated by `(venv)` in your terminal prompt), upgrade packaging tools to ensure binary wheels download correctly, then install project dependencies:

```bash
pip install --upgrade pip setuptools wheel
pip install numpy scipy librosa soundfile pyloudnorm pytest
```

---

## CLI Usage

Run comparisons directly using the Python interpreter inside your active virtual environment:

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
| `--no-progress` | | `flag` | `False` | Disables progress bars and status updates. |
| `--no-color` | | `flag` | `False` | Disables ANSI color codes in console output. |
| `--html` | | `Path` | `None` | Export clickable interactive HTML report to file path. |
| `--open-html` | | `flag` | `False` | Automatically opens generated HTML report in default browser. |
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
audiodelta analyzing tracks [precise mode]...
  [1/6] track_mix_v1.wav: Loading and decoding audio stream...
  [2/6] track_mix_v1.wav: Auditing stereo correlation and phase stability...
  [3/6] track_mix_v1.wav: Calculating ITU-R BS.1770-4 polyphase true peak...
  [4/6] track_mix_v1.wav: Measuring integrated LUFS and loudness range...
  [5/6] track_mix_v1.wav: Resampling and extracting harmonic/timbral features...
  [6/6] track_mix_v1.wav: Evaluating musical key and 4-band spectral distribution...
  [1/6] track_master_v1.wav: Loading and decoding audio stream...
  [2/6] track_master_v1.wav: Auditing stereo correlation and phase stability...
  [3/6] track_master_v1.wav: Calculating ITU-R BS.1770-4 polyphase true peak...
  [4/6] track_master_v1.wav: Measuring integrated LUFS and loudness range...
  [5/6] track_master_v1.wav: Resampling and extracting harmonic/timbral features...
  [6/6] track_master_v1.wav: Evaluating musical key and 4-band spectral distribution...

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
  Song A in Song B (shared duration):  210.0 s (100.0%)
  Song B in Song A (shared duration):  210.0 s (100.0%)
```

---

### 2. Arrangement & Edit Comparison (Simple Mode)
Compare an Original Track against an Extended Mix or Radio Edit using intuitive producer terms:

```bash
python main.py original.wav radio_edit.wav --mode simple
```

#### Example Output (`simple` mode):
```text
audiodelta analyzing tracks [simple mode]...
  [1/6] original.wav: Loading and decoding audio stream...
  [2/6] original.wav: Auditing stereo correlation and phase stability...
  [3/6] original.wav: Calculating ITU-R BS.1770-4 polyphase true peak...
  [4/6] original.wav: Measuring integrated LUFS and loudness range...
  [5/6] original.wav: Resampling and extracting harmonic/timbral features...
  [6/6] original.wav: Evaluating musical key and 4-band spectral distribution...
  [1/6] radio_edit.wav: Loading and decoding audio stream...
  [2/6] radio_edit.wav: Auditing stereo correlation and phase stability...
  [3/6] radio_edit.wav: Calculating ITU-R BS.1770-4 polyphase true peak...
  [4/6] radio_edit.wav: Measuring integrated LUFS and loudness range...
  [5/6] radio_edit.wav: Resampling and extracting harmonic/timbral features...
  [6/6] radio_edit.wav: Evaluating musical key and 4-band spectral distribution...

[SONG A] original.wav
  Length:          05:12.00 (312.0s)
  Channels:        Stereo (2 channels)
  Tempo:           128.0 BPM
  Key:             F# Minor
  Loudness:        -14.0 LUFS
  Dynamic Range:   7.1 LU
  Punch:           11.8 dB crest
  Peak Level:      -0.30 dB (true peak: -0.10 dB)
  Stereo Field:    +0.91 correlation (worst dip: +0.60)
  Phase Health:    0.0% out-of-phase
  Mono Drop:       -0.20 dB loss
  Side vs Center:  -13.5 dB
  Frequency Energy:
    * Bass (20-250 Hz)    : -6.0 dB
    * Lows (250-1000 Hz)  : -5.0 dB
    * Highs (1-5 kHz)     : -8.0 dB
    * Air (5-20 kHz)      : -11.5 dB

[SONG B] radio_edit.wav
  Length:          03:15.00 (195.0s)
  Channels:        Stereo (2 channels)
  Tempo:           128.0 BPM
  Key:             F# Minor
  Loudness:        -13.8 LUFS
  Dynamic Range:   6.9 LU
  Punch:           11.7 dB crest
  Peak Level:      -0.30 dB (true peak: -0.10 dB)
  Stereo Field:    +0.91 correlation (worst dip: +0.60)
  Phase Health:    0.0% out-of-phase
  Mono Drop:       -0.20 dB loss
  Side vs Center:  -13.5 dB
  Frequency Energy:
    * Bass (20-250 Hz)    : -6.0 dB
    * Lows (250-1000 Hz)  : -5.0 dB
    * Highs (1-5 kHz)     : -8.0 dB
    * Air (5-20 kHz)      : -11.5 dB

=================================================================
MIX COMPARISON (Song B vs. Song A)
=================================================================
  Loudness:        +0.2 LU (Matched)
  Dynamic Range:   -0.2 LU (Consistent)
  Punch:           -0.1 dB (Preserved)
  Peak Level:      +0.00 dB (-0.30 dB sample peak)
  True Peak:       +0.00 dB (-0.10 dB inter-sample)
  Stereo Field:    +0.91 correlation (+0.60 worst)
  Phase Health:    0.0% out-of-phase
  Mono Drop:       -0.20 dB in mono
  Stereo Image:    +0.0 dB side energy (Unchanged)
  Frequency Shifts:
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
ARRANGEMENT BREAKDOWN
-----------------------------------------------------------------
  Song A in Song B:  195.0 s (62.5% of Song A)
  Song B in Song A:  195.0 s (100.0% of Song B)

  Cut from Song A (missing in Song B):
     * 00:00.00 - 00:45.00 (45.0 s)
     * 02:15.00 - 03:00.00 (45.0 s)
     * 04:45.00 - 05:12.00 (27.0 s)
```

---

### 3. Interactive Clickable HTML Report Export
Generate an interactive report and automatically launch it in your default web browser:

```bash
python main.py mix.wav master.wav --html comparison.html --open-html
```

The exported HTML file uses responsive CSS and native `<details><summary>` tags so you can click to expand and collapse every shared section, mix delta, and arrangement edit.

---

## Programmatic Python API

You can import and use `audiodelta` directly within Python scripts, analysis pipelines, and test harnesses. See [API](API.md) for more details.

```python
from pathlib import Path
from core.features import analyze_track
from core.snippets import find_snippets
from core.coverage import compute_verified_coverage
from core.report import generate_html_report, print_full_report

# 1. Analyze both tracks
track_a = analyze_track(Path("reference.wav"))
track_b = analyze_track(Path("target.wav"))

# Access individual metrics
print(f"Track A Integrated LUFS: {track_a.lufs}")
print(f"Track A True Peak:        {track_a.true_peak_dbtp:.2f} dBTP")
print(f"Track B Crest Factor:     {track_b.crest_factor_db:.2f} dB")
print(f"Track B Phase Inversion:  {track_b.stereo.pct_negative_windows:.1f}%")

# 2. Find matching musical snippets (with competitive suppression)
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

print(f"Song A in Song B: {sec_a:.1f}s ({pct_a:.1f}%)")
print(f"Cuts unique to Track A: {cuts_a}")
print(f"Additions unique to Track B: {additions_b}")

# 4. Print console report or generate interactive HTML report
print_full_report(track_a, track_b, snippets, mode="simple")
generate_html_report(track_a, track_b, snippets, output_path=Path("report.html"))
```

---

## Running Unit Tests

With your virtual environment active, run the test suite using `pytest`:

```bash
pytest
```

To run with verbose test output:

```bash
pytest -v
```

### Test Coverage Highlights
- **`tests/testmodels.py`:** Dataclass structures, default types, and property initializations.
- **`tests/testutils.py`:** Centisecond time formatting, interval subtraction arithmetic, `ProgressBar` TTY logic, ANSI `Colors` formatting, and C-level stderr suppression.
- **`tests/testpeak.py`:** Polyphase 48-tap FIR interpolation validation, inter-sample true-peak overshoot detection ($> 0.0\text{ dBTP}$), high sample rate fallback ($> 96\text{ kHz}$), chunking loops, and progress callbacks.
- **`tests/teststereo.py`:** Mono (1D and 2D arrays), perfect dual-mono ($+1.0$), complete anti-phase ($-1.0$), decorrelated orthogonal stereo ($0.0$), and phase cancellation fallback to single-channel routing.
- **`tests/testfeatures.py`:** Key profile estimation across all 12 major and minor modes, Welch PSD spectral power distributions, and MFCC CMVN zero-mean normalizations.
- **`tests/testsnippets.py`:** Feature matrix time-stretching, diagonal cross-correlation run extraction, offset alignment deduplication, tempo scaling ($0.5\times, 1.0\times, 2.0\times$), contiguous snippet consolidation, and verification that distinct unrelated songs yield zero false-positive matches.
- **`tests/testcoverage.py`:** Bipartite non-overlapping allocation, interval subtraction verification, and structural edit discovery.
- **`tests/testreport.py`:** Output verification confirming both simple and precise modes print all metrics, overhauled wording (`Song A in Song B`), and interactive HTML report generation with clickable `<details>` and `<summary>` tags.
- **`tests/testintegration.py`:** End-to-end execution of synthetic audio analysis, report generation, `--html` export, and CLI exit codes for valid and invalid argument sets.

---

## Project Structure

```text
audiodelta/
├── core/
│   ├── __init__.py      # Public exports and package definition
│   ├── constants.py     # Filter taps, frequency limits, pitch profiles
│   ├── models.py        # Dataclasses (MatchedSnippet, StereoAudit, TrackAnalysis)
│   ├── utils.py         # Time formatting, interval arithmetic, colors, progress bars
│   ├── peak.py          # Sample peak and polyphase FIR true-peak engine
│   ├── stereo.py        # Stereo field, sliding correlation, and mid/side analysis
│   ├── features.py      # Welch PSD, key detection, centered chroma, track analysis
│   ├── snippets.py      # Cross-correlation, suppression, and consolidation
│   ├── coverage.py      # Bipartite interval subtraction allocator
│   ├── report.py        # Dual-mode console reports and interactive HTML generation
│   └── cli.py           # Argument parser and CLI execution entry point
├── tests/
│   ├── __init__.py
│   ├── testmodels.py    # Dataclass initialization and field validation tests
│   ├── testutils.py     # Time formatting, interval subtraction, and color tests
│   ├── testpeak.py      # Sample peak and true-peak DSP oversampling tests
│   ├── teststereo.py    # Mono, stereo, anti-phase, and multichannel tests
│   ├── testfeatures.py  # Key estimation, spectral balance, and centered chroma tests
│   ├── testsnippets.py  # Feature resampling, consolidation, and false-positive tests
│   ├── testcoverage.py  # Bipartite non-overlapping allocation and edit gap tests
│   ├── testreport.py    # Output format validation and HTML report tests
│   └── testintegration.py # End-to-end integration and CLI pipeline tests
├── main.py              # Top-level executable runner
├── pytest.ini           # Pytest test discovery configuration
├── .gitignore           # Ignored files, virtual environments, and caches
└── README.md            # Project documentation and usage guide
```

---

## License

MIT License. Engineered for open source audio quality control, production, and mastering research.