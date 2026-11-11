"""
Dual-mode console reporting for audiodelta supporting simple/precise wording,
ANSI terminal coloring, and collapsible shared section views.
"""

from typing import List
from core.coverage import compute_verified_coverage
from core.models import MatchedSnippet, TrackAnalysis
from core.utils import Colors, colorize, fmt_time


def _format_key_simple(key: str) -> str:
    if key == "Undetermined":
        return "Unknown"
    return key.replace("maj", "Major").replace("min", "Minor")


def print_track_card(
    label: str,
    t: TrackAnalysis,
    mode: str = "precise",
    use_color: bool = True
) -> None:
    """Prints a structured summary card for a single analyzed track with optional coloring."""
    tag = colorize(f"[{label}]", Colors.BOLD_CYAN, use_color)

    if mode == "simple":
        print(f"\n{tag} {colorize(t.path.name, Colors.BOLD_WHITE, use_color)}")
        print(f"  Duration:        {fmt_time(t.duration)} ({t.duration:.2f} s)")
        ch_desc = "Mono (1 channel)" if t.stereo.channel_count == 1 else f"Stereo ({t.stereo.channel_count} channels)"
        print(f"  Channels:        {ch_desc}")
        print(f"  Tempo:           {t.bpm:.1f} BPM")
        print(f"  Key:             {_format_key_simple(t.key)}")
        lufs_str = f"{t.lufs:.1f} LUFS" if t.lufs is not None else "N/A"
        lra_str = f"{t.lra:.1f} LU" if t.lra is not None else "N/A"
        print(f"  Loudness:        {lufs_str}")
        print(f"  Dynamic Range:   {lra_str}")
        print(f"  Peak Level:      {t.sample_peak_dbfs:.2f} dB")
        print(f"  True Peak:       {t.true_peak_dbtp:.2f} dB")
        print(f"  Punch:           {t.crest_factor_db:.2f} dB")

        st = t.stereo
        print(f"  Stereo Width:    {st.global_correlation:+.2f} (Worst spot: {st.min_window_correlation:+.2f})")
        inv_str = f"{st.pct_negative_windows:.1f}% inverted"
        if st.pct_negative_windows > 10.0:
            inv_str = colorize(inv_str, Colors.YELLOW, use_color)
        print(f"  Phase Inversion: {inv_str}")
        print(f"  Mono Drop:       {st.mono_loss_db:.2f} dB")
        ms_str = "Center only" if st.mid_side_ratio_db <= -90.0 else f"{st.mid_side_ratio_db:.1f} dB"
        print(f"  Center vs Sides: {ms_str}")

        if t.spectral_balance_db:
            print("  Frequencies:")
            band_map = {
                "Sub/Bass (20-250 Hz)": "Bass (20-250 Hz)",
                "Low-Mid (250-1000 Hz)": "Lows (250-1000 Hz)",
                "High-Mid (1-5 kHz)": "Highs (1-5 kHz)",
                "Air (5-20 kHz)": "Air (5-20 kHz)"
            }
            for k, v in t.spectral_balance_db.items():
                print(f"    * {band_map.get(k, k):<20}: {v:.1f} dB")
    else:
        print(f"\n{tag} {colorize(t.path.name, Colors.BOLD_WHITE, use_color)}")
        print(f"  Duration:         {fmt_time(t.duration)} ({t.duration:.2f} s)")
        print(f"  Channel Format:   {t.stereo.channel_layout_desc}")
        print(f"  Estimated BPM:    {t.bpm:.1f}")
        print(f"  Estimated Key:    {t.key}")
        lufs_str = f"{t.lufs:.1f} LUFS" if t.lufs is not None else "N/A"
        lra_str = f"{t.lra:.1f} LU" if t.lra is not None else "N/A"
        print(f"  Loudness (LUFS):  {lufs_str:<10} | LRA: {lra_str}")
        print(f"  Sample Peak:      {t.sample_peak_dbfs:.2f} dBFS   | True Peak: {t.true_peak_dbtp:.2f} dBTP")
        print(f"  Dynamic Punch:    {t.crest_factor_db:.2f} dB Crest Factor (True Peak / RMS)")

        st = t.stereo
        ms_str = "Mono" if st.mid_side_ratio_db <= -90.0 else f"{st.mid_side_ratio_db:.1f} dB"
        print(f"  Stereo Field:     Global Corr: {st.global_correlation:+.2f} | M/S Energy Ratio: {ms_str}")
        print(f"  Phase Stability:  Min Window:  {st.min_window_correlation:+.2f} | Negative Windows: {st.pct_negative_windows:.1f}%")
        if st.mono_loss_db < -0.2:
            print(f"  Mono Fold-Down:   {st.mono_loss_db:.2f} dB Power Loss upon mono sum")
        if st.mono_fallback_used:
            notice = colorize("  [NOTICE] Severe out-of-phase mono cancellation; Left channel was used for feature extraction.", Colors.YELLOW, use_color)
            print(notice)

        if t.spectral_balance_db:
            print("  4-Band Relative Power:")
            for k, v in t.spectral_balance_db.items():
                print(f"    * {k:<24}: {v:.1f} dB")


def print_mix_deltas(
    a: TrackAnalysis,
    b: TrackAnalysis,
    mode: str = "precise",
    use_color: bool = True
) -> None:
    """Prints comparative mix deltas between Song A and Song B with color indicators."""
    rule = colorize("=" * 65, Colors.CYAN, use_color)

    if mode == "simple":
        print(f"\n{rule}")
        print(colorize("MIX COMPARISON (Song B vs. Song A)", Colors.BOLD_CYAN, use_color))
        print(rule)

        if a.lufs is not None and b.lufs is not None:
            delta_lufs = b.lufs - a.lufs
            if delta_lufs > 0.2:
                direction = colorize("Louder", Colors.GREEN, use_color)
            elif delta_lufs < -0.2:
                direction = colorize("Quieter", Colors.YELLOW, use_color)
            else:
                direction = "Matched"
            print(f"  Loudness:         {delta_lufs:+.1f} LU ({direction})")
        else:
            print("  Loudness:         N/A")

        if a.lra is not None and b.lra is not None:
            delta_lra = b.lra - a.lra
            spread = "More dynamic" if delta_lra > 0.5 else "Tighter" if delta_lra < -0.5 else "Consistent"
            print(f"  Dynamic Range:    {delta_lra:+.1f} LU ({spread})")

        delta_crest = b.crest_factor_db - a.crest_factor_db
        crest_tag = "Punchier" if delta_crest > 0.5 else "More compressed" if delta_crest < -0.5 else "Preserved"
        print(f"  Punch:            {delta_crest:+.1f} dB ({crest_tag})")

        delta_sample = b.sample_peak_dbfs - a.sample_peak_dbfs
        delta_true = b.true_peak_dbtp - a.true_peak_dbtp
        print(f"  Peak Level:       {delta_sample:+.2f} dB ({b.sample_peak_dbfs:.2f} dB)")
        print(f"  True Peak:        {delta_true:+.2f} dB ({b.true_peak_dbtp:.2f} dB)")

        st_b = b.stereo
        print(f"  Stereo Width:     {st_b.global_correlation:+.2f} ({st_b.min_window_correlation:+.2f} worst)")
        print(f"  Phase Inversion:  {st_b.pct_negative_windows:.1f}% inverted")
        print(f"  Mono Drop:        {st_b.mono_loss_db:.2f} dB")

        if a.stereo.mid_side_ratio_db > -90.0 and st_b.mid_side_ratio_db > -90.0:
            delta_ms = st_b.mid_side_ratio_db - a.stereo.mid_side_ratio_db
            w_dir = "Wider" if delta_ms > 0.5 else "Narrower" if delta_ms < -0.5 else "Unchanged"
            print(f"  Width Change:     {delta_ms:+.1f} dB ({w_dir})")

        if a.spectral_balance_db and b.spectral_balance_db:
            print("  Frequency Changes:")
            band_map = {
                "Sub/Bass (20-250 Hz)": "Bass",
                "Low-Mid (250-1000 Hz)": "Lows",
                "High-Mid (1-5 kHz)": "Highs",
                "Air (5-20 kHz)": "Air"
            }
            for band_name in a.spectral_balance_db:
                if band_name in b.spectral_balance_db:
                    delta_p = b.spectral_balance_db[band_name] - a.spectral_balance_db[band_name]
                    if delta_p > 0.5:
                        status = colorize("More", Colors.GREEN, use_color)
                    elif delta_p < -0.5:
                        status = colorize("Less", Colors.YELLOW, use_color)
                    else:
                        status = "Same"
                    short_name = band_map.get(band_name, band_name)
                    print(f"    * {short_name:<16}: {delta_p:+.1f} dB ({status})")
    else:
        print(f"\n{rule}")
        print(colorize("MIX AND MASTERING AUDIT (Song B vs. Song A)", Colors.BOLD_CYAN, use_color))
        print(rule)

        if a.lufs is not None and b.lufs is not None:
            delta_lufs = b.lufs - a.lufs
            if delta_lufs > 0.2:
                indicator = colorize("Louder", Colors.GREEN, use_color)
            elif delta_lufs < -0.2:
                indicator = colorize("Quieter", Colors.YELLOW, use_color)
            else:
                indicator = "Matched loudness"
            print(f"  Loudness:         Song B is {delta_lufs:+.1f} LU ({indicator})")
        else:
            print("  Loudness:         N/A (LUFS not available)")

        if a.lra is not None and b.lra is not None:
            delta_lra = b.lra - a.lra
            indicator = "More dynamic variation" if delta_lra > 0.5 else "More compressed" if delta_lra < -0.5 else "Consistent range"
            print(f"  Loudness Range:   Song B is {delta_lra:+.1f} LU ({indicator})")

        delta_crest = b.crest_factor_db - a.crest_factor_db
        crest_indicator = "More dynamic punch" if delta_crest > 0.5 else "More limited/compressed" if delta_crest < -0.5 else "Preserved dynamics"
        print(f"  Dynamic Punch:    {delta_crest:+.1f} dB Crest Factor ({crest_indicator})")

        delta_sample = b.sample_peak_dbfs - a.sample_peak_dbfs
        delta_true = b.true_peak_dbtp - a.true_peak_dbtp
        print(f"  Sample Peak:      Song B is {b.sample_peak_dbfs:.2f} dBFS (Delta: {delta_sample:+.2f} dBFS)")
        if b.sample_peak_dbfs > 0.0:
            print(colorize("  [ALERT] Decoded audio file contains floating-point samples exceeding 0.0 dBFS.", Colors.BOLD_RED, use_color))

        print(f"  True Peak:        Song B is {b.true_peak_dbtp:.2f} dBTP (Delta: {delta_true:+.2f} dBTP)")
        if b.true_peak_dbtp > 0.0 and b.sample_peak_dbfs <= 0.0:
            print(colorize("  [WARNING] Inter-sample true-peak overshoot detected (> 0.0 dBTP). Risk of DAC/lossy clipping.", Colors.BOLD_YELLOW, use_color))
        elif b.true_peak_dbtp > -1.0 and b.true_peak_dbtp <= 0.0:
            print(colorize("  [NOTICE] True peak exceeds standard streaming ceiling (-1.0 dBTP target margin).", Colors.CYAN, use_color))

        st_b = b.stereo
        print(f"\n  Stereo Correlation: {st_b.global_correlation:+.2f} (Worst Window: {st_b.min_window_correlation:+.2f})")
        if st_b.pct_negative_windows > 15.0:
            print(colorize(f"  [CRITICAL] Severe mono cancellation: {st_b.pct_negative_windows:.1f}% of active windows have negative correlation.", Colors.BOLD_RED, use_color))
        elif st_b.pct_negative_windows > 5.0:
            print(colorize(f"  [WARNING] Out-of-phase material: {st_b.pct_negative_windows:.1f}% negative correlation windows.", Colors.BOLD_YELLOW, use_color))

        if st_b.mono_loss_db < -3.0:
            print(colorize(f"  [WARNING] Severe mono fold-down attenuation: {st_b.mono_loss_db:.1f} dB loss when collapsed to mono.", Colors.BOLD_YELLOW, use_color))

        if a.stereo.mid_side_ratio_db > -90.0 and st_b.mid_side_ratio_db > -90.0:
            delta_ms = st_b.mid_side_ratio_db - a.stereo.mid_side_ratio_db
            width_indicator = "Wider" if delta_ms > 0.5 else "Narrower" if delta_ms < -0.5 else "Unchanged"
            print(f"  Stereo Width:     Side/Mid energy changed by {delta_ms:+.1f} dB ({width_indicator})")

        if a.spectral_balance_db and b.spectral_balance_db:
            print("\n  4-Band Spectral Balance Shift (Relative Share):")
            for band_name in a.spectral_balance_db:
                if band_name in b.spectral_balance_db:
                    delta_p = b.spectral_balance_db[band_name] - a.spectral_balance_db[band_name]
                    if delta_p > 0.5:
                        indicator = colorize("Boosted", Colors.GREEN, use_color)
                    elif delta_p < -0.5:
                        indicator = colorize("Cut", Colors.YELLOW, use_color)
                    else:
                        indicator = "Even"
                    print(f"    * {band_name:<24}: {delta_p:+.1f} dB [{indicator}]")


def print_snippet_report(
    a: TrackAnalysis,
    b: TrackAnalysis,
    matches: List[MatchedSnippet],
    mode: str = "precise",
    use_color: bool = True,
    max_snippets: int = 10,
    all_snippets: bool = False,
    collapse_snippets: bool = False
) -> None:
    """
    Prints snippet match list and bipartite interval edit audit with collapsible display.
    """
    rule = colorize("=" * 65, Colors.CYAN, use_color)
    subrule = colorize("-" * 65, Colors.CYAN, use_color)
    header_shared = "SHARED SECTIONS" if mode == "simple" else "SHARED SNIPPETS AND SAMPLES"

    print(f"\n{rule}")
    print(colorize(header_shared, Colors.BOLD_CYAN, use_color))
    print(rule)

    total_matches = len(matches)

    if not matches:
        print("  No shared musical snippets detected above threshold.")
    elif collapse_snippets:
        collapse_msg = colorize(
            f"  [v] Snippet table collapsed ({total_matches} sections detected). Pass '--all-snippets' to expand.",
            Colors.DIM,
            use_color
        )
        print(collapse_msg)
    else:
        if mode == "simple":
            print(f"{'Song A':<18}{'Song B':<18}{'Length':<10}{'Match':<11}{'Melody':<10}{'Tone'}")
        else:
            print(f"{'Song A Location':<18}{'Song B Location':<18}{'Duration':<10}{'Composite':<11}{'Harmonic':<10}{'Timbral'}")
        print("-" * 75)

        display_limit = total_matches if all_snippets else min(total_matches, max_snippets)
        displayed_matches = matches[:display_limit]

        for m in displayed_matches:
            loc_a = f"{fmt_time(m.a_start)}-{fmt_time(m.a_end)}"
            loc_b = f"{fmt_time(m.b_start)}-{fmt_time(m.b_end)}"
            dur = f"{m.duration_a:.1f} s"

            # Color-code strong matches
            comp_val = m.composite_similarity * 100
            comp_str = f"{comp_val:.1f}%"
            if comp_val >= 90.0:
                comp_str = colorize(comp_str, Colors.BOLD_GREEN, use_color)
            elif comp_val >= 75.0:
                comp_str = colorize(comp_str, Colors.YELLOW, use_color)

            harm = f"{m.harmonic_similarity * 100:.1f}%"
            timb = f"{m.timbral_similarity * 100:.1f}%"
            print(f"{loc_a:<18}{loc_b:<18}{dur:<10}{comp_str:<20}{harm:<10}{timb}")

        if not all_snippets and total_matches > max_snippets:
            hidden = total_matches - max_snippets
            fold_note = colorize(
                f"\n  [+] Showing {max_snippets} of {total_matches} sections ({hidden} additional sections collapsed).\n"
                f"      Run with '--all-snippets' to display all, or '--max-snippets <N>' to adjust.",
                Colors.CYAN,
                use_color
            )
            print(fold_note)

    sec_a, sec_b, pct_a, pct_b, gaps_a, gaps_b = compute_verified_coverage(a.duration, b.duration, matches)

    header_audit = "ARRANGEMENT CHANGES" if mode == "simple" else "STRUCTURAL EDIT AUDIT (Verified Bipartite Allocation)"
    print(f"\n{subrule}")
    print(colorize(header_audit, Colors.BOLD_CYAN, use_color))
    print(subrule)
    print(f"  Song A shared:    {sec_a:.1f} s ({pct_a:.1f}%)")
    print(f"  Song B shared:    {sec_b:.1f} s ({pct_b:.1f}%)")

    if gaps_a and (not matches or pct_a < 99.0):
        cut_label = "Removed from Song A:" if mode == "simple" else "[CUT] Unique to Song A (Cut or missing in Song B):"
        cut_label_colored = colorize(f"\n  {cut_label}", Colors.YELLOW, use_color)
        print(cut_label_colored)
        for s, e in gaps_a:
            print(f"     * {fmt_time(s)} - {fmt_time(e)} ({e - s:.1f} s)")

    if gaps_b and (not matches or pct_b < 99.0):
        added_label = "Added to Song B:" if mode == "simple" else "[ADDED] Unique to Song B (New intro/outro, extended cuts, or distinct material):"
        added_label_colored = colorize(f"\n  {added_label}", Colors.GREEN, use_color)
        print(added_label_colored)
        for s, e in gaps_b:
            print(f"     * {fmt_time(s)} - {fmt_time(e)} ({e - s:.1f} s)")


def print_full_report(
    a: TrackAnalysis,
    b: TrackAnalysis,
    matches: List[MatchedSnippet],
    mode: str = "precise",
    use_color: bool = True,
    max_snippets: int = 10,
    all_snippets: bool = False,
    collapse_snippets: bool = False
) -> None:
    """Prints the complete comparison report with color and collapsible snippet options."""
    print_track_card("SONG A", a, mode=mode, use_color=use_color)
    print_track_card("SONG B", b, mode=mode, use_color=use_color)
    print_mix_deltas(a, b, mode=mode, use_color=use_color)
    print_snippet_report(
        a,
        b,
        matches,
        mode=mode,
        use_color=use_color,
        max_snippets=max_snippets,
        all_snippets=all_snippets,
        collapse_snippets=collapse_snippets
    )