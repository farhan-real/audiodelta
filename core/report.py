"""
Dual-mode console reporting for audiodelta supporting simple/precise wording,
vibrant ANSI terminal coloring for numbers and metrics, and interactive HTML generation.
"""

from pathlib import Path
from typing import List
import webbrowser

from core.coverage import compute_verified_coverage
from core.models import MatchedSnippet, TrackAnalysis
from core.utils import (
    Colors,
    c_corr,
    c_delta,
    c_num,
    c_pct,
    c_peak,
    colorize,
    fmt_time,
)


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
    """Prints a structured summary card for a single analyzed track with vibrant metric coloring."""
    tag_color = Colors.BOLD_CYAN if label == "SONG A" else Colors.BOLD_MAGENTA
    tag = colorize(f"[{label}]", tag_color, use_color)
    name_str = colorize(t.path.name, Colors.BOLD_WHITE, use_color)

    if mode == "simple":
        print(f"\n{tag} {name_str}")
        t_str = f"{fmt_time(t.duration)} ({t.duration:.1f}s)"
        print(f"  Length:          {colorize(t_str, Colors.BOLD_YELLOW, use_color)}")
        ch_desc = "Mono (1 channel)" if t.stereo.channel_count == 1 else f"Stereo ({t.stereo.channel_count} channels)"
        print(f"  Channels:        {colorize(ch_desc, Colors.WHITE, use_color)}")
        print(f"  Tempo:           {c_num(f'{t.bpm:.1f}', Colors.BOLD_YELLOW, use_color)} {colorize('BPM', Colors.DIM, use_color)}")
        print(f"  Key:             {colorize(_format_key_simple(t.key), Colors.BOLD_CYAN, use_color)}")

        lufs_str = f"{c_num(f'{t.lufs:.1f}', Colors.BOLD_YELLOW, use_color)} {colorize('LUFS', Colors.DIM, use_color)}" if t.lufs is not None else colorize("N/A", Colors.DIM, use_color)
        lra_str = f"{c_num(f'{t.lra:.1f}', Colors.BOLD_YELLOW, use_color)} {colorize('LU', Colors.DIM, use_color)}" if t.lra is not None else colorize("N/A", Colors.DIM, use_color)
        print(f"  Loudness:        {lufs_str}")
        print(f"  Dynamic Range:   {lra_str}")
        print(f"  Punch:           {c_num(f'{t.crest_factor_db:.1f}', Colors.BOLD_YELLOW, use_color)} {colorize('dB crest', Colors.DIM, use_color)}")

        sp_col = c_peak(t.sample_peak_dbfs, f"{t.sample_peak_dbfs:.2f} dB", use_color)
        tp_col = c_peak(t.true_peak_dbtp, f"{t.true_peak_dbtp:.2f} dB", use_color)
        print(f"  Peak Level:      {sp_col} (true peak: {tp_col})")

        st = t.stereo
        corr_col = c_corr(st.global_correlation, f"{st.global_correlation:+.2f}", use_color)
        worst_col = c_corr(st.min_window_correlation, f"{st.min_window_correlation:+.2f}", use_color)
        print(f"  Stereo Field:    {corr_col} correlation (worst dip: {worst_col})")

        inv_color = Colors.BOLD_GREEN if st.pct_negative_windows == 0.0 else (Colors.BOLD_YELLOW if st.pct_negative_windows < 10.0 else Colors.BOLD_RED)
        inv_str = colorize(f"{st.pct_negative_windows:.1f}% out-of-phase", inv_color, use_color)
        print(f"  Phase Health:    {inv_str}")

        drop_col = Colors.BOLD_GREEN if st.mono_loss_db > -0.5 else (Colors.BOLD_YELLOW if st.mono_loss_db > -2.0 else Colors.BOLD_RED)
        print(f"  Mono Drop:       {colorize(f'{st.mono_loss_db:.2f} dB', drop_col, use_color)} loss")
        ms_str = "Center only" if st.mid_side_ratio_db <= -90.0 else f"{st.mid_side_ratio_db:.1f} dB"
        print(f"  Side vs Center:  {c_num(ms_str, Colors.BOLD_YELLOW, use_color)}")

        if t.spectral_balance_db:
            print(f"  {colorize('Frequency Energy:', Colors.CYAN, use_color)}")
            band_map = {
                "Sub/Bass (20-250 Hz)": "Bass (20-250 Hz)",
                "Low-Mid (250-1000 Hz)": "Lows (250-1000 Hz)",
                "High-Mid (1-5 kHz)": "Highs (1-5 kHz)",
                "Air (5-20 kHz)": "Air (5-20 kHz)"
            }
            for k, v in t.spectral_balance_db.items():
                b_name = band_map.get(k, k)
                print(f"    * {colorize(f'{b_name:<20}', Colors.WHITE, use_color)}: {c_num(f'{v:.1f} dB', Colors.BOLD_YELLOW, use_color)}")
    else:
        print(f"\n{tag} {name_str}")
        t_str = f"{fmt_time(t.duration)} ({t.duration:.2f} s)"
        print(f"  Duration:         {colorize(t_str, Colors.BOLD_YELLOW, use_color)}")
        print(f"  Channel Format:   {colorize(t.stereo.channel_layout_desc, Colors.WHITE, use_color)}")
        print(f"  Estimated BPM:    {c_num(f'{t.bpm:.1f}', Colors.BOLD_YELLOW, use_color)}")
        print(f"  Estimated Key:    {colorize(t.key, Colors.BOLD_CYAN, use_color)}")

        lufs_str = f"{c_num(f'{t.lufs:.1f}', Colors.BOLD_YELLOW, use_color)} {colorize('LUFS', Colors.DIM, use_color)}" if t.lufs is not None else colorize("N/A", Colors.DIM, use_color)
        lra_str = f"{c_num(f'{t.lra:.1f}', Colors.BOLD_YELLOW, use_color)} {colorize('LU', Colors.DIM, use_color)}" if t.lra is not None else colorize("N/A", Colors.DIM, use_color)
        print(f"  Loudness (LUFS):  {lufs_str:<18} | LRA: {lra_str}")

        sp_col = c_peak(t.sample_peak_dbfs, f"{t.sample_peak_dbfs:.2f} dBFS", use_color)
        tp_col = c_peak(t.true_peak_dbtp, f"{t.true_peak_dbtp:.2f} dBTP", use_color)
        print(f"  Sample Peak:      {sp_col:<18} | True Peak: {tp_col}")
        print(f"  Dynamic Punch:    {c_num(f'{t.crest_factor_db:.2f} dB', Colors.BOLD_YELLOW, use_color)} {colorize('Crest Factor (True Peak / RMS)', Colors.DIM, use_color)}")

        st = t.stereo
        ms_str = "Mono" if st.mid_side_ratio_db <= -90.0 else f"{st.mid_side_ratio_db:.1f} dB"
        corr_col = c_corr(st.global_correlation, f"{st.global_correlation:+.2f}", use_color)
        print(f"  Stereo Field:     Global Corr: {corr_col} | M/S Energy Ratio: {c_num(ms_str, Colors.BOLD_YELLOW, use_color)}")

        worst_col = c_corr(st.min_window_correlation, f"{st.min_window_correlation:+.2f}", use_color)
        inv_color = Colors.BOLD_GREEN if st.pct_negative_windows == 0.0 else (Colors.BOLD_YELLOW if st.pct_negative_windows < 10.0 else Colors.BOLD_RED)
        inv_str = colorize(f"{st.pct_negative_windows:.1f}%", inv_color, use_color)
        print(f"  Phase Stability:  Min Window:  {worst_col} | Negative Windows: {inv_str}")

        if st.mono_loss_db < -0.2:
            print(f"  Mono Fold-Down:   {c_num(f'{st.mono_loss_db:.2f} dB', Colors.BOLD_YELLOW, use_color)} Power Loss upon mono sum")
        if st.mono_fallback_used:
            notice = colorize("  [NOTICE] Severe out-of-phase mono cancellation; Left channel was used for feature extraction.", Colors.BOLD_YELLOW, use_color)
            print(notice)

        if t.spectral_balance_db:
            print(f"  {colorize('4-Band Relative Power:', Colors.CYAN, use_color)}")
            for k, v in t.spectral_balance_db.items():
                print(f"    * {colorize(f'{k:<24}', Colors.WHITE, use_color)}: {c_num(f'{v:.1f} dB', Colors.BOLD_YELLOW, use_color)}")


def print_mix_deltas(
    a: TrackAnalysis,
    b: TrackAnalysis,
    mode: str = "precise",
    use_color: bool = True
) -> None:
    """Prints comparative mix deltas between Song A and Song B with rich metric color highlights."""
    rule = colorize("=" * 65, Colors.CYAN, use_color)

    if mode == "simple":
        print(f"\n{rule}")
        print(colorize("MIX COMPARISON (Song B vs. Song A)", Colors.BOLD_CYAN, use_color))
        print(rule)

        if a.lufs is not None and b.lufs is not None:
            delta_lufs = b.lufs - a.lufs
            if delta_lufs > 0.2:
                direction = colorize("Louder", Colors.BOLD_GREEN, use_color)
                d_str = c_num(f"{delta_lufs:+.1f} LU", Colors.BOLD_GREEN, use_color)
            elif delta_lufs < -0.2:
                direction = colorize("Quieter", Colors.BOLD_YELLOW, use_color)
                d_str = c_num(f"{delta_lufs:+.1f} LU", Colors.BOLD_YELLOW, use_color)
            else:
                direction = colorize("Matched", Colors.DIM, use_color)
                d_str = c_num(f"{delta_lufs:+.1f} LU", Colors.DIM, use_color)
            print(f"  Loudness:        {d_str} ({direction})")
        else:
            print("  Loudness:        N/A")

        if a.lra is not None and b.lra is not None:
            delta_lra = b.lra - a.lra
            spread = colorize("More dynamic", Colors.BOLD_GREEN, use_color) if delta_lra > 0.5 else (colorize("Tighter", Colors.BOLD_YELLOW, use_color) if delta_lra < -0.5 else colorize("Consistent", Colors.DIM, use_color))
            print(f"  Dynamic Range:   {c_delta(delta_lra, 'LU', enabled=use_color)} ({spread})")

        delta_crest = b.crest_factor_db - a.crest_factor_db
        crest_tag = colorize("Punchier", Colors.BOLD_GREEN, use_color) if delta_crest > 0.5 else (colorize("More compressed", Colors.BOLD_YELLOW, use_color) if delta_crest < -0.5 else colorize("Preserved", Colors.DIM, use_color))
        print(f"  Punch:           {c_delta(delta_crest, 'dB', enabled=use_color)} ({crest_tag})")

        delta_sample = b.sample_peak_dbfs - a.sample_peak_dbfs
        delta_true = b.true_peak_dbtp - a.true_peak_dbtp
        sp_sub = c_peak(b.sample_peak_dbfs, f"{b.sample_peak_dbfs:.2f} dB sample peak", use_color)
        tp_sub = c_peak(b.true_peak_dbtp, f"{b.true_peak_dbtp:.2f} dB inter-sample", use_color)
        print(f"  Peak Level:      {c_delta(delta_sample, 'dB', positive_good=False, enabled=use_color)} ({sp_sub})")
        print(f"  True Peak:       {c_delta(delta_true, 'dB', positive_good=False, enabled=use_color)} ({tp_sub})")

        st_b = b.stereo
        corr_b = c_corr(st_b.global_correlation, f"{st_b.global_correlation:+.2f}", use_color)
        worst_b = c_corr(st_b.min_window_correlation, f"{st_b.min_window_correlation:+.2f} worst", use_color)
        print(f"  Stereo Field:    {corr_b} correlation ({worst_b})")

        inv_color = Colors.BOLD_GREEN if st_b.pct_negative_windows == 0.0 else (Colors.BOLD_YELLOW if st_b.pct_negative_windows < 10.0 else Colors.BOLD_RED)
        print(f"  Phase Health:    {colorize(f'{st_b.pct_negative_windows:.1f}% out-of-phase', inv_color, use_color)}")

        drop_col = Colors.BOLD_GREEN if st_b.mono_loss_db > -0.5 else (Colors.BOLD_YELLOW if st_b.mono_loss_db > -2.0 else Colors.BOLD_RED)
        print(f"  Mono Drop:       {colorize(f'{st_b.mono_loss_db:.2f} dB', drop_col, use_color)} in mono")

        if a.stereo.mid_side_ratio_db > -90.0 and st_b.mid_side_ratio_db > -90.0:
            delta_ms = st_b.mid_side_ratio_db - a.stereo.mid_side_ratio_db
            w_dir = colorize("Wider", Colors.BOLD_CYAN, use_color) if delta_ms > 0.5 else (colorize("Narrower", Colors.BOLD_YELLOW, use_color) if delta_ms < -0.5 else colorize("Unchanged", Colors.DIM, use_color))
            print(f"  Stereo Image:    {c_num(f'{delta_ms:+.1f} dB', Colors.BOLD_CYAN, use_color)} side energy ({w_dir})")

        if a.spectral_balance_db and b.spectral_balance_db:
            print(f"  {colorize('Frequency Shifts:', Colors.CYAN, use_color)}")
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
                        status = colorize("Boosted", Colors.BOLD_GREEN, use_color)
                        d_col = Colors.BOLD_GREEN
                    elif delta_p < -0.5:
                        status = colorize("Cut", Colors.BOLD_YELLOW, use_color)
                        d_col = Colors.BOLD_YELLOW
                    else:
                        status = colorize("Same", Colors.DIM, use_color)
                        d_col = Colors.DIM
                    short_name = band_map.get(band_name, band_name)
                    print(f"    * {colorize(f'{short_name:<16}', Colors.WHITE, use_color)}: {c_num(f'{delta_p:+.1f} dB', d_col, use_color)} ({status})")
    else:
        print(f"\n{rule}")
        print(colorize("MIX AND MASTERING AUDIT (Song B vs. Song A)", Colors.BOLD_CYAN, use_color))
        print(rule)

        if a.lufs is not None and b.lufs is not None:
            delta_lufs = b.lufs - a.lufs
            if delta_lufs > 0.2:
                indicator = colorize("Louder", Colors.BOLD_GREEN, use_color)
                d_str = c_num(f"{delta_lufs:+.1f} LU", Colors.BOLD_GREEN, use_color)
            elif delta_lufs < -0.2:
                indicator = colorize("Quieter", Colors.BOLD_YELLOW, use_color)
                d_str = c_num(f"{delta_lufs:+.1f} LU", Colors.BOLD_YELLOW, use_color)
            else:
                indicator = colorize("Matched loudness", Colors.DIM, use_color)
                d_str = c_num(f"{delta_lufs:+.1f} LU", Colors.DIM, use_color)
            print(f"  Loudness:         Song B is {d_str} ({indicator})")
        else:
            print("  Loudness:         N/A (LUFS not available)")

        if a.lra is not None and b.lra is not None:
            delta_lra = b.lra - a.lra
            indicator = colorize("More dynamic variation", Colors.BOLD_GREEN, use_color) if delta_lra > 0.5 else (colorize("More compressed", Colors.BOLD_YELLOW, use_color) if delta_lra < -0.5 else colorize("Consistent range", Colors.DIM, use_color))
            print(f"  Loudness Range:   Song B is {c_delta(delta_lra, 'LU', enabled=use_color)} ({indicator})")

        delta_crest = b.crest_factor_db - a.crest_factor_db
        crest_indicator = colorize("More dynamic punch", Colors.BOLD_GREEN, use_color) if delta_crest > 0.5 else (colorize("More limited/compressed", Colors.BOLD_YELLOW, use_color) if delta_crest < -0.5 else colorize("Preserved dynamics", Colors.DIM, use_color))
        print(f"  Dynamic Punch:    {c_delta(delta_crest, 'dB', enabled=use_color)} Crest Factor ({crest_indicator})")

        delta_sample = b.sample_peak_dbfs - a.sample_peak_dbfs
        delta_true = b.true_peak_dbtp - a.true_peak_dbtp
        sp_col = c_peak(b.sample_peak_dbfs, f"{b.sample_peak_dbfs:.2f} dBFS", use_color)
        d_sp_col = c_delta(delta_sample, "dBFS", positive_good=False, enabled=use_color)
        print(f"  Sample Peak:      Song B is {sp_col} (Delta: {d_sp_col})")
        if b.sample_peak_dbfs > 0.0:
            print(colorize("  [ALERT] Decoded audio file contains floating-point samples exceeding 0.0 dBFS.", Colors.BOLD_RED, use_color))

        tp_col = c_peak(b.true_peak_dbtp, f"{b.true_peak_dbtp:.2f} dBTP", use_color)
        d_tp_col = c_delta(delta_true, "dBTP", positive_good=False, enabled=use_color)
        print(f"  True Peak:        Song B is {tp_col} (Delta: {d_tp_col})")
        if b.true_peak_dbtp > 0.0 and b.sample_peak_dbfs <= 0.0:
            print(colorize("  [WARNING] Inter-sample true-peak overshoot detected (> 0.0 dBTP). Risk of DAC/lossy clipping.", Colors.BOLD_YELLOW, use_color))
        elif b.true_peak_dbtp > -1.0 and b.true_peak_dbtp <= 0.0:
            print(colorize("  [NOTICE] True peak exceeds standard streaming ceiling (-1.0 dBTP target margin).", Colors.BOLD_CYAN, use_color))

        st_b = b.stereo
        corr_b = c_corr(st_b.global_correlation, f"{st_b.global_correlation:+.2f}", use_color)
        worst_b = c_corr(st_b.min_window_correlation, f"{st_b.min_window_correlation:+.2f}", use_color)
        print(f"\n  Stereo Correlation: {corr_b} (Worst Window: {worst_b})")

        if st_b.pct_negative_windows > 15.0:
            print(colorize(f"  [CRITICAL] Severe mono cancellation: {st_b.pct_negative_windows:.1f}% of active windows have negative correlation.", Colors.BOLD_RED, use_color))
        elif st_b.pct_negative_windows > 5.0:
            print(colorize(f"  [WARNING] Out-of-phase material: {st_b.pct_negative_windows:.1f}% negative correlation windows.", Colors.BOLD_YELLOW, use_color))

        if st_b.mono_loss_db < -3.0:
            print(colorize(f"  [WARNING] Severe mono fold-down attenuation: {st_b.mono_loss_db:.1f} dB loss when collapsed to mono.", Colors.BOLD_YELLOW, use_color))

        if a.stereo.mid_side_ratio_db > -90.0 and st_b.mid_side_ratio_db > -90.0:
            delta_ms = st_b.mid_side_ratio_db - a.stereo.mid_side_ratio_db
            width_indicator = colorize("Wider", Colors.BOLD_CYAN, use_color) if delta_ms > 0.5 else (colorize("Narrower", Colors.BOLD_YELLOW, use_color) if delta_ms < -0.5 else colorize("Unchanged", Colors.DIM, use_color))
            print(f"  Stereo Width:     Side/Mid energy changed by {c_num(f'{delta_ms:+.1f} dB', Colors.BOLD_CYAN, use_color)} ({width_indicator})")

        if a.spectral_balance_db and b.spectral_balance_db:
            print(f"\n  {colorize('4-Band Spectral Balance Shift (Relative Share):', Colors.CYAN, use_color)}")
            for band_name in a.spectral_balance_db:
                if band_name in b.spectral_balance_db:
                    delta_p = b.spectral_balance_db[band_name] - a.spectral_balance_db[band_name]
                    if delta_p > 0.5:
                        indicator = colorize("Boosted", Colors.BOLD_GREEN, use_color)
                        d_col = Colors.BOLD_GREEN
                    elif delta_p < -0.5:
                        indicator = colorize("Cut", Colors.BOLD_YELLOW, use_color)
                        d_col = Colors.BOLD_YELLOW
                    else:
                        indicator = colorize("Even", Colors.DIM, use_color)
                        d_col = Colors.DIM
                    print(f"    * {colorize(f'{band_name:<24}', Colors.WHITE, use_color)}: {c_num(f'{delta_p:+.1f} dB', d_col, use_color)} [{indicator}]")


def print_snippet_report(
    a: TrackAnalysis,
    b: TrackAnalysis,
    matches: List[MatchedSnippet],
    mode: str = "precise",
    use_color: bool = True
) -> None:
    """Prints snippet match list and arrangement breakdown with complete color highlights."""
    rule = colorize("=" * 65, Colors.CYAN, use_color)
    subrule = colorize("-" * 65, Colors.CYAN, use_color)
    header_shared = "SHARED SECTIONS" if mode == "simple" else "SHARED SNIPPETS AND SAMPLES"

    print(f"\n{rule}")
    print(colorize(header_shared, Colors.BOLD_CYAN, use_color))
    print(rule)

    if not matches:
        print("  No shared musical snippets detected above threshold.")
    else:
        if mode == "simple":
            hdr = f"{'Song A':<18}{'Song B':<18}{'Length':<10}{'Match':<11}{'Melody':<10}{'Tone'}"
        else:
            hdr = f"{'Song A Location':<18}{'Song B Location':<18}{'Duration':<10}{'Composite':<11}{'Harmonic':<10}{'Timbral'}"
        print(colorize(hdr, Colors.BOLD_WHITE, use_color))
        print(colorize("-" * 75, Colors.DIM, use_color))

        for m in matches:
            loc_a = colorize(f"{fmt_time(m.a_start)}-{fmt_time(m.a_end)}", Colors.CYAN, use_color)
            loc_b = colorize(f"{fmt_time(m.b_start)}-{fmt_time(m.b_end)}", Colors.MAGENTA, use_color)
            dur = c_num(f"{m.duration_a:.1f} s", Colors.BOLD_YELLOW, use_color)

            comp_str = c_pct(m.composite_similarity * 100, use_color)
            harm_str = c_pct(m.harmonic_similarity * 100, use_color)
            timb_str = c_num(f"{m.timbral_similarity * 100:.1f}%", Colors.WHITE, use_color)

            # Pad considering invisible ANSI escape codes
            pad_a = 18 + (len(loc_a) - len(f"{fmt_time(m.a_start)}-{fmt_time(m.a_end)}"))
            pad_b = 18 + (len(loc_b) - len(f"{fmt_time(m.b_start)}-{fmt_time(m.b_end)}"))
            pad_dur = 10 + (len(dur) - len(f"{m.duration_a:.1f} s"))
            pad_comp = 11 + (len(comp_str) - len(f"{m.composite_similarity * 100:.1f}%"))
            pad_harm = 10 + (len(harm_str) - len(f"{m.harmonic_similarity * 100:.1f}%"))

            print(f"{loc_a:<{pad_a}}{loc_b:<{pad_b}}{dur:<{pad_dur}}{comp_str:<{pad_comp}}{harm_str:<{pad_harm}}{timb_str}")

    sec_a, sec_b, pct_a, pct_b, gaps_a, gaps_b = compute_verified_coverage(a.duration, b.duration, matches)

    if mode == "simple":
        header_audit = "ARRANGEMENT BREAKDOWN"
        print(f"\n{subrule}")
        print(colorize(header_audit, Colors.BOLD_CYAN, use_color))
        print(subrule)
        sec_a_col = c_num(f"{sec_a:.1f} s", Colors.BOLD_YELLOW, use_color)
        pct_a_col = c_pct(pct_a, use_color)
        sec_b_col = c_num(f"{sec_b:.1f} s", Colors.BOLD_YELLOW, use_color)
        pct_b_col = c_pct(pct_b, use_color)
        print(f"  Song A in Song B:  {sec_a_col} ({pct_a_col} of Song A)")
        print(f"  Song B in Song A:  {sec_b_col} ({pct_b_col} of Song B)")

        if gaps_a and (not matches or pct_a < 99.0):
            cut_label = colorize("Cut from Song A (missing in Song B):", Colors.BOLD_YELLOW, use_color)
            print(f"\n  {cut_label}")
            for s, e in gaps_a:
                bullet = colorize("*", Colors.YELLOW, use_color)
                t_range = colorize(f"{fmt_time(s)} - {fmt_time(e)}", Colors.BOLD_YELLOW, use_color)
                d_str = colorize(f"({e - s:.1f} s)", Colors.DIM, use_color)
                print(f"     {bullet} {t_range} {d_str}")

        if gaps_b and (not matches or pct_b < 99.0):
            added_label = colorize("New in Song B (added / extended):", Colors.BOLD_GREEN, use_color)
            print(f"\n  {added_label}")
            for s, e in gaps_b:
                bullet = colorize("*", Colors.GREEN, use_color)
                t_range = colorize(f"{fmt_time(s)} - {fmt_time(e)}", Colors.BOLD_GREEN, use_color)
                d_str = colorize(f"({e - s:.1f} s)", Colors.DIM, use_color)
                print(f"     {bullet} {t_range} {d_str}")
    else:
        header_audit = "STRUCTURAL EDIT AUDIT (Verified Bipartite Allocation)"
        print(f"\n{subrule}")
        print(colorize(header_audit, Colors.BOLD_CYAN, use_color))
        print(subrule)
        sec_a_col = c_num(f"{sec_a:.1f} s", Colors.BOLD_YELLOW, use_color)
        pct_a_col = c_pct(pct_a, use_color)
        sec_b_col = c_num(f"{sec_b:.1f} s", Colors.BOLD_YELLOW, use_color)
        pct_b_col = c_pct(pct_b, use_color)
        print(f"  Song A in Song B (shared duration):  {sec_a_col} ({pct_a_col})")
        print(f"  Song B in Song A (shared duration):  {sec_b_col} ({pct_b_col})")

        if gaps_a and (not matches or pct_a < 99.0):
            cut_label = colorize("[CUT] Unique to Song A (absent in Song B):", Colors.BOLD_YELLOW, use_color)
            print(f"\n  {cut_label}")
            for s, e in gaps_a:
                bullet = colorize("*", Colors.YELLOW, use_color)
                t_range = colorize(f"{fmt_time(s)} - {fmt_time(e)}", Colors.BOLD_YELLOW, use_color)
                d_str = colorize(f"({e - s:.1f} s)", Colors.DIM, use_color)
                print(f"     {bullet} {t_range} {d_str}")

        if gaps_b and (not matches or pct_b < 99.0):
            added_label = colorize("[ADDED] Unique to Song B (new / extended material):", Colors.BOLD_GREEN, use_color)
            print(f"\n  {added_label}")
            for s, e in gaps_b:
                bullet = colorize("*", Colors.GREEN, use_color)
                t_range = colorize(f"{fmt_time(s)} - {fmt_time(e)}", Colors.BOLD_GREEN, use_color)
                d_str = colorize(f"({e - s:.1f} s)", Colors.DIM, use_color)
                print(f"     {bullet} {t_range} {d_str}")


def generate_html_report(
    a: TrackAnalysis,
    b: TrackAnalysis,
    matches: List[MatchedSnippet],
    output_path: Path,
    mode: str = "precise",
    auto_open: bool = False
) -> None:
    """
    Generates an interactive HTML comparison report with native clickable
    <details><summary> accordions to expand and collapse shared sections.
    """
    sec_a, sec_b, pct_a, pct_b, gaps_a, gaps_b = compute_verified_coverage(a.duration, b.duration, matches)

    html_lines = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "  <meta charset='utf-8'>",
        "  <meta name='viewport' content='width=device-width, initial-scale=1.0'>",
        f"  <title>audiodelta: {a.path.name} vs {b.path.name}</title>",
        "  <style>",
        "    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 24px; }",
        "    .container { max-width: 960px; margin: 0 auto; }",
        "    h1, h2, h3 { color: #38bdf8; margin-top: 0; }",
        "    .card { background: #1e293b; border-radius: 8px; padding: 20px; margin-bottom: 20px; border: 1px solid #334155; }",
        "    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }",
        "    details { background: #1e293b; border-radius: 8px; margin-bottom: 16px; border: 1px solid #334155; padding: 12px 16px; }",
        "    summary { font-weight: bold; cursor: pointer; color: #38bdf8; outline: none; user-select: none; font-size: 1.1em; }",
        "    summary:hover { color: #7dd3fc; }",
        "    table { width: 100%; border-collapse: collapse; margin-top: 12px; }",
        "    th, td { text-align: left; padding: 10px; border-bottom: 1px solid #334155; font-size: 0.95em; }",
        "    th { color: #94a3b8; }",
        "    .badge { display: inline-block; padding: 3px 8px; border-radius: 4px; font-weight: bold; font-size: 0.85em; }",
        "    .badge-green { background: #065f46; color: #34d399; }",
        "    .badge-yellow { background: #854d0e; color: #fde047; }",
        "    ul { margin: 8px 0; padding-left: 20px; }",
        "    li { margin-bottom: 4px; }",
        "  </style>",
        "</head>",
        "<body>",
        "  <div class='container'>",
        f"    <h1>audiodelta Analysis Report</h1>",
        f"    <p style='color:#94a3b8'>Comparing <strong>{a.path.name}</strong> (Song A) vs <strong>{b.path.name}</strong> (Song B)</p>",
        "    <div class='grid'>",
        "      <div class='card'>",
        f"        <h3>Song A: {a.path.name}</h3>",
        f"        <p><strong>Length:</strong> {fmt_time(a.duration)} ({a.duration:.1f}s)<br>",
        f"        <strong>Channels:</strong> {a.stereo.channel_layout_desc}<br>",
        f"        <strong>Tempo:</strong> {a.bpm:.1f} BPM | <strong>Key:</strong> {_format_key_simple(a.key)}<br>",
        f"        <strong>Loudness:</strong> {a.lufs:.1f} LUFS | <strong>Range:</strong> {a.lra:.1f} LU<br>",
        f"        <strong>Peak:</strong> {a.sample_peak_dbfs:.2f} dBFS | <strong>True Peak:</strong> {a.true_peak_dbtp:.2f} dBTP<br>",
        f"        <strong>Punch:</strong> {a.crest_factor_db:.1f} dB crest</p>",
        "      </div>",
        "      <div class='card'>",
        f"        <h3>Song B: {b.path.name}</h3>",
        f"        <p><strong>Length:</strong> {fmt_time(b.duration)} ({b.duration:.1f}s)<br>",
        f"        <strong>Channels:</strong> {b.stereo.channel_layout_desc}<br>",
        f"        <strong>Tempo:</strong> {b.bpm:.1f} BPM | <strong>Key:</strong> {_format_key_simple(b.key)}<br>",
        f"        <strong>Loudness:</strong> {b.lufs:.1f} LUFS | <strong>Range:</strong> {b.lra:.1f} LU<br>",
        f"        <strong>Peak:</strong> {b.sample_peak_dbfs:.2f} dBFS | <strong>True Peak:</strong> {b.true_peak_dbtp:.2f} dBTP<br>",
        f"        <strong>Punch:</strong> {b.crest_factor_db:.1f} dB crest</p>",
        "      </div>",
        "    </div>",
        "    <details open>",
        "      <summary>Arrangement Overview (Click to collapse/expand)</summary>",
        f"      <p style='margin-top:12px'><strong>Song A in Song B:</strong> {sec_a:.1f} s ({pct_a:.1f}% of Song A)<br>",
        f"      <strong>Song B in Song A:</strong> {sec_b:.1f} s ({pct_b:.1f}% of Song B)</p>",
    ]

    if gaps_a:
        html_lines.append("      <h4 style='color:#fde047;margin-bottom:6px'>Cut from Song A (missing in Song B):</h4><ul>")
        for s, e in gaps_a:
            html_lines.append(f"        <li>{fmt_time(s)} - {fmt_time(e)} ({e - s:.1f} s)</li>")
        html_lines.append("      </ul>")

    if gaps_b:
        html_lines.append("      <h4 style='color:#34d399;margin-bottom:6px'>New in Song B (added / extended):</h4><ul>")
        for s, e in gaps_b:
            html_lines.append(f"        <li>{fmt_time(s)} - {fmt_time(e)} ({e - s:.1f} s)</li>")
        html_lines.append("      </ul>")

    html_lines.append("    </details>")

    section_count = len(matches)
    html_lines.append(f"    <details open>")
    html_lines.append(f"      <summary>Shared Sections ({section_count} detected &bull; Click to collapse/expand)</summary>")

    if not matches:
        html_lines.append("      <p style='margin-top:12px;color:#94a3b8'>No shared musical snippets detected above threshold.</p>")
    else:
        html_lines.append("      <table>")
        html_lines.append("        <thead><tr><th>Song A</th><th>Song B</th><th>Length</th><th>Match Score</th><th>Melody Match</th><th>Tone Match</th></tr></thead>")
        html_lines.append("        <tbody>")
        for m in matches:
            comp_val = m.composite_similarity * 100
            badge_cls = "badge-green" if comp_val >= 90.0 else "badge-yellow"
            html_lines.append(f"          <tr>")
            html_lines.append(f"            <td>{fmt_time(m.a_start)} - {fmt_time(m.a_end)}</td>")
            html_lines.append(f"            <td>{fmt_time(m.b_start)} - {fmt_time(m.b_end)}</td>")
            html_lines.append(f"            <td>{m.duration_a:.1f}s</td>")
            html_lines.append(f"            <td><span class='badge {badge_cls}'>{comp_val:.1f}%</span></td>")
            html_lines.append(f"            <td>{m.harmonic_similarity * 100:.1f}%</td>")
            html_lines.append(f"            <td>{m.timbral_similarity * 100:.1f}%</td>")
            html_lines.append(f"          </tr>")
        html_lines.append("        </tbody>")
        html_lines.append("      </table>")

    html_lines.append("    </details>")
    html_lines.append("  </div>")
    html_lines.append("</body>")
    html_lines.append("</html>")

    output_path.write_text("\n".join(html_lines), encoding="utf-8")
    if auto_open:
        try:
            webbrowser.open(output_path.as_uri())
        except Exception:
            pass


def print_full_report(
    a: TrackAnalysis,
    b: TrackAnalysis,
    matches: List[MatchedSnippet],
    mode: str = "precise",
    use_color: bool = True
) -> None:
    """Prints the complete comparison report with intuitive wording and vibrant coloring."""
    print_track_card("SONG A", a, mode=mode, use_color=use_color)
    print_track_card("SONG B", b, mode=mode, use_color=use_color)
    print_mix_deltas(a, b, mode=mode, use_color=use_color)
    print_snippet_report(a, b, matches, mode=mode, use_color=use_color)
