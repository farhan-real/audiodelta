"""
Dual-mode console reporting for audiodelta supporting simple/precise wording,
ANSI terminal coloring, and interactive clickable HTML report generation.
"""

from pathlib import Path
from typing import List, Optional
import webbrowser

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
        print(f"  Length:          {fmt_time(t.duration)} ({t.duration:.1f}s)")
        ch_desc = "Mono (1 channel)" if t.stereo.channel_count == 1 else f"Stereo ({t.stereo.channel_count} channels)"
        print(f"  Channels:        {ch_desc}")
        print(f"  Tempo:           {t.bpm:.1f} BPM")
        print(f"  Key:             {_format_key_simple(t.key)}")
        lufs_str = f"{t.lufs:.1f} LUFS" if t.lufs is not None else "N/A"
        lra_str = f"{t.lra:.1f} LU" if t.lra is not None else "N/A"
        print(f"  Loudness:        {lufs_str}")
        print(f"  Dynamic Range:   {lra_str}")
        print(f"  Punch:           {t.crest_factor_db:.1f} dB crest")
        print(f"  Peak Level:      {t.sample_peak_dbfs:.2f} dB (true peak: {t.true_peak_dbtp:.2f} dB)")

        st = t.stereo
        print(f"  Stereo Field:    {st.global_correlation:+.2f} correlation (worst dip: {st.min_window_correlation:+.2f})")
        inv_str = f"{st.pct_negative_windows:.1f}% out-of-phase"
        if st.pct_negative_windows > 10.0:
            inv_str = colorize(inv_str, Colors.YELLOW, use_color)
        print(f"  Phase Health:    {inv_str}")
        print(f"  Mono Drop:       {st.mono_loss_db:.2f} dB loss")
        ms_str = "Center only" if st.mid_side_ratio_db <= -90.0 else f"{st.mid_side_ratio_db:.1f} dB"
        print(f"  Side vs Center:  {ms_str}")

        if t.spectral_balance_db:
            print("  Frequency Energy:")
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
            print(f"  Loudness:        {delta_lufs:+.1f} LU ({direction})")
        else:
            print("  Loudness:        N/A")

        if a.lra is not None and b.lra is not None:
            delta_lra = b.lra - a.lra
            spread = "More dynamic" if delta_lra > 0.5 else "Tighter" if delta_lra < -0.5 else "Consistent"
            print(f"  Dynamic Range:   {delta_lra:+.1f} LU ({spread})")

        delta_crest = b.crest_factor_db - a.crest_factor_db
        crest_tag = "Punchier" if delta_crest > 0.5 else "More compressed" if delta_crest < -0.5 else "Preserved"
        print(f"  Punch:           {delta_crest:+.1f} dB ({crest_tag})")

        delta_sample = b.sample_peak_dbfs - a.sample_peak_dbfs
        delta_true = b.true_peak_dbtp - a.true_peak_dbtp
        print(f"  Peak Level:      {delta_sample:+.2f} dB ({b.sample_peak_dbfs:.2f} dB sample peak)")
        print(f"  True Peak:       {delta_true:+.2f} dB ({b.true_peak_dbtp:.2f} dB inter-sample)")

        st_b = b.stereo
        print(f"  Stereo Field:    {st_b.global_correlation:+.2f} correlation ({st_b.min_window_correlation:+.2f} worst)")
        print(f"  Phase Health:    {st_b.pct_negative_windows:.1f}% out-of-phase")
        print(f"  Mono Drop:       {st_b.mono_loss_db:.2f} dB in mono")

        if a.stereo.mid_side_ratio_db > -90.0 and st_b.mid_side_ratio_db > -90.0:
            delta_ms = st_b.mid_side_ratio_db - a.stereo.mid_side_ratio_db
            w_dir = "Wider" if delta_ms > 0.5 else "Narrower" if delta_ms < -0.5 else "Unchanged"
            print(f"  Stereo Image:    {delta_ms:+.1f} dB side energy ({w_dir})")

        if a.spectral_balance_db and b.spectral_balance_db:
            print("  Frequency Shifts:")
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
                        status = colorize("Boosted", Colors.GREEN, use_color)
                    elif delta_p < -0.5:
                        status = colorize("Cut", Colors.YELLOW, use_color)
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
    use_color: bool = True
) -> None:
    """Prints snippet match list and arrangement breakdown."""
    rule = colorize("=" * 65, Colors.CYAN, use_color)
    subrule = colorize("-" * 65, Colors.CYAN, use_color)
    header_shared = "SHARED SECTIONS" if mode == "simple" else "SHARED SNIPPETS AND SAMPLES"

    print(f"\n{rule}")
    print(colorize(header_shared, Colors.BOLD_CYAN, use_color))
    print(rule)

    total_matches = len(matches)

    if not matches:
        print("  No shared musical snippets detected above threshold.")
    else:
        if mode == "simple":
            print(f"{'Song A':<18}{'Song B':<18}{'Length':<10}{'Match':<11}{'Melody':<10}{'Tone'}")
        else:
            print(f"{'Song A Location':<18}{'Song B Location':<18}{'Duration':<10}{'Composite':<11}{'Harmonic':<10}{'Timbral'}")
        print("-" * 75)

        for m in matches:
            loc_a = f"{fmt_time(m.a_start)}-{fmt_time(m.a_end)}"
            loc_b = f"{fmt_time(m.b_start)}-{fmt_time(m.b_end)}"
            dur = f"{m.duration_a:.1f} s"

            comp_val = m.composite_similarity * 100
            comp_str = f"{comp_val:.1f}%"
            if comp_val >= 90.0:
                comp_str = colorize(comp_str, Colors.BOLD_GREEN, use_color)
            elif comp_val >= 75.0:
                comp_str = colorize(comp_str, Colors.YELLOW, use_color)

            harm = f"{m.harmonic_similarity * 100:.1f}%"
            timb = f"{m.timbral_similarity * 100:.1f}%"
            print(f"{loc_a:<18}{loc_b:<18}{dur:<10}{comp_str:<20}{harm:<10}{timb}")

    sec_a, sec_b, pct_a, pct_b, gaps_a, gaps_b = compute_verified_coverage(a.duration, b.duration, matches)

    if mode == "simple":
        header_audit = "ARRANGEMENT BREAKDOWN"
        print(f"\n{subrule}")
        print(colorize(header_audit, Colors.BOLD_CYAN, use_color))
        print(subrule)
        print(f"  Song A in Song B:  {sec_a:.1f} s ({pct_a:.1f}% of Song A)")
        print(f"  Song B in Song A:  {sec_b:.1f} s ({pct_b:.1f}% of Song B)")

        if gaps_a and (not matches or pct_a < 99.0):
            cut_label = "Cut from Song A (missing in Song B):"
            print(colorize(f"\n  {cut_label}", Colors.YELLOW, use_color))
            for s, e in gaps_a:
                print(f"     * {fmt_time(s)} - {fmt_time(e)} ({e - s:.1f} s)")

        if gaps_b and (not matches or pct_b < 99.0):
            added_label = "New in Song B (added / extended):"
            print(colorize(f"\n  {added_label}", Colors.GREEN, use_color))
            for s, e in gaps_b:
                print(f"     * {fmt_time(s)} - {fmt_time(e)} ({e - s:.1f} s)")
    else:
        header_audit = "STRUCTURAL EDIT AUDIT (Verified Bipartite Allocation)"
        print(f"\n{subrule}")
        print(colorize(header_audit, Colors.BOLD_CYAN, use_color))
        print(subrule)
        print(f"  Song A in Song B (shared duration):  {sec_a:.1f} s ({pct_a:.1f}%)")
        print(f"  Song B in Song A (shared duration):  {sec_b:.1f} s ({pct_b:.1f}%)")

        if gaps_a and (not matches or pct_a < 99.0):
            cut_label = "[CUT] Unique to Song A (absent in Song B):"
            print(colorize(f"\n  {cut_label}", Colors.YELLOW, use_color))
            for s, e in gaps_a:
                print(f"     * {fmt_time(s)} - {fmt_time(e)} ({e - s:.1f} s)")

        if gaps_b and (not matches or pct_b < 99.0):
            added_label = "[ADDED] Unique to Song B (new / extended material):"
            print(colorize(f"\n  {added_label}", Colors.GREEN, use_color))
            for s, e in gaps_b:
                print(f"     * {fmt_time(s)} - {fmt_time(e)} ({e - s:.1f} s)")


def generate_html_report(
    a: TrackAnalysis,
    b: TrackAnalysis,
    matches: List[MatchedSnippet],
    output_path: Path,
    mode: str = "precise",
    auto_open: bool = False
) -> None:
    """
    Generates an interactive, standalone HTML comparison report with native
    clickable <details><summary> accordions to expand and collapse shared sections.
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
        "    .badge-blue { background: #0369a1; color: #7dd3fc; }",
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
        f"      <p style='margin-top:12px'><strong>Song A in Song B:</strong> {sec_a:.1f} s ({pct_a:.1f}% shared)<br>",
        f"      <strong>Song B in Song A:</strong> {sec_b:.1f} s ({pct_b:.1f}% shared)</p>",
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

    # Clickable Collapsible Shared Sections
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
    """Prints the complete comparison report with intuitive wording and optional coloring."""
    print_track_card("SONG A", a, mode=mode, use_color=use_color)
    print_track_card("SONG B", b, mode=mode, use_color=use_color)
    print_mix_deltas(a, b, mode=mode, use_color=use_color)
    print_snippet_report(a, b, matches, mode=mode, use_color=use_color)