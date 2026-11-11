"""
Bipartite interval coverage allocation guaranteeing non-overlapping attribution.
"""

from typing import List, Tuple
from core.constants import EPS
from core.models import MatchedSnippet
from core.utils import subtract_intervals


def compute_verified_coverage(
    a_dur: float,
    b_dur: float,
    matches: List[MatchedSnippet]
) -> Tuple[float, float, float, float, List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    Allocates structural coverage through analytic interval subtraction.
    Guarantees that mapped intervals in A and B are strictly mutual, non-overlapping,
    and verified without interpolating across gaps.
    """
    if not matches or a_dur <= 0 or b_dur <= 0:
        return 0.0, 0.0, 0.0, 0.0, [(0.0, max(0.0, a_dur))], [(0.0, max(0.0, b_dur))]

    claimed_a: List[Tuple[float, float]] = []
    claimed_b: List[Tuple[float, float]] = []

    sorted_matches = sorted(matches, key=lambda x: (x.composite_similarity * x.duration_a), reverse=True)

    for m in sorted_matches:
        r = m.speed_ratio
        c = m.b_start - r * m.a_start

        unclaimed_a = subtract_intervals((m.a_start, m.a_end), claimed_a)

        for ua_s, ua_e in unclaimed_a:
            ub_s = r * ua_s + c
            ub_e = r * ua_e + c

            valid_b = subtract_intervals((ub_s, ub_e), claimed_b)

            for vb_s, vb_e in valid_b:
                va_s = (vb_s - c) / r
                va_e = (vb_e - c) / r

                if (va_e - va_s) >= 0.5:
                    claimed_a.append((va_s, va_e))
                    claimed_b.append((vb_s, vb_e))

                    claimed_a = sorted(claimed_a, key=lambda x: x[0])
                    merged_a = [claimed_a[0]]
                    for s, e in claimed_a[1:]:
                        if s <= merged_a[-1][1]:
                            merged_a[-1] = (merged_a[-1][0], max(merged_a[-1][1], e))
                        else:
                            merged_a.append((s, e))
                    claimed_a = merged_a

                    claimed_b = sorted(claimed_b, key=lambda x: x[0])
                    merged_b = [claimed_b[0]]
                    for s, e in claimed_b[1:]:
                        if s <= merged_b[-1][1]:
                            merged_b[-1] = (merged_b[-1][0], max(merged_b[-1][1], e))
                        else:
                            merged_b.append((s, e))
                    claimed_b = merged_b

    covered_sec_a = sum(e - s for s, e in claimed_a)
    covered_sec_b = sum(e - s for s, e in claimed_b)

    pct_a = min(100.0, (covered_sec_a / max(a_dur, EPS)) * 100.0)
    pct_b = min(100.0, (covered_sec_b / max(b_dur, EPS)) * 100.0)

    gaps_a = [g for g in subtract_intervals((0.0, a_dur), claimed_a) if (g[1] - g[0]) >= 1.5]
    gaps_b = [g for g in subtract_intervals((0.0, b_dur), claimed_b) if (g[1] - g[0]) >= 1.5]

    return covered_sec_a, covered_sec_b, pct_a, pct_b, gaps_a, gaps_b