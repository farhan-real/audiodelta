"""
Utility functions for time formatting and 1D interval subtraction arithmetic.
"""

from typing import List, Tuple


def fmt_time(seconds: float) -> str:
    """Formats a duration in seconds into MM:SS.ss string format."""
    seconds = max(0.0, float(seconds))
    total_hundredths = int(round(seconds * 100))
    minutes = total_hundredths // 6000
    secs = (total_hundredths % 6000) / 100.0
    return f"{minutes:02d}:{secs:05.2f}"


def subtract_intervals(
    interval: Tuple[float, float],
    claimed: List[Tuple[float, float]]
) -> List[Tuple[float, float]]:
    """
    Subtracts a list of claimed disjoint intervals from a single candidate interval.
    Returns the remaining disjoint sub-intervals longer than 0.05 seconds.
    """
    remaining = [interval]
    for cs, ce in claimed:
        next_rem = []
        for rs, re in remaining:
            if ce <= rs or cs >= re:
                next_rem.append((rs, re))
            else:
                if cs > rs:
                    next_rem.append((rs, cs))
                if ce < re:
                    next_rem.append((ce, re))
        remaining = next_rem
        if not remaining:
            break
    return [iv for iv in remaining if (iv[1] - iv[0]) > 0.05]