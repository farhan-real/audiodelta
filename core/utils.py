"""
Utility functions for time formatting, interval subtraction, and terminal progress bars.
"""

import sys
from typing import List, Tuple


class ProgressBar:
    """
    Lightweight, dependency-free terminal progress bar with TTY auto-detection.
    """

    def __init__(
        self,
        total: int,
        prefix: str = "",
        length: int = 26,
        file=sys.stdout
    ) -> None:
        self.total = max(1, total)
        self.prefix = prefix
        self.length = length
        self.file = file
        self.is_tty = hasattr(file, "isatty") and file.isatty()

    def update(self, current: int, suffix: str = "") -> None:
        """Draws or refreshes the progress bar."""
        if not self.is_tty:
            return
        current = min(self.total, max(0, current))
        fraction = current / self.total
        filled = int(round(self.length * fraction))
        bar = "#" * filled + "-" * (self.length - filled)
        pct = fraction * 100.0
        line = f"\r{self.prefix} [{bar}] {pct:5.1f}% {suffix}"
        self.file.write(line)
        self.file.flush()

    def finish(self, message: str = "") -> None:
        """Clears the progress bar line or writes a completion message."""
        if not self.is_tty:
            return
        padding = " " * (self.length + len(self.prefix) + 25)
        self.file.write(f"\r{padding}\r")
        if message:
            self.file.write(f"{message}\n")
        self.file.flush()


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