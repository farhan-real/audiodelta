"""
Utility functions for time formatting, interval subtraction, progress bars, and ANSI terminal colors.
"""

import os
import sys
from contextlib import contextmanager
from typing import Generator, List, Optional, Tuple


class Colors:
    """Standard ANSI color and style escape sequences."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BOLD_RED = "\033[1;31m"
    BOLD_GREEN = "\033[1;32m"
    BOLD_YELLOW = "\033[1;33m"
    BOLD_CYAN = "\033[1;36m"
    BOLD_WHITE = "\033[1;37m"


def should_use_color(file=sys.stdout, force_color: Optional[bool] = None) -> bool:
    """Determines whether ANSI color codes should be emitted based on TTY and environment."""
    if force_color is not None:
        return force_color
    if os.environ.get("NO_COLOR"):
        return False
    return hasattr(file, "isatty") and file.isatty()


def colorize(text: str, color_code: str, enabled: bool = True) -> str:
    """Wraps text with ANSI color codes when color is enabled."""
    if not enabled or not color_code:
        return text
    return f"{color_code}{text}{Colors.RESET}"


@contextmanager
def suppress_c_stderr() -> Generator[None, None, None]:
    """Temporarily redirect low-level C library stderr output to the null device."""
    redirected = False
    saved_stderr_fd = -1
    stderr_fileno = -1

    try:
        stderr_fileno = sys.stderr.fileno()
        saved_stderr_fd = os.dup(stderr_fileno)
        null_fd = os.open(os.devnull, os.O_WRONLY)
        try:
            sys.stderr.flush()
            os.dup2(null_fd, stderr_fileno)
            redirected = True
        finally:
            os.close(null_fd)
    except (AttributeError, OSError, ValueError):
        if saved_stderr_fd >= 0:
            os.close(saved_stderr_fd)
        saved_stderr_fd = -1

    try:
        yield
    finally:
        if redirected:
            try:
                sys.stderr.flush()
                os.dup2(saved_stderr_fd, stderr_fileno)
            finally:
                os.close(saved_stderr_fd)


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
