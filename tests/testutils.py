"""
Unit tests and edge cases for time formatting, interval subtraction, progress bars, and colors.
"""

import io
import os
from core.utils import (
    Colors,
    ProgressBar,
    colorize,
    fmt_time,
    should_use_color,
    subtract_intervals,
)


def test_fmt_time_zero():
    assert fmt_time(0.0) == "00:00.00"


def test_fmt_time_negative():
    assert fmt_time(-5.2) == "00:00.00"


def test_fmt_time_seconds_and_centiseconds():
    assert fmt_time(45.678) == "00:45.68"
    assert fmt_time(9.05) == "00:09.05"


def test_fmt_time_minutes():
    assert fmt_time(60.0) == "01:00.00"
    assert fmt_time(125.45) == "02:05.45"
    assert fmt_time(3605.20) == "60:05.20"


def test_subtract_intervals_empty_claimed():
    candidate = (10.0, 20.0)
    assert subtract_intervals(candidate, []) == [(10.0, 20.0)]


def test_subtract_intervals_no_overlap():
    candidate = (10.0, 20.0)
    claimed = [(0.0, 5.0), (25.0, 30.0)]
    assert subtract_intervals(candidate, claimed) == [(10.0, 20.0)]


def test_subtract_intervals_head_overlap():
    candidate = (10.0, 20.0)
    claimed = [(5.0, 14.0)]
    result = subtract_intervals(candidate, claimed)
    assert len(result) == 1
    assert result[0] == (14.0, 20.0)


def test_subtract_intervals_tail_overlap():
    candidate = (10.0, 20.0)
    claimed = [(18.0, 25.0)]
    result = subtract_intervals(candidate, claimed)
    assert len(result) == 1
    assert result[0] == (10.0, 18.0)


def test_subtract_intervals_middle_split():
    candidate = (10.0, 20.0)
    claimed = [(13.0, 16.0)]
    result = subtract_intervals(candidate, claimed)
    assert len(result) == 2
    assert result[0] == (10.0, 13.0)
    assert result[1] == (16.0, 20.0)


def test_subtract_intervals_completely_claimed():
    candidate = (10.0, 20.0)
    claimed = [(5.0, 25.0)]
    assert subtract_intervals(candidate, claimed) == []


def test_subtract_intervals_multiple_claimed():
    candidate = (0.0, 100.0)
    claimed = [(10.0, 20.0), (30.0, 40.0), (80.0, 95.0)]
    result = subtract_intervals(candidate, claimed)
    expected = [
        (0.0, 10.0),
        (20.0, 30.0),
        (40.0, 80.0),
        (95.0, 100.0)
    ]
    assert result == expected


def test_subtract_intervals_discards_sub_threshold():
    candidate = (10.0, 20.0)
    claimed = [(10.0, 19.96)]
    assert subtract_intervals(candidate, claimed) == []


def test_progress_bar_non_tty():
    stream = io.StringIO()
    pb = ProgressBar(total=10, prefix="Test", length=20, file=stream)
    pb.update(5)
    pb.finish()
    assert stream.getvalue() == ""


def test_progress_bar_tty():
    class MockTTYStream(io.StringIO):
        def isatty(self):
            return True

    stream = MockTTYStream()
    pb = ProgressBar(total=10, prefix="Processing", length=10, file=stream)
    pb.update(5, suffix="working")
    val = stream.getvalue()
    assert "\rProcessing [" in val
    assert "50.0%" in val
    assert "working" in val

    pb.update(10)
    pb.finish(message="Done")
    final_val = stream.getvalue()
    assert "Done\n" in final_val


def test_colorize_enabled_and_disabled():
    assert colorize("Hello", Colors.GREEN, enabled=True) == f"{Colors.GREEN}Hello{Colors.RESET}"
    assert colorize("Hello", Colors.GREEN, enabled=False) == "Hello"
    assert colorize("Hello", "", enabled=True) == "Hello"


def test_should_use_color():
    class DummyTTY:
        def isatty(self):
            return True

    class DummyNonTTY:
        def isatty(self):
            return False

    assert should_use_color(force_color=True) is True
    assert should_use_color(force_color=False) is False
    assert should_use_color(file=DummyTTY()) is True
    assert should_use_color(file=DummyNonTTY()) is False

    os.environ["NO_COLOR"] = "1"
    try:
        assert should_use_color(file=DummyTTY()) is False
    finally:
        os.environ.pop("NO_COLOR", None)