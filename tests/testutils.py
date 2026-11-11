"""
Unit tests and edge cases for time formatting and 1D interval subtraction arithmetic.
"""

from core.utils import fmt_time, subtract_intervals


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