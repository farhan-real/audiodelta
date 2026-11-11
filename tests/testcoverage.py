"""
Unit tests for bipartite interval coverage allocation and structural edit discovery.
"""

import pytest

from core.coverage import compute_verified_coverage
from core.models import MatchedSnippet


def test_compute_verified_coverage_empty():
    sec_a, sec_b, pct_a, pct_b, gaps_a, gaps_b = compute_verified_coverage(100.0, 100.0, [])
    assert sec_a == 0.0
    assert sec_b == 0.0
    assert pct_a == 0.0
    assert pct_b == 0.0
    assert gaps_a == [(0.0, 100.0)]
    assert gaps_b == [(0.0, 100.0)]


def test_compute_verified_coverage_invalid_durations():
    sec_a, sec_b, pct_a, pct_b, gaps_a, gaps_b = compute_verified_coverage(0.0, -10.0, [])
    assert sec_a == 0.0
    assert sec_b == 0.0
    assert pct_a == 0.0
    assert pct_b == 0.0


def test_compute_verified_coverage_exact_full_match():
    snippet = MatchedSnippet(
        a_start=0.0,
        a_end=50.0,
        b_start=0.0,
        b_end=50.0,
        duration_a=50.0,
        duration_b=50.0,
        speed_ratio=1.0,
        composite_similarity=0.95,
        harmonic_similarity=0.96,
        timbral_similarity=0.94
    )

    sec_a, sec_b, pct_a, pct_b, gaps_a, gaps_b = compute_verified_coverage(50.0, 50.0, [snippet])
    assert pytest.approx(sec_a, abs=1e-2) == 50.0
    assert pytest.approx(sec_b, abs=1e-2) == 50.0
    assert pytest.approx(pct_a, abs=1e-2) == 100.0
    assert pytest.approx(pct_b, abs=1e-2) == 100.0
    assert gaps_a == []
    assert gaps_b == []


def test_compute_verified_coverage_with_gaps_and_edits():
    snippet = MatchedSnippet(
        a_start=10.0,
        a_end=40.0,
        b_start=20.0,
        b_end=50.0,
        duration_a=30.0,
        duration_b=30.0,
        speed_ratio=1.0,
        composite_similarity=0.90,
        harmonic_similarity=0.90,
        timbral_similarity=0.90
    )

    sec_a, sec_b, pct_a, pct_b, gaps_a, gaps_b = compute_verified_coverage(60.0, 80.0, [snippet])

    assert pytest.approx(sec_a, abs=1e-2) == 30.0
    assert pytest.approx(sec_b, abs=1e-2) == 30.0
    assert pytest.approx(pct_a, abs=1e-2) == 50.0
    assert pytest.approx(pct_b, abs=1e-2) == 37.5

    assert len(gaps_a) == 2
    assert gaps_a[0] == (0.0, 10.0)
    assert gaps_a[1] == (40.0, 60.0)

    assert len(gaps_b) == 2
    assert gaps_b[0] == (0.0, 20.0)
    assert gaps_b[1] == (50.0, 80.0)


def test_compute_verified_coverage_deduplicates_overlapping_candidates():
    m1 = MatchedSnippet(10.0, 30.0, 10.0, 30.0, 20.0, 20.0, 1.0, 0.90, 0.90, 0.90)
    m2 = MatchedSnippet(20.0, 40.0, 20.0, 40.0, 20.0, 20.0, 1.0, 0.85, 0.85, 0.85)

    sec_a, sec_b, pct_a, pct_b, _, _ = compute_verified_coverage(50.0, 50.0, [m1, m2])

    assert pytest.approx(sec_a, abs=1e-2) == 30.0
    assert pytest.approx(sec_b, abs=1e-2) == 30.0
    assert pytest.approx(pct_a, abs=1e-2) == 60.0