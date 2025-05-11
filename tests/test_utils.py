import pytest
from quell.utils import generate_intervals, generate_overlapping_intervals


def test_generate_intervals():
    # Test cases where k > n
    assert generate_intervals(5, 6) == []

    # Test cases where intervals fit perfectly into n
    assert generate_intervals(10, 2) == [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)]
    assert generate_intervals(9, 3) == [(0, 3), (3, 6), (6, 9)]

    # Test cases where there is a gap
    assert generate_intervals(10, 3) == [(0, 3), (3, 6), (6, 9)]
    assert generate_intervals(15, 4) == [(1, 5), (5, 9), (9, 13)]

    # Edge case where k == n
    assert generate_intervals(5, 5) == [(0, 5)]

    # Test cases where n or k is zero
    assert generate_intervals(0, 1) == []
    assert generate_intervals(1, 0) == []

    # Test cases with large n and k values
    assert generate_intervals(100, 25) == [(0, 25), (25, 50), (50, 75), (75, 100)]
    assert generate_intervals(100, 30) == [(5, 35), (35, 65), (65, 95)]


def test_generate_overlapping_intervals_variable_size():    
    assert generate_overlapping_intervals(20, 5, 2, variable_size=True) == [(0, 5), (3, 8), (6, 11), (9, 14), (12, 17), (15, 20)]
    assert generate_overlapping_intervals(10, 3, 1, variable_size=True) == [(0, 3), (2, 5), (4, 7), (6, 9), (7, 10)]
    assert generate_overlapping_intervals(5, 5, 2, variable_size=True) == [(0, 5)]
    assert generate_overlapping_intervals(5, 3, 0, variable_size=True) == [(0, 3), (2, 5)]
    assert generate_overlapping_intervals(0, 5, 2, variable_size=True) == []
    with pytest.raises(AssertionError):
        assert generate_overlapping_intervals(10, 3, 3, variable_size=True)



def test_generate_overlapping_intervals_fixed_size():    
    assert generate_overlapping_intervals(20, 5, 2, variable_size=False) == [(0, 5), (3, 8), (6, 11), (9, 14), (12, 17), (15, 20)]
    assert generate_overlapping_intervals(10, 3, 1, variable_size=False) == [(0, 3), (2, 5), (4, 7), (6, 9), (7, 10)]
    assert generate_overlapping_intervals(5, 5, 2, variable_size=False) == [(0, 5)]
    assert generate_overlapping_intervals(5, 3, 0, variable_size=False) == [(0, 3), (2, 5)]
    assert generate_overlapping_intervals(0, 5, 2, variable_size=False) == []
    with pytest.raises(AssertionError):
        assert generate_overlapping_intervals(10, 3, 3, variable_size=False)
