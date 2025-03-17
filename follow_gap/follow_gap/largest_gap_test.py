import pytest
import numpy as np
from scratch_sheet import find_largest_gap

def test_find_largest_gap():
    ranges = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    start, end = find_largest_gap(ranges)
    assert start == 0
    assert end == 9

    ranges = np.array([1, 1, 1, 10, 10, 10, 1, 1, 1])
    start, end = find_largest_gap(ranges)
    assert start == 3
    assert end == 5

    ranges = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
    start, end = find_largest_gap(ranges)
    assert start == 0
    assert end == 8

    ranges = np.array([10, 10, 10, 1, 1, 1, 10, 10, 10])
    start, end = find_largest_gap(ranges)
    assert start == 0
    assert end == 2

    ranges = np.array([1, 1, 1, 1, 1, 1, 1, 1, 10])
    start, end = find_largest_gap(ranges)
    assert start == 8
    assert end == 8

if __name__ == "__main__":
    pytest.main()