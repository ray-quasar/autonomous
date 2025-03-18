import numpy as np
from scratch_sheet import find_disparities

def test_find_disparities_left():
    ranges = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    check_value = 0.5
    disparities = find_disparities(ranges, check_value)
    assert disparities == [3]

def test_find_disparities_right():
    ranges = np.array([2, 2, 2, 2, 1, 1, 1, 1])
    check_value = 0.5
    disparities = find_disparities(ranges, check_value)
    assert disparities == [4]

def test_find_disparities_multiple():
    ranges = np.array([1, 1, 2, 2, 1, 1, 2, 2])
    check_value = 0.5
    disparities = find_disparities(ranges, check_value)
    assert disparities == [1, 4, 5]

def test_find_disparities_none():
    ranges = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    ranges = np.array([1, 1, 1, 0, 1, 1, 1, 1])
    check_value = 0.5
    disparities = find_disparities(ranges, check_value)
    assert disparities == []

test_find_disparities_left()
test_find_disparities_right()
test_find_disparities_multiple()    
test_find_disparities_none()
print("All tests pass")