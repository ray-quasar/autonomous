import numpy as np
import time
from scratch_sheet import find_disparities_convolution
from test_scan import laser_scan

def test_find_disparities_left():
    ranges = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    check_value = 0.5
    disparities = (find_disparities_convolution(ranges, check_value))[1:]
    print(disparities)
    assert np.array_equal(disparities, [-4])

def test_find_disparities_right():
    ranges = np.array([2, 2, 2, 2, 1, 1, 1, 1])
    check_value = 0.5
    disparities = (find_disparities_convolution(ranges, check_value))[1:]
    print(disparities)
    assert np.array_equal(disparities, [4])

def test_find_disparities_multiple():
    ranges = np.array([1, 1, 2, 2, 1, 1, 2, 2])
    check_value = 0.5
    disparities = (find_disparities_convolution(ranges, check_value))[1:]
    print(disparities)
    assert np.array_equal(disparities, [-2, 4, -6])

def test_find_disparities_none():
    ranges = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    check_value = 0.5
    disparities = (find_disparities_convolution(ranges, check_value))[1:]
    print(disparities)
    assert disparities.size == 0

# test_find_disparities_left()
# test_find_disparities_right()
# test_find_disparities_multiple()    
# test_find_disparities_none()
# print("All tests pass")

# The indices returned are always the second index of the disparity
# The sign of the disparity indicates whether the disparity is to the left or right
# As we go from left to right, a positive value indicates a decrease in range (opens left)
# A negative value indicates an increase in range (opens right)

ranges = np.array([1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2])
check_value = 0.5
disparities = (find_disparities_convolution(ranges, check_value))[1:] # [ -4   8 -12]

extension_distance = 2 

# We want to turn the signed disparities into a 2D matrix of indices:
# disparities: [x1, -x2, x3, -x4] 
# -> indices: [ [x1-2, x1-1, x1], [x2, x2+1, x2+2], [x3-2, x3-1, x3], [x4, x4+1, x4+2] ]
# Ideally, we want to do this in a single pass

extension_indices = []
