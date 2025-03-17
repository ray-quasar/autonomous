#!/usr/bin/env python3

import numpy as np

def find_largest_gap(ranges):
    """
    Finds the largest contiguous segment ("gap") where the range values are high.
    In this toy example the gap is defined by indices with values at or above 95%
    of the maximum range in the scan.
    
    Parameters:
        ranges (np.array): Array of (extended) range values.
        
    Returns:
        tuple: (start_index, end_index) of the largest gap.
    """
    threshold = 0.95 * np.max(ranges)
    best_gap_start, best_gap_end, best_gap_size = 0, 0, 0
    in_gap = False
    start = 0
    for i, r in enumerate(ranges):
        if r >= threshold:
            if not in_gap:
                in_gap = True
                start = i
        else:
            if in_gap:
                in_gap = False
                gap_size = i - start
                if gap_size > best_gap_size:
                    best_gap_size = gap_size
                    best_gap_start, best_gap_end = start, i - 1
    # Check if a gap extends to the end
    if in_gap:
        gap_size = len(ranges) - start
        if gap_size > best_gap_size:
            best_gap_start, best_gap_end = start, len(ranges) - 1
    return best_gap_start, best_gap_end

def run_test(ranges, test_name="Test"):
    print(f"--- {test_name} ---")
    print("Input ranges:", ranges)
    start, end = find_largest_gap(ranges)
    print(f"Largest gap: start index = {start}, end index = {end}")
    print("Gap values:", ranges[start:end+1])
    print()

if __name__ == '__main__':
    # Test 1: Multiple gaps with maximum value 5; threshold = 0.95*5 = 4.75.
    # Expected: Gap with contiguous 5's should be chosen.
    ranges1 = np.array([5, 5, 5, 2, 5, 5, 5, 5, 5, 1, 1, 5, 5])
    run_test(ranges1, "Test 1")

    # Test 2: All values are equal (1). Entire array qualifies.
    ranges2 = np.array([1, 1, 1, 1, 1])
    run_test(ranges2, "Test 2")

    # Test 3: All zeros.
    ranges3 = np.array([0, 0, 0, 0])
    run_test(ranges3, "Test 3")

    # Test 4: Mixed values; maximum is 5.
    ranges4 = np.array([0, 5, 5, 0, 5, 5, 5, 0])
    run_test(ranges4, "Test 4")
