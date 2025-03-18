import numpy as np

# Example 1D array
arr = np.array([1, 2, 3, 4, 5, 6])

# Roll by half the length
rotated_arr = np.roll(arr, len(arr) // 2)

print(rotated_arr)