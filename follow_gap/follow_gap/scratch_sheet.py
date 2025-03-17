import numpy as np

# Create arrays with different types of values
array1 = np.array([1, 2, np.nan, 4, 5])
array2 = np.array([1, 2, np.inf, 4, -np.inf])
array3 = np.array([1, 2, 3, 4, 5])

# Check for NaN values
isnan_array1 = np.isnan(array1)
isnan_array2 = np.isnan(array2)
isnan_array3 = np.isnan(array3)

# Check for Inf values
isinf_array1 = np.isinf(array1)
isinf_array2 = np.isinf(array2)
isinf_array3 = np.isinf(array3)

# Print results
print("Array 1:", array1)
print("isnan Array 1:", isnan_array1)
print("isinf Array 1:", isinf_array1)
print()

print("Array 2:", array2)
print("isnan Array 2:", isnan_array2)
print("isinf Array 2:", isinf_array2)
print()

print("Array 3:", array3)
print("isnan Array 3:", isnan_array3)
print("isinf Array 3:", isinf_array3)