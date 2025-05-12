import numpy as np
import math
from test_scan import laser_scan

# First, we do preprocessing on the raw scan data
# We copy in the raw scan data into a NumPy array
valid_ranges = np.copy(laser_scan["ranges"])

# We know we cannot trust values that are beyond the specified range_min and range_max
valid_ranges = np.clip(valid_ranges, laser_scan["range_min"], laser_scan["range_max"])

# infs will get clipped to range_max, but NaNs will stay as NaNs
# We can replace NaNs with 0.0
valid_ranges = np.nan_to_num(valid_ranges, nan=0.0)

# Using averaging filter to smooth the data
valid_ranges = np.convolve(valid_ranges, np.ones(5)/5, mode='same')

valid_ranges[:len(valid_ranges)//4] = 0
# print(len(valid_ranges)//4)
valid_ranges[3*len(valid_ranges)//4:] = 0
# print(3*len(valid_ranges)//4)

## ABOVE IS VALIDATED

disparity_check = 0.85

def find_disparities_convolution(ranges, check_value):
    """
    Identifies disparities in the LiDAR scan data using convolution.

    A disparity is defined as a significant jump in range values between consecutive points.
    This function uses a convolution with an edge detection kernel to find disparities.

    Parameters:
        ranges (np.array): Array of range values from the LiDAR scan.
        check_value (float): The minimum difference between consecutive range values to be considered a disparity.

    Returns:
        list: A list of indices where disparities are detected.
    """
    # Apply convolution with an edge detection kernel
    edge_filter = np.array([-1, 1])
    convolved = np.convolve(ranges, edge_filter, mode='same')

    # Find indices where the absolute value of the convolution exceeds the check_value
    # We multiply the indices by the sign of the convolved value to get the signed disparity
    signed_disparity_indices = np.where(np.abs(convolved) >= check_value)[0] * np.sign(convolved[np.abs(convolved) >= check_value])

    # Adjust indices to account for the 'valid' mode of convolution
    return signed_disparity_indices

# test_disparities = find_disparities(valid_ranges, disparity_check)
test_disparities = find_disparities_convolution(valid_ranges, disparity_check)
# print(test_disparities)
# print(test_disparities_convolution)

angles = np.linspace(laser_scan["angle_min"], laser_scan["angle_max"], len(valid_ranges))

# Now that we know where the disparities are, we can EXTEND them
# For each disparity, we want to extend it towards the increase in range values
# The extension distance needs to be half the width of the car
extension_distance = 0.15


def extend_disparities_convolution(ranges, disparities, extension_distance):
    """
    Extends the disparites in the LiDAR scan data with input data from the convolutional 
    disparity detection function.
    
    Parameters:
        ranges (np.array): Array of range values from the LiDAR scan.
        signed_disparities (list): List of indices where disparities are detected. 
            The indices are signed to indicate the direction of the disparity.
        extension_distance (float): The distance to extend the disparities.
        
    Returns:
        extended_ranges (np.array): The ranges with extended disparities.
    """



# todo: publish the ranges to a new /extended_scans topic
extended_ranges = extend_disparities_convolution(valid_ranges, test_disparities, extension_distance)

# Now that we have extended the disparities, we have a set of virtual ranges in configuration space
# There are two options/strategies for following the gap
# 1. Follow the widest gap
# 2. Follow the deepest gap
# The largest gap is the widest gap in the ranges
# The deepest gap is the gap with the highest range values
# The largest gap is the safest option and would probably better for the obstacle avoidance run
# The deepest gap is the fastest option and would probably be better for the time trial run

# BUILD A FUNCTION TO FIND THE LARGEST GAP

