import numpy as np
import math
from test_scan import laser_scan

# This is a toy example of the disparityExtender algorithm
# It takes in a LiDAR scan and outputs a steering angle and speed
# The LiDAR scan is like a LaserScan message and is defined as an array in scan_data.py

# First, we do preprocessing on the raw scan data
# We copy in the raw scan data into a NumPy array
valid_ranges = np.copy(laser_scan["ranges"])

# We know we cannot trust values that are beyond the specified range_min and range_max
valid_ranges = np.clip(valid_ranges, laser_scan["range_min"], laser_scan["range_max"])

# F1T recommends clipping to even lower than range_max to prevent swinging in long straightaways
# But this might not be necessary for a track with no right angles
# However it does help with reading false positive disparities where we are getting very spread apart readings on far away walls
# An outlier filter might be better for this

# infs will get clipped to range_max, but NaNs will stay as NaNs
# We can replace NaNs with 0s
valid_ranges = np.nan_to_num(valid_ranges)
# I think NaNs should go to zero so they don't ever become the deepest point
# And as long as the NaNs are disparate, just one or two missed points, they should be extended over in the next step

## ABOVE IS VALIDATED

# We also have to throw away the rear values of the scan
# We assume that theta = 0 is directly in front of the car
# We want to ignore the rear 90 degrees of the scan
# This bit of code assumes that the scan is 360 degrees and keeps the middle half
# The rest of the values are set to 0
valid_ranges[:len(valid_ranges)//4] = 0
# print(len(valid_ranges)//4)
valid_ranges[3*len(valid_ranges)//4:] = 0
# print(3*len(valid_ranges)//4)

# 90 degrees might not be enough, but this can be parameterized later
# To test this I'm going to print the indices of the first and last non-zero values
nonzero_indices = np.nonzero(valid_ranges)
first_nonzero = nonzero_indices[0][0]
last_nonzero = nonzero_indices[0][-1]
# print(first_nonzero, last_nonzero)

# This method will also be helpful for finding the gaps later

## ABOVE IS VALIDATED

# Next we need a mechanism to identify disparities in the scan
# A disparity is a large jump in range values
# We expect the car to running in a track about 2 meters wide
# So we can set a guesstimate for disparities being at least 0.5 meters jump in range values
disparity_check = 0.5
# I think this should actually be something about the size and turning radius of the car but I have no idea how to calculate that

def find_disparities(ranges, check_value):
    """
    Identifies disparities in the LiDAR scan data.

    A disparity is defined as a significant jump in range values between consecutive points.
    This function iterates through the range values and records the indices where the jump
    exceeds the specified check value.

    Parameters:
        ranges (np.array): Array of range values from the LiDAR scan.
        check_value (float): The minimum difference between consecutive range values to be considered a disparity.

    Returns:
        list: A list of indices where disparities are detected.
    """
    disparities = []
    for i in range(0, len(ranges)-1):
        # Need to check if the current or the next range is zero
        # If it is, we need to skip it
        if ranges[i] == 0 or ranges[i+1] == 0:
            continue
        # If the next range is positively larger than the current range, we have a disparity at i
        if -(ranges[i] - ranges[i+1]) >= check_value:
            disparities.append(i)
        # Check if the difference between the current range and the next range is greater than the check value
        if ranges[i] - ranges[i+1] >= check_value:
            disparities.append(i+1)
    return disparities

test_disparities = find_disparities(valid_ranges, disparity_check)
# print(test_disparities)

## ABOVE IS VALIDATED

angles = np.linspace(laser_scan["angle_min"], laser_scan["angle_max"], len(valid_ranges))
# Now that we know where the disparities are, we can EXTEND them
# For each disparity, we want to extend it towards the increase in range values
# The extension distance needs to be half the width of the car
extension_distance = 0.15
# But the amount of lidar scan points we rewrite depends on the distance from the car
# The further away the disparity, the fewer points we need to rewrite
# To calculate the number of points to rewrite, we can use the angle increment of the scan
# The angle increment tells us how many radians each scan point represents
# We imagine an isoceles triangle with the base as the disparity extension distance and the height as distance from the car
# The angle at the spear tip of the triangle gets divided by the angle increment to get the number of points to rewrite
# I think it's actually half the angle at the spear tip

def extend_disparities(ranges, disparities, extension_distance):
    for i in disparities:
        triangle_height = ranges[i]
        triangle_base = extension_distance
        angle_to_extend = math.atan(triangle_base / triangle_height)
        points_to_rewrite = int(angle_to_extend / laser_scan["angle_increment"])
        # print(points_to_rewrite)
        # print(ranges[i-points_to_rewrite:i+points_to_rewrite])
        if i - points_to_rewrite < 0:
            for j in range(i + points_to_rewrite):
                if ranges[j] > ranges[i]:
                    ranges[j] = ranges[i]
        elif i + points_to_rewrite > len(ranges):
            for j in range(i - points_to_rewrite, len(ranges)):
                if ranges[j] > ranges[i]:
                    ranges[j] = ranges[i]
        else:
            for j in range(i - points_to_rewrite, i + points_to_rewrite):
                if ranges[j] > ranges[i]:
                    ranges[j] = ranges[i]
        # print(ranges[i-points_to_rewrite:i+points_to_rewrite])
    return ranges

# todo: publish the ranges to a new /extended_scans topic
extended_ranges = extend_disparities(valid_ranges, test_disparities, extension_distance)

# Now that we have extended the disparities, we have a set of virtual ranges in configuration space
# There are two options/strategies for following the gap
# 1. Follow the widest gap
# 2. Follow the deepest gap
# The largest gap is the widest gap in the ranges
# The deepest gap is the gap with the highest range values
# The largest gap is the safest option and would probably better for the obstacle avoidance run
# The deepest gap is the fastest option and would probably be better for the time trial run

# BUILD A FUNCTION TO FIND THE LARGEST GAP

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

# BUILD A FUNCTION TO FIND THE DEEPEST

def find_deepest_index(ranges):
    deep_index = np.argmax(ranges)    
    return deep_index

# This is going to be edited to return the average of all the indices 
# with max values or values above a certain threshold


desired_index = find_deepest_index(extended_ranges)
print(desired_index)
print(angles[desired_index])
print(valid_ranges[desired_index])
print(extended_ranges[desired_index])



# Now that we have the desired gap, we can calculate the steering angle and speed

# The steering angle is the average angle of the gap
# We can get this by choosing the angle of the index in the middle of the gap
# The steering angle should be positive if the gap is to the left, and negative if to the right
# The orientation of the LiDAR readings may require us to invert the steering angle
# The steering angle should be limited to +/- 0.34 radians (20 degrees)

# The speed is the base speed if the gap is wide, and slower if it's narrow
# The speed should also be moderated by the depth of the gap and the steering angle

# If there is no gap, stop the car
