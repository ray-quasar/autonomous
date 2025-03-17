import numpy as np
from scan_data import laser_scan

# This is a toy example of the disparityExtender algorithm
# It takes in a LiDAR scan and outputs a steering angle and speed
# The LiDAR scan is like a LaserScan message and is defined as an array in scan_data.py

# First, we do preprocessing on the raw scan data
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

# We also have to throw away the rear values of the scan
# We assume that theta = 0 is directly in front of the car
# We want to ignore the rear 90 degrees of the scan
# This bit of code assumes that the scan is 360 degrees and keeps the middle half
valid_ranges = valid_ranges[len(valid_ranges)//4:3*len(valid_ranges)//4]
# 90 degrees might not be enough, but this can be parameterized 

# Next we need a mechanism to identify disparities in the scan
# A disparity is a large jump in range values
# We expect the car to running in a track about 2 meters wide
# So we can set a guesstimate for disparities being at least 0.5 meters jump in range values
disparity_check = 0.5
# I think this should actually be something about the size and turning radius of the car but I have no idea how to calculate that

# BUILD A FUNCTTION TO DETECT DISPARITIES

def find_disparities(ranges, check_value):


    return # return the indices of the disparities

# Now that we know where the disparities are, we can EXTEND them
# For each disparity, we want to extend it towards the increase in range values
# The extension distance needs to be half the width of the car
extension_distance = 0.5
# But the amount of lidar scan points we rewrite depends on the distance from the car
# The further away the disparity, the fewer points we need to rewrite
# To calculate the number of points to rewrite, we can use the angle increment of the scan
# The angle increment tells us how many radians each scan point represents
# We imagine an isoceles triangle with the base as the disparity extension distance and the height as distance from the car
# The angle at the spear tip of the triangle gets divided by the angle increment to get the number of points to rewrite
# I think it's actually half the angle at the spear tip

# BUILD A FUNCTION TO EXTEND DISPARITIES

def extend_disparities(ranges, disparities, extension_distance):

    return # return the extended ranges

# Now that we have extended the disparities, we have a set of virtual ranges in configuration space
# There are two options/strategies for following the gap
# 1. Follow the largest gap
# 2. Follow the deepest gap
# The largest gap is the widest gap in the ranges
# The deepest gap is the gap with the highest range values
# The largest gap is the safest option and would probably better for the obstacle avoidance run
# The deepest gap is the fastest option and would probably be better for the time trial run

# BUILD A FUNCTION TO FIND THE LARGEST GAP

def find_largest_gap(ranges):

    return # return the start and end indices of the largest

# BUILD A FUNCTION TO FIND THE DEEPEST

def find_deepest_gap(ranges):
    
    return # return the start and end indices of the deepest

# Now that we have the desired gap, we can calculate the steering angle and speed

# The steering angle is the average angle of the gap
# We can get this by choosing the angle of the index in the middle of the gap
# The steering angle should be positive if the gap is to the left, and negative if to the right
# The orientation of the LiDAR readings may require us to invert the steering angle
# The steering angle should be limited to +/- 0.34 radians (20 degrees)

# The speed is the base speed if the gap is wide, and slower if it's narrow
# The speed should also be moderated by the depth of the gap and the steering angle

# If there is no gap, stop the car
