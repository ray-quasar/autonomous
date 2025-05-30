from test_scan import laser_scan

import numpy as np
from scipy.ndimage import convolve1d, gaussian_filter1d
import matplotlib.pyplot as plt

    
def convolutional_disp_extender2(ranges, check_value):
    ## Detection
    # Edge detection kernel
    edge_filter = np.array([-1,1])
    convolved = np.zeros(len(laser_scan["ranges"]))
    convolved[
        len(laser_scan["ranges"])//4 : 3*len(laser_scan["ranges"])//4] = (
        convolve1d(ranges, edge_filter, mode='wrap')
    )[len(laser_scan["ranges"])//4 : 3*len(laser_scan["ranges"])//4]

    ## Identification
    # The convolution result is NEGATIVE when we go from a small value to a large one:
    negative_disparities = np.where(
        (convolved < -check_value) & 
        # We only care to extend disparities within the lookahead_distance
        # For negative disparities, the point of interest is one index less, we roll the other way
        (np.roll(ranges, +1) < lookahead_distance)
    )[0] - 1  
    # The convolution result is POSITIVE when we go from a small value to a large one
    positive_disparities = np.where(
        (convolved > check_value) &
        (np.roll(ranges, -1) < lookahead_distance)
    )[0] + 1
    disparities = np.concatenate((positive_disparities,negative_disparities)) 

    ## Extension
    # Initialize output
    extended_ranges = np.copy(ranges)

    # Compute number of points to rewrite (vectorized)
    def compute_extension_pts(disparities):
        angles = np.arctan(extension_distance / ranges[disparities])
        return np.clip((angles / laser_scan["angle_increment"]).astype(int), 0, len(laser_scan["ranges"])) # type: ignore
    
    # Disparities are rewritten according to the value at the disparity index:
        # In this rewrite I'm ensuring the value at the dispartiy index is infact the "low" value
    num_pts_neg = compute_extension_pts(negative_disparities)
    num_pts_pos = compute_extension_pts(positive_disparities)

    ## Indices increase CCW. 
    # Negative disparities are caught on the "small"est index, closer to zero
    # negative_starts = negative_disparities
    negative_ends = np.clip(negative_disparities + num_pts_neg, 0, len(laser_scan["ranges"])) 
        # ^ To ensure we are not running over into indices that do not exist
    for start, end in zip(negative_disparities, negative_ends):
        mask = extended_ranges[start:end] > ranges[start]
            # ^ To ensure we are not rewriting values with greater numbers
        extended_ranges[start:end][mask] = ranges[start]
            # We always want to rewrtie to the value at the disparity
            # In the negative case, this is the starting index
    # Positive dispartities are caught on the "largest" index, closer to 1080
    # positive_ends = positive_disparities
    positive_starts = np.clip(positive_disparities - num_pts_pos, 0, len(laser_scan["ranges"]))
        # ^ To ensure we do not start at a negative index
    for start, end in zip(positive_starts, positive_disparities):
        mask = extended_ranges[start:end] > ranges[end]
        extended_ranges[start:end][mask] = ranges[end]
            # We always want to rewrite to the value at the disparity
            # In the positive case, this is the ending index
    return disparities, extended_ranges, convolved
    
def find_gaussian_max(ranges):
    ranges = gaussian_filter1d(ranges, sigma = 1, mode='wrap')
    return np.argmax(ranges)


# Lookahead distance (in meters)
lookahead_distance = 5.0 # steering
# Wheelbase of the car (in meters)
wheelbase = 0.325
# Threshold for detecting disparities (in meters)
disparity_check = 0.85
# Threshold for extending disparities (in meters)
extension_distance = 0.185
# Preprocess the scan data
full_ranges =   np.flip(np.roll(   # 3. Rotate the scan pi/2 about both x and z 
                    np.nan_to_num(
                        np.clip(  # 2. Get rid of garbage values
                            laser_scan["ranges"],
                        laser_scan["range_min"], laser_scan["range_max"]), 
                    nan = 0.0), 
                len(laser_scan["ranges"])//2))

## NOTE: Indices start exactly behind the car and increase CCW

# Find disparities and modify ranges
disparities, ext_ranges, convolved = convolutional_disp_extender2(
    full_ranges, disparity_check
    )

# Set values above lookahead distance to zero
ext_ranges = np.where( 
            ext_ranges > lookahead_distance,
            0.0,
            ext_ranges
        )
# Occlude the ranges to a 180-degree FOV
ext_ranges[:len(laser_scan["ranges"])//4] = 0.0
ext_ranges[-len(laser_scan["ranges"])//4:] = 0.0
target_index = find_gaussian_max(ext_ranges)

def plot_ranges_stem(ranges, color='b', filename='plot.png'):
    plt.rcParams['font.family'] = 'Jost'
    plt.rcParams['axes.facecolor'] = 'black'
    plt.rcParams['savefig.facecolor'] = 'black'
    plt.rcParams['figure.facecolor'] = 'black'
    plt.figure()
    markerline, stemlines, baseline = plt.stem(ranges, linefmt=color+'-', markerfmt=color+'o', basefmt="w-")
    plt.setp(markerline, markersize=3)
    plt.xlabel('Index', color='white')
    plt.ylabel('Range (m)', color='white')
    plt.title('Input Ranges Stem Plot', color='white')
    plt.ylim(0, 6)
    ax = plt.gca()
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    plt.savefig(filename)
    plt.close()

def plot_convolved_with_disparities(convolved, disparities, filename='convolved_with_disparities.png'):
    plt.rcParams['font.family'] = 'Jost'
    plt.rcParams['axes.facecolor'] = 'black'
    plt.rcParams['savefig.facecolor'] = 'black'
    plt.rcParams['figure.facecolor'] = 'black'
    plt.figure()
    plt.plot(convolved, color='cyan', label='Convolved')
    plt.scatter(disparities, convolved[disparities], color='magenta', s=12, label='Disparities', zorder=5)
    plt.xlabel('Index', color='white')
    plt.ylabel('Convolved Value', color='white')
    plt.title('Convolved Data with Disparities', color='white')
    plt.legend(facecolor='black', edgecolor='white', labelcolor='white')
    ax = plt.gca()
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    plt.savefig(filename)
    plt.close()

def plot_extended_ranges_with_rewrites(original_ranges, ext_ranges, filename='extended_ranges_with_rewrites.png'):
    plt.rcParams['font.family'] = 'Jost'
    plt.rcParams['axes.facecolor'] = 'black'
    plt.rcParams['savefig.facecolor'] = 'black'
    plt.rcParams['figure.facecolor'] = 'black'
    plt.figure()
    plt.plot(ext_ranges, color='lime', label='Extended Ranges')
    # Highlight rewritten indices
    rewritten = np.where(ext_ranges != original_ranges)[0]
    plt.scatter(rewritten, ext_ranges[rewritten], color='yellow', s=10, label='Rewritten', zorder=5)
    plt.xlabel('Index', color='white')
    plt.ylabel('Range (m)', color='white')
    plt.title('Extended Ranges with Rewritten Data', color='white')
    plt.legend(facecolor='black', edgecolor='white', labelcolor='white')
    ax = plt.gca()
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    plt.savefig(filename)
    plt.close()

def plot_gaussian_smoothed_with_max(ext_ranges, original_ranges, filename='gaussian_smoothed_with_max.png'):
    plt.rcParams['font.family'] = 'Jost'
    plt.rcParams['axes.facecolor'] = 'black'
    plt.rcParams['savefig.facecolor'] = 'black'
    plt.rcParams['figure.facecolor'] = 'black'
    plt.figure()
    smoothed = gaussian_filter1d(ext_ranges, sigma=1, mode='wrap')
    max_idx = np.argmax(smoothed)
    # plt.plot(ext_ranges, color='lime', label='Extended Ranges')
    plt.plot(smoothed, color='orange', label='Gaussian Smoothed')
    plt.scatter([max_idx], [smoothed[max_idx]], color='cyan', s=30, label='Max', zorder=5)
    plt.xlabel('Index', color='white')
    plt.ylabel('Range (m)', color='white')
    plt.title('Gaussian Smoothed Extended Ranges', color='white')
    plt.legend(facecolor='black', edgecolor='white', labelcolor='white')
    ax = plt.gca()
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    plt.savefig(filename)
    plt.close()

# Plot the input ranges as a stem plot
plot_ranges_stem(full_ranges, color='r', filename='input_ranges_stem.png')
# Plot the convolved data and highlight disparities
plot_convolved_with_disparities(convolved, disparities, filename='convolved_with_disparities.png')
# Plot the extended ranges and highlight rewritten data
plot_extended_ranges_with_rewrites(full_ranges, ext_ranges, filename='extended_ranges_with_rewrites.png')
# Plot the Gaussian smoothed data over the extended ranges and highlight the max
plot_gaussian_smoothed_with_max(ext_ranges, full_ranges, filename='gaussian_smoothed_with_max.png')