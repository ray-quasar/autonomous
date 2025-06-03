#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np
from scipy.ndimage import convolve1d, gaussian_filter1d

class LaserScanLogger(Node):
    def __init__(self):
        super().__init__('laser_scan_logger')
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',  # Change this to your LaserScan topic if different
            self.scan_callback,
            10)
        self.scan_received = False

    def convolutional_disp_extender2(self, ranges, check_value, angle_increment):

        lookahead_distance = 5.0
        extension_distance = 0.185

        ## Detection
        # Edge detection kernel
        edge_filter = np.array([-1,1])
        convolved = np.zeros(len(ranges))
        convolved[
            len(ranges)//4 : 3*len(ranges)//4] = (
            convolve1d(ranges, edge_filter, mode='wrap')
        )[len(ranges)//4 : 3*len(ranges)//4]

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
            return np.clip((angles / angle_increment).astype(int), 0, len(ranges)) # type: ignore
        
        # Disparities are rewritten according to the value at the disparity index:
            # In this rewrite I'm ensuring the value at the dispartiy index is infact the "low" value
        num_pts_neg = compute_extension_pts(negative_disparities)
        num_pts_pos = compute_extension_pts(positive_disparities)

        ## Indices increase CCW. 
        # Negative disparities are caught on the "small"est index, closer to zero
        # negative_starts = negative_disparities
        negative_ends = np.clip(negative_disparities + num_pts_neg, 0, len(ranges)) 
            # ^ To ensure we are not running over into indices that do not exist
        for start, end in zip(negative_disparities, negative_ends):
            mask = extended_ranges[start:end] > ranges[start]
                # ^ To ensure we are not rewriting values with greater numbers
            extended_ranges[start:end][mask] = ranges[start]
                # We always want to rewrtie to the value at the disparity
                # In the negative case, this is the starting index
        # Positive dispartities are caught on the "largest" index, closer to 1080
        # positive_ends = positive_disparities
        positive_starts = np.clip(positive_disparities - num_pts_pos, 0, len(ranges))
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

    def scan_callback(self, msg):
        if not self.scan_received:
            self.get_logger().info("Received a LaserScan message. Writing to file...")
            
            # Extract relevant data
            ranges = np.array(msg.ranges)
            # ranges = np.flip(np.roll(ranges, len(ranges)//2))

            ranges =   np.flip(np.roll(   # 3. Rotate the scan pi/2 about both x and z 
                            np.nan_to_num(
                                np.clip(  # 2. Get rid of garbage values
                                    ranges,
                                msg.range_min, msg.range_max), 
                            nan = 0.0), 
                        len(ranges)//2))
            angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))

            angle_increment = msg.angle_increment

            # Apply processing functions

            disparities, extended_ranges, convolved = self.convolutional_disp_extender2(ranges, 0.85, angle_increment)

            disparity_mask = np.zeros_like(extended_ranges)
            disparity_mask[disparities] = 1

            # Set values above lookahead distance to zero
            extended_ranges = np.where( 
                        extended_ranges > 5.0,
                        0.0,
                        extended_ranges
                    )
            # Occlude the ranges to a 180-degree FOV
            extended_ranges[:len(ranges)//4] = 0.0
            extended_ranges[-len(ranges)//4:] = 0.0
            
            smoothed_ranges = gaussian_filter1d(extended_ranges, sigma = 1, mode='wrap')
            
            # Save to a text file (CSV format)
            data = np.column_stack((angles, ranges, convolved, disparity_mask, extended_ranges, smoothed_ranges))
            np.savetxt(
                'laser_scan_data.txt', 
                data, 
                header="angle(rad),range(m),convolved,disparities,extended,smoothed",
                comments='', 
                fmt='%.6f'
            )
            
            self.scan_received = True
            self.get_logger().info("Data written to laser_scan_data.txt. Exiting...")
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = LaserScanLogger()
    rclpy.spin(node)
    node.destroy_node()

if __name__ == '__main__':
    main()