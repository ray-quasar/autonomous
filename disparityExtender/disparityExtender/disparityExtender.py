import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import Joy
from scipy.ndimage import convolve1d

class disparityExtender(Node):
    def __init__(self):
        super().__init__('disparityExtender')

        self.get_logger().info(
            f"""
            Thank you for choosing Ray-Quasar for your racing needs.
            Initializing disparityExtender.
            """
        )

        # Debug and visualization flags
        self.enable_logging = self.declare_parameter('enable_logging', True).value
        self.enable_visualization = self.declare_parameter('enable_visualization', True).value
        self.get_logger().info(
            f"""
            Logging enabled: {self.enable_logging}
            Visualization enabled: {self.enable_visualization}
            """
        )
        
        # Subscribe to LiDAR scans (assumed on '/scan')
        self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.get_logger().info(
            f"Subscribed to LiDAR scan data on '/scan'."
        )
        # Cache for scan parameters
        self._scan_params = None
        # Lookahead distance (in meters)
        self.lookahead_distance = 8.0 

        # Publisher for AckermannDriveStamped messages on '/drive'
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.get_logger().info(
            f"Created publisher for AckermannDriveMsgs on '/drive'."
        )
        # Wheelbase of the car (in meters)
        self.wheelbase = 0.325
        # Maximum steering angle (radians)
        self.max_steering_angle = 0.34 

        # Publisher for disparity identifier scan data
        self.disparity_pub = self.create_publisher(LaserScan, '/disparity', 10)
        self.get_logger().info(
            f"Created publisher for disparity visualizer on '/disparity'."
        )
        # Threshold for detecting disparities (in meters)
        self.disparity_check = 0.85

        # Publisher for modified range data
        self.ext_scan_publisher = self.create_publisher(LaserScan, '/ext_scan', 10)
        self.get_logger().info(
            f"Created publisher for modified range visualizer on '/ext_scan'."
        )
        # Threshold for extending disparities (in meters)
        self.extension_distance = 0.185

        # TODO: Initialize point rewrite lookup table

    def lidar_callback(self, scan):
        """
        Callback function for processing incoming LiDAR scan data.
        """
        # # Convert raw scan data to a NumPy array
        # ranges = np.array(scan.ranges)  # Ignore the angle_min, angle_increment, etc.
        
        # # Using averaging filter to smooth the data with wrap-around
        # # ranges = convolve1d(ranges, np.ones(3)/3, mode='wrap')

        # # Rotate the scan data about z-axis
        # ranges = np.roll(ranges, len(ranges)//2)
        # # Rotate the scan data about x-axis
        # ranges = np.flip(ranges)

        # # Preprocess the scan data
        # ranges = np.clip(ranges, scan.range_min, self.lookahead_distance)
        # ranges = np.nan_to_num(ranges, nan=0.0)

        # Cache parameters on first scan
        if self._scan_params is None:
            self._scan_params = {
                'angle_min': scan.angle_min,
                'angle_max': scan.angle_max,
                'angle_increment': scan.angle_increment,
                'range_min': scan.range_min,
                'range_max': scan.range_max,
                'num_points': len(scan.ranges)
            }

        ranges = np.flip(np.roll(
            # convolve1d(
                np.nan_to_num(np.clip(
                    np.array(scan.ranges), 
                scan.range_min, self.lookahead_distance), nan=0.0), 
            # np.ones(3)/3, mode='wrap'),
        self._scan_params['num_points']//2))

        # Find disparities in the LiDAR scan data
        # disparities = self.find_disparities_diff(ranges, self.disparity_check)
        # disparities = self.find_disparities_convolution(ranges, self.disparity_check)

        

        # Extend disparities in the LiDAR scan data
        # ranges = self.extend_disparities(ranges, disparities, scan.angle_increment)

        disparities, ranges = self.convolutional_disp_extender2(
            ranges, self.disparity_check, scan.angle_increment
            )
        
        # Publish the disparity points to the '/disparities' topic
        self.publish_disparity_scan(ranges, disparities, scan)

        # Occlude the ranges to a specified FOV (Field of View)
        ranges[:self._scan_params['num_points']//4] = 0.0
        ranges[-self._scan_params['num_points']//4:] = 0.0

        # Find the index of the deepest point in the LiDAR scan data
        target_index = self.find_deepest_gap(ranges)

        self.publish_drive_command(scan, ranges, target_index)
        self.publish_laser_scan(ranges, scan)

    # Helper functions

    def find_disparities_diff(self, ranges, check_value):
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
        valid = (ranges[:-1] > 0) & (ranges[1:] > 0)  # Exclude comparisons with invalid (0) ranges
        diffs = np.diff(ranges)

        disparities = np.where((np.abs(diffs) >= check_value) & valid)[0]

        # Assign i and i+1 depending on direction
        results = []
        for i in disparities:
            if diffs[i] > 0:
                results.append(i)    # right is farther, left is closer
            else:
                results.append(i + 1)  # left is farther, right is closer
        return results
    
    def convolutional_disp_extender(self, ranges, check_value, angle_increment):
        # The disparity is defined as the the JUMP in the range values >= check_value

        # Apply convolution with an edge detection kernel
        edge_filter = np.array([-1, 1])
        convolved = np.convolve(ranges, edge_filter, mode='same')

        # Find indices where the absolute value of the convolution exceeds the check_value
        # disparity_indices = np.where(np.abs(convolved) >= check_value)[0]
        left_disparities = np.where(convolved < -check_value)[0]
        right_disparities = np.where(convolved > check_value)[0] - 1

        num_pts_rewrite_left = np.array( 
            np.arctan(self.extension_distance / ranges[left_disparities]) # Angle to extend
            / angle_increment
            ).astype(int)
        num_pts_rewrite_right = np.array( 
            np.arctan(self.extension_distance / ranges[right_disparities]) # Angle to extend
            / angle_increment
            ).astype(int)
        
        extended_ranges = ranges
        for i in np.arange(0, len(left_disparities)):
            extended_ranges[
                (left_disparities[i] - num_pts_rewrite_left[i]) : left_disparities[i]
                ] = ranges[left_disparities[i]] 
        for j in np.arange(0, right_disparities):
            extended_ranges[
                right_disparities[j] : (right_disparities[j] + num_pts_rewrite_right[i])
            ] = ranges[right_disparities[j]]

        return extended_ranges
    
    def convolutional_disp_extender2(self, ranges, check_value, angle_increment):
        # Edge detection kernel
        edge_filter = np.array([-1, 1])
        # convolved = np.zeros(self._scan_params['num_points'])
        convolved = convolve1d(ranges, edge_filter, mode='wrap')

        # Detect left and right disparities
        right_disparities = np.clip(
            np.where(
                convolved < -check_value
            )[0], self._scan_params['num_points']//4, 3*self._scan_params['num_points']//4
        )
        # right_disparities = [0]
        left_disparities = np.clip(
            np.where(
                convolved > check_value
            )[0], self._scan_params['num_points']//4, 3*self._scan_params['num_points']//4
        ) 
        left_disparities = [0]

        disparities = np.concatenate((left_disparities,right_disparities))

        # Compute number of points to rewrite (vectorized)
        def compute_extension_pts(disparities):
            angles = np.arctan(self.extension_distance / ranges[disparities])
            return np.clip((angles / self._scan_params['angle_increment']).astype(int), 0, self._scan_params['num_points'])

        num_pts_left = compute_extension_pts(left_disparities)
        num_pts_right = compute_extension_pts(right_disparities)

        # Initialize output
        extended_ranges = np.copy(ranges)

        # Vectorized left extension
        left_starts = np.clip(left_disparities - num_pts_left, 0, self._scan_params['num_points'])
        for start, end in zip(left_starts, left_disparities):
            extended_ranges[start:end] = ranges[end]

        # Vectorized right extension
        right_ends = np.clip(right_disparities + num_pts_right, 0, self._scan_params['num_points'])
        for start, end in zip(right_disparities, right_ends):
            extended_ranges[start:end] = ranges[start]

        return disparities, extended_ranges

    def extend_disparities(self, ranges, disparities, angle_increment):
        for i in disparities:

            disparity_distance = np.min([ranges[i-1], ranges[i]])
            angle_to_extend = np.arctan(self.extension_distance / disparity_distance)
            points_to_rewrite = int(angle_to_extend / angle_increment ) 

            if i - points_to_rewrite < 0:
                for j in range(i + points_to_rewrite):
                    if ranges[j] > ranges[i]:
                        ranges[j] = disparity_distance
            elif i + points_to_rewrite > len(ranges):
                for j in range(i - points_to_rewrite, len(ranges)):
                    if ranges[j] > ranges[i]:
                        ranges[j] = disparity_distance
            else:
                for j in range(i - points_to_rewrite, i + points_to_rewrite):
                    if ranges[j] > ranges[i]:
                        ranges[j] = disparity_distance
            # print(ranges[i-points_to_rewrite:i+points_to_rewrite])
        return ranges
    
    def find_deepest_gap(self, ranges):
        """
        Finds the "deepest" gap in the scan by first locating the index with the maximum
        range value and then expanding left and right until the values drop below 90% of
        that maximum.

        Parameters:
            ranges (np.array): Array of (extended) range values.
        Returns:
            Middle of the deepest gap.
        """
        # # Find the index of the maximum range value
        # best_index = np.argmax(ranges)
        # # Expand left and right until the values drop below 90% of the maximum
        # threshold = 0.9 * ranges[best_index]
        # left = best_index
        # right = best_index
        # while left > 0 and ranges[left - 1] >= threshold:
        #     left -= 1
        # while right < len(ranges) - 1 and ranges[right + 1] >= threshold:
        #     right += 1
        # middle = (left + right) // 2
        # return middle

        return np.average( 
            np.where(
                ranges > (0.9 * np.max(ranges))
            ) 
        ).astype(int)
    

    def publish_drive_command(self, scan, ranges, deep_index):
        """
        Publish an AckermannDriveStamped command message to the '/drive' topic.
        Parameters:

        """   
        forward_distance = ranges[len(ranges)//2]   # The distance directly in front of the car
        target_distance = ranges[deep_index]  # The distance to the target point
        target_angle = scan.angle_min + deep_index * scan.angle_increment  # The angle to the target point


        # Ackermann Steering Angle Calculation:
        # - Based on curvature of the path and the wheelbase of the car
        # - The formula for the steering angle is:
        #       steering_angle = atan(wheelbase * curvature)
        #       curvature = 2 * target_distance * sin(target_angle) / target_distance^2
        #  - target_distance cancels out of the numerator, the calculation is simplified to:
        #       steering_angle = atan(wheelbase * 2 * sin(target_angle) / target_distance)
        
        # - The width of the track constrains target_distance*cos(target_angle) to <= forward_distance
        # - The updated value of the hypotenuse is:
        #      hypotenuse = forward_distance / cos(target_angle)

        new_target_distance = forward_distance / np.cos(target_angle)
        if new_target_distance < target_distance:
            target_distance = new_target_distance

        theoretical_steering_angle = np.arctan(
            self.wheelbase * 2 * np.sin(target_angle) / target_distance
        )
        
        # Limit the steering angle to the maximum steering angle of the car
        bounded_steering_angle = max(min(theoretical_steering_angle, 0.34), -0.34)
        
        forward_distance = max(ranges[len(ranges)//2 - 5 : len(ranges)//2 + 5])

        speed_max = 6.0
        speed_min = 1.0
        accel = 1.0
        a_center = 3.0
        
        speed = (
                (speed_max - speed_min) 
                / (1 + np.exp(- accel * (forward_distance - a_center))) 
                + speed_min
        )

        # Test mode
        speed = 0.0

        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = "base_link"
        
        drive_msg.drive.steering_angle = bounded_steering_angle
        drive_msg.drive.speed = speed
        
        self.drive_pub.publish(drive_msg)

        if self.enable_logging:
            self.get_logger().info(
                f"""
                Target Location: {target_distance:.2f} m, {np.rad2deg(-target_angle):.2f} deg, 
                Published Steering: {np.rad2deg(-bounded_steering_angle):.2f} deg, 
                Speed: {speed:.2f} m/s
                """
            )

    def publish_disparity_scan(self, ranges, disparities, raw_scan_data):
        """
        Publish the disparity points to the '/disparities' topic.

        Parameters:
            disparities (np.array): Modified range data.
            scan_data (LaserScan): Original LiDAR scan data.
        """

        if not self.enable_visualization:
            return
        
        disparities_values = np.zeros(len(ranges))
        disparities_values[disparities] = ranges[disparities]

        disparities_values = np.flip(np.roll(disparities_values, -len(disparities_values)//2)).tolist()
        
        modified_scan = LaserScan()
        modified_scan.header.stamp = raw_scan_data.header.stamp
        modified_scan.header.frame_id = 'laser' # The modified scan data is published in the laser frame
        modified_scan.angle_min = raw_scan_data.angle_min
        modified_scan.angle_max = raw_scan_data.angle_max
        modified_scan.angle_increment = raw_scan_data.angle_increment
        modified_scan.scan_time = raw_scan_data.scan_time
        modified_scan.range_min = raw_scan_data.range_min
        modified_scan.range_max = raw_scan_data.range_max
        modified_scan.ranges = disparities_values
        modified_scan.intensities = raw_scan_data.intensities

        self.disparity_pub.publish(modified_scan)

    def publish_laser_scan(self, ranges, raw_scan_data):
        """
        Publish the modified range data to the '/ext_scan' topic.

        Parameters:
            ranges (np.array): Modified range data.
            scan_data (LaserScan): Original LiDAR scan data.
        """
        if not self.enable_visualization:
            return

        ranges = np.flip(np.roll(ranges, -len(ranges)//2)).tolist()
       
        modified_scan = LaserScan()
        modified_scan.header.stamp = raw_scan_data.header.stamp
        modified_scan.header.frame_id = 'laser' # The modified scan data is published in the laser frame
        modified_scan.angle_min = raw_scan_data.angle_min
        modified_scan.angle_max = raw_scan_data.angle_max
        modified_scan.angle_increment = raw_scan_data.angle_increment
        modified_scan.scan_time = raw_scan_data.scan_time
        modified_scan.range_min = raw_scan_data.range_min
        modified_scan.range_max = raw_scan_data.range_max
        modified_scan.ranges = ranges
        modified_scan.intensities = raw_scan_data.intensities

        self.ext_scan_publisher.publish(modified_scan)

def main(args=None):
    rclpy.init(args=args)
    node = disparityExtender()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
