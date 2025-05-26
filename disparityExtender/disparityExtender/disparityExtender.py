import rclpy
import numpy as np
import copy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import UInt8, Float64
from scipy.ndimage import convolve1d, gaussian_filter1d

class disparityExtender(Node):
    def __init__(self):
        super().__init__('disparityExtender')

        self.get_logger().info(
            r"""
____ ____ _   _    ____ _  _ ____ ____ ____ ____ 
|__/ |__|  \_/  __ |  | |  | |__| [__  |__| |__/ 
|  \ |  |   |      |_\| |__| |  | ___] |  | |  \ 
                                                                                                                            
        Initializing disparityExtender...
            """
        )

        # Debug and visualization flags
        self.enable_speed = self.declare_parameter('enable_speed', False).value
        # self.enable_speed = True # Just for testing
        self.enable_logging = self.declare_parameter('enable_logging', False).value
        self.enable_visualization = self.declare_parameter('enable_visualization', True).value
        self.get_logger().info(
            f"""
Launching with parameters:
    - Motor {'enabled' if self.enable_speed else 'disabled'}
    - Logging {'enabled' if self.enable_logging else 'disabled'}
    - Visualization {'enabled' if self.enable_visualization else 'disabled'}
            """
        )
        self.get_logger().info(
            f"\n\rSubscribing to LiDAR scan data on '/scan'. \
                \n\rCreating publisher for AckermannDriveMsgs on '/drive'. \
                \n\rCreating publisher for EMERGENCY STOP on '/commands/motor/brake'. \
                \n\rCreating publisher for disparity visualizer on '/disparity'. \
                \n\rCreating publisher for modified range visualizer on '/ext_scan'. \
                \n\rSubscribing to navigation control on '/nav_control'. \
            "
        )

        # # Subscribe to LiDAR scans (assumed on '/scan')
        self.lidar_sub = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, 10
        )
        # Cache for scan parameters        
        self._scan_params = None
        # Lookup tables for wall proximity checks
        self._proximity_lut = None
        # Lookahead distance (in meters)
        self.lookahead_distance = 5.0 # steering

        # # Publisher for AckermannDriveStamped messages on '/drive'
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, '/drive', 10
        )
        # Wheelbase of the car (in meters)
        self.wheelbase = 0.325
        # Maximum steering angle (radians)
        self.max_steering_angle = 0.34 

        # # Publisher for E-stop
        self.emergency_stop_pub = self.create_publisher(
            Float64, '/commands/motor/brake', 10
        )
        self.brake_counter = 0

        # # Publisher for disparity identifier scan data
        self.disparity_pub = self.create_publisher(
            LaserScan, '/disparity', 10
        )
        # Threshold for detecting disparities (in meters)
        self.disparity_check = 0.85

        # # Publisher for modified range data
        self.ext_scan_publisher = self.create_publisher(
            LaserScan, '/ext_scan', 10
        )
        # Threshold for extending disparities (in meters)
        self.extension_distance = 0.185

        # # Navigation control subscriber
        self.nav_control_sub = self.create_subscription(
            UInt8, 'nav_control', self.nav_control_callback, 10
        )
        
        self.get_logger().info(
            f"\n\rInitialization complete. \n \
                \n\r Press Y to toggle visualization. \
                \n\r Press X to toggle logger. \
                \n\r Press B to activate emergency brake. \n \
                \n\r Press A to start car..."
        )

        # TODO: Initialize point rewrite lookup table
        # TODO: Do not start program until controller confirmation

    def nav_control_callback(self, msg):
        """Toggle navigation control states on UInt8 messages"""
        match msg.data:
            case 1: # motor_toggle
                self.enable_speed = not self.enable_speed
                self.get_logger().info(
                    f"\n\rMotor {'enabled' if self.enable_speed else 'disabled'}"
                )
                if not self.enable_speed:
                    self.brake_counter = 2
                    self.brake_timer = self.create_timer(0.1, self.brake_loop)
            case 2: # e_stop
                self.enable_speed = False
                self.enable_logging = False
                self.brake_counter = 50
                self.get_logger().info(
                    f"\n\r --- EMERGENCY STOP ACTIVATED. --- \
                        \n\r --- APPLYING BRAKE FOR 5 SEC. --- "
                )
                self.brake_timer = self.create_timer(0.1, self.brake_loop)
            case 3: # type: ignore # enable_logging
                self.enable_logging = not self.enable_logging
                self.get_logger().info(
                    f"\n\rLogging {'enabled' if self.enable_logging else 'disabled'}"
                )
            case 4: # enable_visualization
                self.enable_visualization = not self.enable_visualization
                self.get_logger().info(
                    f"\n\rVisualization {'enabled' if self.enable_visualization else 'disabled'}"
                )

    def lidar_callback(self, scan):
        """
        Callback function for processing incoming LiDAR scan data.
        """
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
            # self._initialize_proximity_lut()
            self.get_logger().info(
                f"\n\r Cached scan parameters and initialized LUTs. \
                  \n\r Angles (rad): {scan.angle_min:.2f} to {scan.angle_max:.2f}, by {scan.angle_increment:.2f} \
                  \n\r Ranges (m): {scan.range_min:.2f} to {scan.range_max:.2f}, with {len(scan.ranges)} points"
            )

        # Preprocess the scan data
        full_ranges =   np.flip(np.roll(   # 3. Rotate the scan pi/2 about both x and z 
                            np.nan_to_num(
                                np.clip(  # 2. Get rid of garbage values
                                    np.array(scan.ranges),  # 1. Convert scan to NumPy array
                                scan.range_min, scan.range_max), 
                            nan = 0.0), 
                        self._scan_params['num_points']//2))
        
        ## NOTE: Indices start exactly behind the car and increase CCW
        
        # Find disparities and modify ranges
        disparities, ext_ranges = self.convolutional_disp_extender2(
            full_ranges, self.disparity_check
            )
        
        ext_ranges = np.where( 
                    ext_ranges > self.lookahead_distance,
                    0.0,
                    ext_ranges
                )
        
        # Publish the disparity points to the '/disparities' topic
        self.publish_disparity_scan(full_ranges, disparities, scan)
        # self.publish_disparity_scan(scan.ranges, disparities, scan)

        # Occlude the ranges to a 180-degree FOV
        ext_ranges[:self._scan_params['num_points']//4] = 0.0
        ext_ranges[-self._scan_params['num_points']//4:] = 0.0

        # Find the index of the deepest cluster in the LiDAR scan data
        # target_index = self.find_deepest_gap(ext_ranges)
        target_index = np.argmax(ext_ranges)

        self.publish_drive_command(ext_ranges, full_ranges, target_index)
        self.publish_laser_scan(ext_ranges, scan)

    # # Helper functions
    
    def convolutional_disp_extender2(self, ranges, check_value):

        ## Detection

        # Edge detection kernel
        edge_filter = np.array([-1,1])
        convolved = np.zeros(self._scan_params['num_points'])
        convolved[
            self._scan_params['num_points']//4 : 3*self._scan_params['num_points']//4] = (
            convolve1d(ranges, edge_filter, mode='wrap')
        )[self._scan_params['num_points']//4 : 3*self._scan_params['num_points']//4]

        ## Identification

        # The convolution result is NEGATIVE when we go from a small value to a large one:
        negative_disparities = np.where(
            (convolved < -check_value) & 
            # We only care to extend disparities within the lookahead_distance
            # For negative disparities, the point of interest is one index less, we roll the other way
            (np.roll(ranges, +1) < self.lookahead_distance)
        )[0] - 1  

        # The convolution result is POSITIVE when we go from a small value to a large one
        positive_disparities = np.where(
            (convolved > check_value) &
            (np.roll(ranges, -1) < self.lookahead_distance)
        )[0] + 1

        disparities = np.concatenate((positive_disparities,negative_disparities)) 

        ## Extension

        # Initialize output
        extended_ranges = np.copy(ranges)

        # Compute number of points to rewrite (vectorized)
        def compute_extension_pts(disparities):
            angles = np.arctan(self.extension_distance / ranges[disparities])
            return np.clip((angles / self._scan_params['angle_increment']).astype(int), 0, self._scan_params['num_points']) # type: ignore
        
        # Disparities are rewritten according to the value at the disparity index:
            # In this rewrite I'm ensuring the value at the dispartiy index is infact the "low" value
        num_pts_neg = compute_extension_pts(negative_disparities)
        num_pts_pos = compute_extension_pts(positive_disparities)

        ## Indices increase CCW. 

        # Negative disparities are caught on the "small"est index, closer to zero
        # negative_starts = negative_disparities
        negative_ends = np.clip(negative_disparities + num_pts_neg, 0, self._scan_params['num_points']) 
            # ^ To ensure we are not running over into indices that do not exist

        for start, end in zip(negative_disparities, negative_ends):
            mask = extended_ranges[start:end] > ranges[start]
                # ^ To ensure we are not rewriting values with greater numbers
            extended_ranges[start:end][mask] = ranges[start]
                # We always want to rewrtie to the value at the disparity
                # In the negative case, this is the starting index

        # Positive dispartities are caught on the "largest" index, closer to 1080
        # positive_ends = positive_disparities
        positive_starts = np.clip(positive_disparities - num_pts_pos, 0, self._scan_params['num_points'])
            # ^ To ensure we do not start at a negative index
        for start, end in zip(positive_starts, positive_disparities):
            mask = extended_ranges[start:end] > ranges[end]
            extended_ranges[start:end][mask] = ranges[end]
                # We always want to rewrite to the value at the disparity
                # In the positive case, this is the ending index

        return disparities, extended_ranges
    
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
        best_index = np.argmax(ranges)
        # Expand left and right until the values drop below 90% of the maximum
        threshold = 0.95 * ranges[best_index]
        # threshold = 0.95 * ranges[best_index]
        left = best_index
        right = best_index
        while left > 0 and ranges[left - 1] >= threshold:
            left -= 1
        while right < len(ranges) - 1 and ranges[right + 1] >= threshold:
            right += 1
        middle = (left + right) // 2
        return middle

    def find_90thperc_mean(self, ranges):
        return np.average( 
            np.where(
                ranges > (0.95 * np.max(ranges))
            ) 
        ).astype(int)
    
    def find_gaussian_max(self, ranges):
        ranges = gaussian_filter1d(ranges, sigma = 1, mode='wrap')
        return np.argmax(ranges)
    
    def publish_drive_command(self, ext_ranges, full_ranges, deep_index):
        """
        Publish an AckermannDriveStamped command message to the '/drive' topic.d
        """   
        # # Ackermann Steering Angle Calculation:
        """
        - Based on curvature of the path and the wheelbase of the car
        - The formula for the steering angle is:
              steering_angle = atan(wheelbase * curvature)
              curvature = 2 * target_distance * sin(target_angle) / target_distance^2
         - target_distance cancels out of the numerator, the calculation is simplified to:
              steering_angle = atan(wheelbase * 2 * sin(target_angle) / target_distance)
        - The width of the track constrains target_distance*cos(target_angle) to <= forward_distance
        - The updated value of the hypotenuse is:
             hypotenuse = forward_distance / cos(target_angle)
        """
        forward_distance = min(
                np.min(
                    full_ranges[
                        self._scan_params['num_points']//2 - 25 : self._scan_params['num_points']//2 + 25 # type: ignore
                        ]
                    ), 
                self.lookahead_distance)   # The distance directly in front of the car

        target_distance = min(full_ranges[deep_index], self.lookahead_distance)  # The distance to the target point
        target_angle = self._scan_params['angle_min'] + deep_index * self._scan_params['angle_increment'] # type: ignore  # The angle to the target point
        new_target_distance = forward_distance / np.cos(target_angle)
        if new_target_distance < target_distance:
            target_distance = new_target_distance

        theoretical_steering_angle = np.arctan(
            self.wheelbase * 2 * np.sin(target_angle) / target_distance
        )

        # if np.sign(theoretical_steering_angle): # Left
        #     mask = np.zeros(len(full_ranges))
        #     mask[3*len(full_ranges)//4 : 3*len(full_ranges)//4 + 300] = 1
        #     masked_ranges = full_ranges * mask
        #     # Check if any non-zero masked ranges are less than 0.25m
        #     if np.any((masked_ranges > 0) & (masked_ranges < 0.25)):
        #         print("Wall too close on left side")
        #         theoretical_steering_angle = 0.0
        # else:
        #     mask = np.zeros(len(full_ranges))
        #     mask[(len(full_ranges)//4 - 300):len(full_ranges)//4] = 1
        #     masked_ranges = full_ranges * mask
        #     # Check if any non-zero masked ranges are less than 0.25m
        #     if np.any((masked_ranges > 0) & (masked_ranges < 0.25)):
        #         print("Wall too close on right side")
        #         theoretical_steering_angle = 0.0

        bounded_steering_angle = max(min(theoretical_steering_angle, 0.34), -0.34)

        # # Parametrized Logistic Curve Speed Profile
        # Operates as a function of forward_distance (absolute)
        # Parameters

        forward_distance = min(
                np.max(
                    full_ranges[
                        self._scan_params['num_points']//2 - 25 : self._scan_params['num_points']//2 + 25 # type: ignore
                        ]
                ),
            8.0
        )
        speed_max = 4.0
        speed_min = 1.0
        accel = 0.75
        a_center = 3.0
        speed = (
                (speed_max - speed_min) 
                / (1 + np.exp(- accel * (forward_distance - a_center))) 
                + speed_min
        )

        # From DeepSeek, speed filtering, proportional low-pass
        """
        # In your planning node
        prev_speed = 0.0
        filter_alpha = 0.3  # Adjust between 0.0 (no filtering) and 1.0 (no change)

        # After calculating speed
        filtered_speed = filter_alpha * speed + (1 - filter_alpha) * prev_speed
        prev_speed = filtered_speed
        # Use filtered_speed as your 
        """

        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = "base_link"
        
        drive_msg.drive.steering_angle = bounded_steering_angle
        drive_msg.drive.speed = speed

        # TEST MODE: SPEED ENABLE
        if self.enable_speed:
            self.drive_pub.publish(drive_msg)

        if self.enable_logging:
            if not self.enable_speed:
                self.get_logger().info(
                    f"""
        --- TEST MODE: SPEED DISABLED --- 
Target Location: {target_distance:.2f} m, {np.rad2deg(-target_angle):.2f} deg, 
Published Steering: {np.rad2deg(-bounded_steering_angle):.2f} deg, 
Speed: {speed:.2f} m/s
                """
                )
            else:
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
        
        disparities_values = np.zeros(self._scan_params['num_points']) # type: ignore
        disparities_values[disparities] = ranges[disparities]

        disparities_values = np.flip(np.roll(
            disparities_values, -len(disparities_values)//2)).tolist()
        
        modified_scan = LaserScan()
        modified_scan.header.stamp = raw_scan_data.header.stamp
        modified_scan.header.frame_id = 'laser' 
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

        ranges = np.flip(np.roll(ranges, -self._scan_params['num_points']//2)).tolist() # type: ignore

        modified_scan = copy.deepcopy(raw_scan_data)
        modified_scan.header.frame_id = 'laser'
        modified_scan.ranges = ranges 
       
        # modified_scan = LaserScan()
        # modified_scan.header.stamp = raw_scan_data.header.stamp
        # modified_scan.header.frame_id = 'laser'
        # modified_scan.angle_min = raw_scan_data.angle_min
        # modified_scan.angle_max = raw_scan_data.angle_max
        # modified_scan.angle_increment = raw_scan_data.angle_increment
        # modified_scan.scan_time = raw_scan_data.scan_time
        # modified_scan.range_min = raw_scan_data.range_min
        # modified_scan.range_max = raw_scan_data.range_max
        # modified_scan.ranges = ranges
        # modified_scan.intensities = raw_scan_data.intensities

        self.ext_scan_publisher.publish(modified_scan)

    def brake_loop(self):
        """Send brake commands for 5 seconds"""
        if self.brake_counter > 0:  # 50 * 0.1s = 5 seconds
            brake_msg = Float64()
            brake_msg.data = 20000.0
            self.emergency_stop_pub.publish(brake_msg)
            self.brake_counter -= 1
        else:
            self.brake_timer.cancel()  # Stop the timer
            self.get_logger().info(" --- Brake released. --- ")

    # def _initialize_proximity_lut(self):
    #     """Initialize lookup table for wall proximity checks"""
    #     if self._proximity_lut is not None:
    #         return
        
    #     search_width_angle = np.arctan(25/18)
    #     lut_indices = np.linspace(np.pi/2, np.pi/2+search_width_angle, self._scan_params['angle_increment']) # type: ignore
    #     self._proximity_lut = 0.18 * np.sin(lut_indices)

def main(args=None):
    rclpy.init(args=args)
    node = disparityExtender()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
