import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import Joy

class disparityExtender(Node):
    def __init__(self):
        super().__init__('disparityExtender')
        # Subscribe to LiDAR scans (assumed on '/scan')
        self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        # Publisher for AckermannDriveStamped messages on '/drive'
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        # Publisher for modified range data
        self.ext_scan_publisher = self.create_publisher(LaserScan, '/ext_scan', 10)

        # Wheelbase of the car (in meters)
        self.wheelbase = 0.325
        
        # Threshold for extending disparities (in meters)
        self.extension_distance = 0.15

        # Threshold for detecting disparities (in meters)
        self.disparity_check = 0.65    

        # Base speed (m/s) on straightaways
        self.base_speed = 0.0

        # Maximum steering angle (radians)
        self.max_steering_angle = 0.34     

        # Deadman switch state
        self.deadman_active = False

        # Subscribe to joystick messages to monitor RB button (button 7)
        self.joy_sub = self.create_subscription(
            Joy, '/joy', self.joy_callback, 10
        )
    
    # Main LiDAR scan processing function
    def lidar_callback(self, scan):
        """
        Callback function for processing incoming LiDAR scan data.
        """
        # Convert raw scan data to a NumPy array
        ranges = np.array(scan.ranges)  # Ignore the angle_min, angle_increment, etc.

        # Rotate the scan data about z-axis
        ranges = np.roll(ranges, len(ranges)//2)
        # Rotate the scan data about x-axis
        ranges = np.flip(ranges)

        # Preprocess the scan data
        # ranges = np.clip(ranges, scan.range_min, scan.range_max)
        ranges = np.clip(ranges, scan.range_min, 4.0)
        ranges = np.nan_to_num(ranges, nan=0.0)
        ranges[:len(ranges)//4] = 0.0
        ranges[3*len(ranges)//4:] = 0.0
        # self.occlude_ranges(ranges, 180.0, 180.0

        # Find disparities in the LiDAR scan data
        disparities = self.find_disparities(ranges, self.disparity_check)

        # Extend disparities in the LiDAR scan data
        ranges = self.extend_disparities(ranges, disparities, scan.angle_increment)

        # Find the index of the deepest point in the LiDAR scan data
        deep_index = self.find_deepest_gap(ranges)

        # Determine the steering angle and speed
        # target_angle = scan.angle_min + deep_index * scan.angle_increment
        # steering_angle = max(min(steering_angle, 0.34), -0.34)

        # speed = self.base_speed

        self.publish_drive_command(scan, ranges, deep_index)
        # self.publish_drive_command(steering_angle, speed)
        self.publish_laser_scan(ranges, scan)

    # Helper functions

    def occlude_ranges(ranges, fov_size, fov_center):
        """
        Occlude the ranges to a specified FOV.

        Parameters:
            ranges (np.array): The range data.
            fov_size (float): The size of the FOV in degrees.
            fov_center (float): The center of the FOV in degrees.
        """
        num_ranges = len(ranges)
        
        # Calculate the start and end indices for the FOV using proportions
        fov_size_proportion = fov_size / 360.0
        fov_center_proportion = fov_center / 360.0
        
        start_index = int((fov_center_proportion - fov_size_proportion / 2) * num_ranges)
        end_index = int((fov_center_proportion + fov_size_proportion / 2) * num_ranges)
        
        # Occlude the ranges outside the FOV
        ranges[:start_index] = 0.0
        ranges[end_index:] = 0.0

    def find_disparities(self, ranges, check_value):
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
    
    #self.extend_disparities(ranges, disparities, self.extension_distance, scan.angle_increment)
    def extend_disparities(self, ranges, disparities, angle_increment):
        for i in disparities:
            # if ranges[i] < 1.0: # Disparities are only extended if the range greater than 1.0
            #     continue
            angle_to_extend = np.arctan(self.extension_distance / ranges[i])
            points_to_rewrite = int(angle_to_extend / angle_increment ) # * ranges[i]) 
                # Multiplying by ranges[i] to prevent disparity extension at close ranges
                # Just the ranges value itself is on the right order to scale the extension it seems

                # I may make it so that the extension is also scaled by how close to the edge of the fov the disparity is
                # This way, the disparity extension is more at the edges of the fov

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
        # Find the index of the maximum range value
        best_index = np.argmax(ranges)
        # Expand left and right until the values drop below 90% of the maximum
        threshold = 0.9 * ranges[best_index]
        left = best_index
        right = best_index
        while left > 0 and ranges[left - 1] >= threshold:
            left -= 1
        while right < len(ranges) - 1 and ranges[right + 1] >= threshold:
            right += 1
        # return the start and end indices of the deepest gap
        # can be easily edited to return the middle of the gap
        middle = (left + right) // 2
        #make middle function
        return middle

    def joy_callback(self, msg):
        """
        Callback to monitor joystick input and update deadman state.
        """
        # Check if RB (button 7) is pressed
        self.deadman_active = msg.buttons[7] == 1

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

        # If the target angle is between -pi/4 and pi/4, we can use the regular formula for the steering angle
        if True: #np.pi / 4 >= target_angle >= -np.pi / 4:
            theoretical_steering_angle = np.arctan(
                self.wheelbase * 2 * np.sin(target_angle) / depth
            )
        # If we are aimed greater than pi/4, we need to truncate the length of the x-component of the vector
        # to get a tighter turn and avoid turning arcs that go through walls
        # The formula for the steering angle is: 
        # steering_angle = atan(wheelbase / (depth * cos(target_angle)))
        # But getting rid of the sin term, we lose the ability to turn negative
        # So we need to add a negative sign to the steering angle if the target angle is negative
        elif np.pi / 4 < target_angle < np.pi:
            theoretical_steering_angle = np.arctan(
                self.wheelbase / (depth * np.cos(target_angle))
            )
        elif -np.pi < target_angle < -np.pi / 4:
            theoretical_steering_angle = np.arctan(
                -self.wheelbase / (depth * np.cos(target_angle))
            )
        

        # Limit the steering angle to the maximum steering angle of the car
        bounded_steering_angle = max(min(theoretical_steering_angle, 0.34), -0.34)
        
        speed = self.base_speed 

        # The speed can be limited by the steering angle and/or the depth of the gap
        # and/or the curvature of the path and/or the distance directly in front of the car

        # Approach 1: Limit speed to 2.0 m/s at full turn, scaling linearly to max at 0.0 rad
        # (speed - 2) is the difference between the base speed and the minimum speed
        # We subtract some portion of that difference from the base speed based 
        # on the proportion of the steering angle to the maximum steering angle
        # speed = speed - (speed - 2.0) * (np.abs(bounded_steering_angle) / self.max_steering_angle) 
        # I don't know if this is enough, the car doesn't actually go full lock when taking the turn
        # Also we definitely need to be slowing down before the turn, not at the apex of the turn
        # It might be a good idea to pass in the forward distance to the car as well (Approach 4)

        # Approach 2: Scaling speed based on the depth of the gap
        # I don't think this is feasible, as the depth of the gap is not a good indicator of speed
        # A deep gap to the side of the car requires a slower speed 

        # Approach 3: Scaling speed based on the curvature of the path
        # The curvature is defined as the inverse of the radius of the turn
        # If this is how we moderate the speed, we should modify the steering angle calc to be a 
        # function of the curvature rather than simplifying it out of the equation like we have been doing
        # The formula for the curvature is:


        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = "base_link"
        
        drive_msg.drive.steering_angle = bounded_steering_angle
        drive_msg.drive.speed = speed
        
        self.drive_pub.publish(drive_msg)
        self.get_logger().info(
            f"""
            Target Location: {target_distance:.2f} m, {np.rad2deg(-target_angle):.2f} deg, 
            Published Steering: {np.rad2deg(-bounded_steering_angle):.2f} deg, 
            Speed: {speed:.2f} m/s
            """
        )

    def publish_laser_scan(self, ranges, raw_scan_data):
        """
        Publish the modified range data to the '/ext_scan' topic.

        Parameters:
            ranges (np.array): Modified range data.
            scan_data (LaserScan): Original LiDAR scan data.
        """
        # Experimental: Fixing position of the scan data in RViz
        # Undo the rotation of the scan data about the z-axis
        ranges = np.roll(ranges, -len(ranges)//2)
        # Undo the rotation of the scan data about the x-axis
        ranges = np.flip(ranges)

        # ROS 2 requires the ranges to be a list, not a NumPy array
        ranges = ranges.tolist()
        
        modified_scan = LaserScan()
        modified_scan.header.stamp = raw_scan_data.header.stamp
        #modified_scan.header.frame_id = 'base_link' # The modified scan data is rotated to the base_link frame
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
        self.ext_scan_publisher = self.create_publisher(
            LaserScan,
            '/ext_scan',  # Topic name for modified range data
            10  # QoS
        )

def main(args=None):
    rclpy.init(args=args)
    node = disparityExtender()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
