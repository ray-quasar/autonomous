import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

class disparityExtender(Node):
    def __init__(self):
        super().__init__('disparityExtender')
        # Subscribe to LiDAR scans (assumed on '/scan')
        self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        # Publisher for AckermannDriveStamped messages on '/drive'
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        
        # Threshold for extending disparities (in meters)
        # This should ideally be half the width of the car 
        self.extension_distance = 0.20

        # Threshold for detecting disparities (in meters)
        self.disparity_check = 0.5    

        # Base speed (m/s) on straightaways
        self.base_speed = 0.75           

    def lidar_callback(self, scan):
        """
        Callback function for processing incoming LiDAR scan data.
        """
        # Convert raw scan data to a NumPy array
        ranges = np.array(scan.ranges)  # Ignore the angle_min, angle_increment, etc.

        # Remove values behind the car entirely
        valid_ranges = np.copy(ranges)
        valid_ranges = valid_ranges[len(valid_ranges)//4:3*len(valid_ranges)//4]

        
        
        # Find the largest gap in the extended valid ranges
        gap_start, gap_end = self.find_largest_gap(ext_ranges)

        # Calculate the steering angle and speed
        # The steering angle should be the average angle of the largest gap
        # The speed should be the base speed if the gap is wide, and slower if it's narrow
        # Speed should also be moderated by the depth of the gap and the steering angle
        # The steering angle should be positive if the gap is to the left, and negative if to the right
        # If there is no gap, stop the car
        steering_angle = 0.0
        speed = 0.0

        self.publish_command(steering_angle, speed)

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

    def publish_command(self, steering_angle, speed):
        """
        Publish an AckermannDriveStamped command with limited steering angle.
        """
        # Limit the steering angle to +/- 0.34 radians (20 degrees)
        # This should really be a parameter, but we'll hard-code it for now

        steering_angle = max(min(steering_angle, 0.34), -0.34)
        
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = "base_link"
        
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = speed
        
        self.drive_pub.publish(drive_msg)
        self.get_logger().info(
            f"Steering: {steering_angle:.2f} rad, Speed: {speed:.2f} m/s"
        )

def main(args=None):
    rclpy.init(args=args)
    node = disparityExtender()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
