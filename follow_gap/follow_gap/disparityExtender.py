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
        self.disparity_threshold = 0.5    

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

    def find_largest_gap(self, ranges):
        """
        Identify the largest contiguous segment of nonzero values in the ranges.
        """
        largest_gap = (0, 0)
        return largest_gap

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
