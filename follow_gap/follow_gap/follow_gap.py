import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
# We don't need Twist or Odometry anymore, so they're removed
# from geometry_msgs.msg import Twist
# from nav_msgs.msg import Odometry

# Import AckermannDriveStamped
from ackermann_msgs.msg import AckermannDriveStamped

class FollowGapPID(Node):
    def __init__(self):
        super().__init__('follow_gap_pid')
        # Subscribe to LiDAR scans
        self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        
        # Publisher for AckermannDriveStamped messages on '/drive'
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        
        # Parameter: Ignore points closer than safe_distance (in meters)
        self.safe_distance = 3.0

    def lidar_callback(self, scan):
        # Convert laser scan ranges into a NumPy array and clip to valid values.
        ranges = np.array(scan.ranges)
        ranges = np.clip(ranges, scan.range_min, scan.range_max)
        ranges[np.isnan(ranges)] = 0
        ranges[np.isinf(ranges)] = 0
        # Ignore readings that are too close.
        ranges[ranges < self.safe_distance] = 0

        # Find the largest gap in the scan.
        gap_start, gap_end = self.find_largest_gap(ranges)
        # Select the midpoint of the gap as our target.
        best_point = int((gap_start + gap_end) * 0.5)
        # Convert the best point into a steering angle (radians).
        angle_increment = scan.angle_increment
        steering_angle = ((best_point - len(ranges) // 2) * angle_increment)

        # Publish the Ackermann drive command.
        self.publish_command(steering_angle)

    def find_largest_gap(self, ranges):
        """Return the start and end indices of the largest contiguous segment (nonzero values)."""
        gaps = []
        gap_start = None
        for i, distance in enumerate(ranges):
            if distance > 0 and gap_start is None:
                gap_start = i
            elif distance == 0 and gap_start is not None:
                gaps.append((gap_start, i - 1))
                gap_start = None
        if gap_start is not None:
            gaps.append((gap_start, len(ranges) - 1))
        if not gaps:
            return (0, len(ranges) - 1)
        largest_gap = max(gaps, key=lambda x: x[1] - x[0])
        return largest_gap

    def publish_command(self, steering_angle):
        """Publish an AckermannDriveStamped command with a speed that decreases for larger steering angles."""
        cmd = AckermannDriveStamped()
        # Add a header if needed
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = "base_link"

        # For a simple approach, reduce speed as the absolute steering angle grows.
        speed = max(1.0, 3.0 - abs(steering_angle) * 2.0)

        # Assign values to the Ackermann drive message
        cmd.drive.steering_angle = steering_angle
        cmd.drive.speed = speed

        # Publish on '/drive'
        self.drive_pub.publish(cmd)
        self.get_logger().info(f"Steering: {steering_angle:.2f} rad, Speed: {speed:.2f} m/s")

def main(args=None):
    rclpy.init(args=args)
    node = FollowGapPID()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
