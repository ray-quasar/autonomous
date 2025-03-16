import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

class FollowGapDisparity(Node):
    def __init__(self):
        super().__init__('follow_gap_disparity')
        # Subscribe to LiDAR scans (assumed on '/scan')
        self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        # Publisher for AckermannDriveStamped messages on '/drive'
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        
        # Parameters
        self.safe_distance = 3.0         # Mask out obstacles closer than 3 m.
        self.disparity_threshold = 0.33   # Disparity threshold (meters)
        self.bias_factor = 0.5           # Fraction of (center - midpoint) to shift the target
        
        # Drive parameters
        self.base_speed = 0.75            # Base speed (m/s) when turning mildly

    def lidar_callback(self, scan):
        # Convert raw scan data to NumPy array and clean it up.
        ranges = np.array(scan.ranges)
        ranges = np.clip(ranges, scan.range_min, scan.range_max)
        ranges[np.isnan(ranges)] = scan.range_max
        ranges[np.isinf(ranges)] = scan.range_max
        
        # Mask out any readings that are too close.
        ranges[ranges < self.safe_distance] = 0
        
        # Apply the disparity extender.
        ext_ranges = self.extend_disparities(ranges)
        
        # Find the largest gap in the extended scan.
        gap_start, gap_end = self.find_largest_gap(ext_ranges)
        # Compute the (raw) midpoint of the gap.
        midpoint = (gap_start + gap_end) / 2.0
        center = len(ranges) / 2.0
        
        # Bias the best point away from the wall:
        # If midpoint is left of center, assume wall is on the left, so shift target right.
        # If midpoint is right of center, shift target left.
        if midpoint < center:
            bias = self.bias_factor * (center - midpoint)
            best_point = midpoint + bias
        else:
            bias = self.bias_factor * (midpoint - center)
            best_point = midpoint - bias
        
        # Convert the chosen best_point index into a steering angle.
        angle_increment = scan.angle_increment
        steering_angle = (best_point - center) * angle_increment
        
        self.publish_command(steering_angle)

    def extend_disparities(self, scan):
        """
        Implements the disparity extender algorithm:
          For each pair of adjacent valid measurements, if the absolute difference 
          exceeds disparity_threshold, replace the smaller (closer) measurement with 
          the larger (farther) measurement.
        """
        ext = np.copy(scan)
        for i in range(len(scan) - 1):
            if scan[i] > 0 and scan[i+1] > 0:
                if abs(scan[i+1] - scan[i]) > self.disparity_threshold:
                    if scan[i] < scan[i+1]:
                        ext[i] = scan[i+1]
                    else:
                        ext[i+1] = scan[i]
        return ext

    def find_largest_gap(self, ranges):
        """Return (start, end) indices of the largest contiguous segment with nonzero readings."""
        gaps = []
        gap_start = None
        for i, r in enumerate(ranges):
            if r > 0 and gap_start is None:
                gap_start = i
            elif r == 0 and gap_start is not None:
                gaps.append((gap_start, i - 1))
                gap_start = None
        if gap_start is not None:
            gaps.append((gap_start, len(ranges) - 1))
        if not gaps:
            return (0, len(ranges) - 1)
        largest_gap = max(gaps, key=lambda g: g[1] - g[0])
        return largest_gap

    def publish_command(self, steering_angle):
        """
        Publish an AckermannDriveStamped command.
        The steering_angle is set directly; you may adjust speed as a function of steering.
        """
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = "base_link"
        
        # Map steering angle to speed: for larger absolute steering, reduce speed.
        # Here we use a simple linear mapping; you might use a non-linear (exponential) mapping if needed.
        speed = max(1.0, self.base_speed - abs(steering_angle) * 2.0)
        
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = speed
        
        self.drive_pub.publish(drive_msg)
        self.get_logger().info(
            f"Best Point: (biased), Steering: {steering_angle:.2f} rad, Speed: {speed:.2f} m/s"
        )

def main(args=None):
    rclpy.init(args=args)
    node = FollowGapDisparity()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

