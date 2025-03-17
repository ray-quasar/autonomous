import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

# This is a test edit

class FollowGapDisparity(Node):
    def __init__(self):
        super().__init__('follow_gap_disparity')
        # Subscribe to LiDAR scans (assumed on '/scan')
        self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        # Publisher for AckermannDriveStamped messages on '/drive'
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        
        # Parameters
        self.safe_distance = 1.0         # Mask out obstacles closer than 1 m.
        self.disparity_threshold = 0.3   # Disparity threshold (meters)
        self.bias_factor = 0.1            # Fraction of (center - midpoint) to shift the target
        
        # Drive parameters
        self.base_speed = 0.75           # Base speed (m/s) when turning mildly

    def lidar_callback(self, scan):
        # Convert raw scan data to a NumPy array and clip to valid range.
        ranges = np.array(scan.ranges)  # Ignore the angle_min, angle_increment, etc.
        angles = np.linspace(scan.angle_min, scan.angle_max, len(ranges))

        # Filter out invalid values (NaNs, infs, etc.).
        # ranges = np.clip(ranges, scan.range_min, scan.range_max)

        ranges[np.isnan(ranges)] = scan.range_max
        ranges[np.isinf(ranges)] = scan.range_max
        
        # I'm not sure this is necessary, but it's in the original code.
        # Ignore readings that are too close.
        #ranges[ranges < self.safe_distance] = 0.0

        # --- Remove values behind the car entirely ---
        # ROS LiDAR scans are in the order of -pi to pi, so we can filter out the rear 180 degrees.
        # The scan is symmetric, so we can just keep the middle two quartiles
        # (i.e., the middle half of the scan).
        valid_ranges = np.copy(ranges)
        valid_ranges = valid_ranges[len(valid_ranges)//4:3*len(valid_ranges)//4]

        valid_angles = np.copy(angles)
        valid_angles = valid_angles[len(valid_angles)//4:3*len(valid_angles)//4]

        
        # The Disparity Extender algorithm is broken, so we'll just use the valid_ranges for now
        # Apply the disparity extender on the filtered (front-facing) ranges.
        #ext_ranges = self.extend_disparities(valid_ranges)
        ext_ranges = valid_ranges
        
        # Find the largest gap in the extended valid ranges.
        gap_start, gap_end = self.find_largest_gap(ext_ranges)

        # Compute the raw midpoint index (as a float) of the gap.
        midpoint = (gap_start + gap_end) / 2.0

        # Define the center of the valid array as the index with angle closest to 0.
        center = np.argmin(np.abs(valid_angles))
        
        # Bias the best point away from the wall.
        if midpoint < center:
            bias = self.bias_factor * (center - midpoint)
            best_point = midpoint + bias
        else:
            bias = self.bias_factor * (midpoint - center)
            best_point = midpoint - bias
        
        # Convert the chosen best point index into a steering angle.
        # Because valid_angles are uniformly spaced, we can simply take the angle at the rounded index.
        best_index = int(round(best_point))
        # Ensure the index is within bounds.
        best_index = max(0, min(best_index, len(valid_angles) - 1))
        steering_angle = valid_angles[best_index]
        
        self.publish_command(steering_angle)

    def extend_disparities(self, scan):
        """
        Implements the disparity extender algorithm:
        For each adjacent pair of valid measurements, if their absolute difference
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
        """Return the start and end indices of the largest contiguous segment (nonzero values)."""
        gaps = []  # List to store the start and end indices of each gap
        gap_start = None  # Variable to mark the start of a gap

        # Iterate through the ranges to identify gaps
        for i, r in enumerate(ranges):
            if r > 0 and gap_start is None:
                # Start of a new gap
                gap_start = i
            elif r == 0 and gap_start is not None:
                # End of the current gap
                gaps.append((gap_start, i - 1))
                gap_start = None

        # If the last gap reaches the end of the ranges, close it
        if gap_start is not None:
            gaps.append((gap_start, len(ranges) - 1))

        # If no gaps are found, return the entire range
        if not gaps:
            return (0, len(ranges) - 1)

        # Find the largest gap by comparing the lengths of all gaps
        largest_gap = max(gaps, key=lambda g: g[1] - g[0])
        return largest_gap

    def publish_command(self, steering_angle):
        """Publish an AckermannDriveStamped command with limited steering angle."""
        # Limit the steering angle to ±0.34 radians.
        steering_angle = max(min(steering_angle, 0.34), -0.34)
        
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = "base_link"
        
        # Map steering angle to speed: slower during sharper turns.
        speed = max(0.75, self.base_speed - abs(steering_angle) * 2.0)
        
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = speed
        
        self.drive_pub.publish(drive_msg)
        self.get_logger().info(
            f"Steering: {steering_angle:.2f} rad, Speed: {speed:.2f} m/s"
        )

def main(args=None):
    rclpy.init(args=args)
    node = FollowGapDisparity()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
