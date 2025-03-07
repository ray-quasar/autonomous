import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

class FollowGapPID(Node):
    def __init__(self):
        super().__init__('follow_gap_pid')
        # Subscribe to LiDAR scans and odometry
        self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        # Publisher for velocity commands
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Parameter: Ignore points closer than safe_distance (in meters)
        self.safe_distance = 3.0
        
        # PID controller parameters for angular velocity control
        self.kp = 0.3    # Lowered proportional gain for gentler response
        self.ki = 0.0    # Integral gain remains zero (enable if needed)
        self.kd = 0.4    # Increased derivative gain to damp oscillations
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = self.get_clock().now()
        
        # Measured angular velocity from odometry (in rad/s)
        self.current_angular_z = 0.0

    def odom_callback(self, msg):
        # Update the measured angular velocity from odometry (in rad/s)
        self.current_angular_z = msg.twist.twist.angular.z

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
        # Convert the best point into an angle.
        angle_increment = scan.angle_increment
        desired_steering = ((best_point - len(ranges) // 2) * angle_increment)
        
        # Interpret desired_steering as desired angular velocity (you may add a scaling factor if needed)
        desired_angular_velocity = 0.5 * desired_steering
        
        # Compute the PID output based on error between desired and measured angular velocity.
        pid_output = self.pid_control(desired_angular_velocity)
        # Publish the command.
        self.publish_command(pid_output)

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

    def pid_control(self, desired_angular_velocity):
        """Compute the PID control signal based on the error between desired and measured angular velocity."""
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds * 1e-9  # dt in seconds
        # Clamp dt to a minimum value to avoid huge derivative spikes.
        if dt < 0.05:
            dt = 0.05
        self.last_time = current_time

        # Compute error between desired and measured angular velocity.
        error = desired_angular_velocity - self.current_angular_z

        # Update the integral term with anti-windup.
        self.integral += error * dt
        max_integral = 1.0  # maximum allowed integral term
        self.integral = max(min(self.integral, max_integral), -max_integral)

        # Compute the derivative term.
        derivative = (error - self.last_error) / dt

        # Compute PID output.
        output = self.kp * error + self.ki * self.integral + self.kd * derivative

        # Update last error.
        self.last_error = error

        # Deadband: if the error is very small, command zero angular velocity.
        deadband = 0.005  # rad/s threshold
        if abs(error) < deadband:
            output = 0.0
            self.integral = 0.0
            self.last_error = 0.0

        # Clamp the output to a maximum angular velocity.
        max_output = 1.0  # rad/s; adjust as appropriate for your vehicle.
        output = max(min(output, max_output), -max_output)

        self.get_logger().info(
            f"Desired: {desired_angular_velocity:.2f} rad/s, "
            f"Measured: {self.current_angular_z:.2f} rad/s, "
            f"Error: {error:.2f}, PID: {output:.2f}"
        )
        return output

    def publish_command(self, angular_velocity):
        """Publish a velocity command with linear speed adjusted by the magnitude of the angular command."""
        cmd = Twist()
        # Adjust linear speed: slower during sharper turns.
        cmd.linear.x = float(max(1.0, 3.0 - abs(angular_velocity) * 2))
        cmd.angular.z = float(angular_velocity)
        self.cmd_pub.publish(cmd)
        self.get_logger().info(
            f"Published cmd: linear {cmd.linear.x:.2f} m/s, angular {cmd.angular.z:.2f} rad/s"
        )

def main(args=None):
    rclpy.init(args=args)
    node = FollowGapPID()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()