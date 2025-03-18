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
        self.extension_distance = 0.20

        # Threshold for detecting disparities (in meters)
        self.disparity_check = 0.5    

        # Base speed (m/s) on straightaways
        self.base_speed = 0.75      

        # Maximum steering angle (radians)
        self.max_steering_angle = 0.34     
    
    # Main LiDAR scan processing function
    def lidar_callback(self, scan):
        """
        Callback function for processing incoming LiDAR scan data.
        """
        # Convert raw scan data to a NumPy array
        ranges = np.array(scan.ranges)  # Ignore the angle_min, angle_increment, etc.

        # Preprocess the scan data
        ranges = np.clip(ranges, scan.range_min, scan.range_max)
        ranges = np.nan_to_num(ranges)
        ranges[:len(ranges)//4] = 0
        ranges[3*len(ranges)//4:] = 0

        # Find disparities in the LiDAR scan data
        disparities = self.find_disparities(ranges, self.disparity_check)

        # Extend disparities in the LiDAR scan data
        ranges = self.extend_disparities(ranges, disparities, self.extension_distance, scan.angle_increment)

        # Find the index of the deepest point in the LiDAR scan data
        deep_index = self.find_deepest_index(ranges)

        # Determine the steering angle and speed
        steering_angle = scan.angle_min + deep_index * scan.angle_increment
        steering_angle = max(min(steering_angle, 0.34), -0.34)
        
        speed = self.base_speed

        self.publish_command(steering_angle, speed)

    # Helper functions

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
    
    def extend_disparities(self, ranges, disparities, angle_increment):
        for i in disparities:
            # triangle_height = ranges[i]
            # triangle_base = self.extension_distance
            angle_to_extend = np.atan(self.extension_distance / ranges[i])``
            points_to_rewrite = int(angle_to_extend / angle_increment)
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
    
    def find_deepest_index(self, ranges):
        deep_index = np.argmax(ranges)    
        return deep_index

    def publish_command(self, steering_angle, speed):
        """
        Publish an AckermannDriveStamped command message.
        """        
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
