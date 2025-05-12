import numpy as np

# LUT for number of points to extend based on disparity distance
# The number of points to rewrite is given by the function:
# Points(disparity_distance) = floor( arctan( extension_distance / disparity_distance ) / angle_increment )
# The parameters extension_distance and angle_increment are passed in on initialization

class ExtensionLUT:
    def __init__(self, extension_distance, angle_increment):
        """
        Initializes the ExtensionLUT with the given extension distance and angle increment.

        Parameters:
            extension_distance (float): The distance to extend the disparities.
            angle_increment (float): The angle increment for the LiDAR scan.
        """
        self.extension_distance = extension_distance
        self.angle_increment = angle_increment
        self.lut = self.create_lut()

    def create_lut(self):
        """
        Creates a lookup table for the number of points to extend based on disparity distance.

        Returns:
            np.array: The lookup table for number of points to extend.
        """
        # Disparity distances are constrained to 0.15 to 10 meters
        disparity_distances = np.linspace(0.15, 10, num=1000)

        # Calculate the number of points to extend for each disparity distance
        points_to_extend = np.floor(np.arctan(self.extension_distance / disparity_distances) / self.angle_increment).astype(int)

        return points_to_extend