import numpy as np
import time

class MotionModel:
    def __init__(self, node):
        """
        Initialize the motion model.
        """
        self.node = node

        # Noise parameters (tune these values based on experimental data)
        self.noise_std_x = 0.1       # Standard deviation for x movement
        self.noise_std_y = 0.05      # Standard deviation for y movement
        self.noise_std_theta = 0.1   # Standard deviation for rotation
        self.prev_t = None

    def evaluate(self, particles, odometry):
        """
        Update the particles based on odometry data with added noise.

        Args:
            particles: An Nx3 matrix of the form:
                [x0 y0 theta0]
                [x1 y1 theta1]
                [...]
            odometry: A 3-vector [dx, dy, dtheta] in the robot's local frame.

        Returns:
            Updated particle matrix of the same size.
        """
        # Extract odometry values

        dx, dy, dtheta = odometry
        dx += np.random.normal(0,self.noise_std_x, (len(particles),))
        dy += np.random.normal(0,self.noise_std_y, (len(particles),))
        dtheta += np.random.normal(0,self.noise_std_theta, (len(particles),))
        # Extract particle states
        x = particles[:, 0]
        y = particles[:, 1]
        theta = particles[:, 2]

        # Apply transformation in the world frame
        x_new = np.cos(theta) * dx - np.sin(theta) * dy + x
        y_new = np.sin(theta) * dx + np.cos(theta) * dy + y
        theta += dtheta

        # Add Gaussian noise
        # x_new += np.random.normal(0, self.noise_std_x, size=particles.shape[0])
        # y_new += np.random.normal(0, self.noise_std_y, size=particles.shape[0])
        # theta_new += np.random.normal(0, self.noise_std_theta, size=particles.shape[0])

        # Keep angles within [-pi, pi]
        theta_new = (theta + np.pi) % (2 * np.pi) - np.pi

        # Update particles
        particles[:, 0] = x_new
        particles[:, 1] = y_new
        particles[:, 2] = theta_new

        return particles
