import numpy as np
import rclpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rclpy.node import Node
from scan_simulator_2d import PyScanSimulator2D
from tf_transformations import euler_from_quaternion
from nav_msgs.msg import OccupancyGrid
import sys

np.set_printoptions(threshold=sys.maxsize)

class SensorModel:
    def __init__(self, node):
        self.node = node # delete this later
        node.declare_parameter('map_topic', "default")
        node.declare_parameter('num_beams_per_particle', 1)
        node.declare_parameter('scan_theta_discretization', 1.0)
        node.declare_parameter('scan_field_of_view', 1.0)
        node.declare_parameter('lidar_scale_to_map_scale', 1.0)

        self.map_topic = node.get_parameter('map_topic').get_parameter_value().string_value
        self.num_beams_per_particle = node.get_parameter('num_beams_per_particle').get_parameter_value().integer_value
        self.scan_theta_discretization = node.get_parameter(
            'scan_theta_discretization').get_parameter_value().double_value
        self.scan_field_of_view = node.get_parameter('scan_field_of_view').get_parameter_value().double_value
        self.lidar_scale_to_map_scale = node.get_parameter(
            'lidar_scale_to_map_scale').get_parameter_value().double_value

        # Sensor model parameters
        self.alpha_hit = 0.74
        self.alpha_short = 0.07
        self.alpha_max = 0.07
        self.alpha_rand = 0.12
        self.sigma_hit = 8.0
        self.z_max = 200.0

        self.table_width = 201

        node.get_logger().info("%s" % self.map_topic)

        self.sensor_model_table = np.empty((self.table_width, self.table_width))
        self.precompute_sensor_model()

        self.scan_sim = PyScanSimulator2D(
            self.num_beams_per_particle,
            self.scan_field_of_view,
            0,
            0.01,
            self.scan_theta_discretization)

        self.map = None
        self.map_set = False
        self.map_subscriber = node.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_callback,
            1)

    def precompute_sensor_model(self):
        z_values = np.linspace(0, self.z_max, self.table_width).reshape(-1, 1) # column vector
        d_values = np.linspace(0, self.z_max, self.table_width).reshape(1, -1) # row vector

        self.node.get_logger().info(f"z values:{z_values}")


        # Precompute the Gaussian function (for p_hit)
        gaussian_kernel = np.where((z_values >= 0) & (z_values <= self.z_max), np.exp(-((z_values - d_values)**2) / (2 * self.sigma_hit**2)) / (np.sqrt(2 * np.pi * self.sigma_hit**2)), 0)
        self.node.get_logger().info(f"gaussian_kernel shape:{gaussian_kernel.shape}")

        sum_hit = np.sum(gaussian_kernel, axis=0).reshape(1,-1) # sum column

        # normalize p_hit
        p_hit = gaussian_kernel / sum_hit

        p_short = np.where((z_values<= d_values) & (d_values != 0) & (z_values >= 0), 2 / d_values * (1 - z_values/d_values), 0)
        p_max = np.where((z_values == self.z_max), 1, 0)
        p_rand = np.where((z_values >= 0) & (z_values <= self.z_max), 1/self.z_max, 0)

        # Combine all components to form the sensor model table (using broadcasting)
        sensor_model = (self.alpha_hit * p_hit +
                        self.alpha_short * p_short +
                        self.alpha_max * p_max +
                        self.alpha_rand * p_rand)

        # Normalize the table
        sensor_model /= np.sum(sensor_model, axis=0)

        self.sensor_model_table = sensor_model


    def evaluate(self, particles, observation):
        """
        Evaluate how likely each particle is given
        the observed scan.

        args:
            particles: An Nx3 matrix of the form:

                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            observation: A vector of lidar data measured
                from the actual lidar. THIS IS Z_K. Each range in Z_K is Z_K^i

        returns:
           probabilities: A vector of length N representing
               the probability of each particle existing
               given the observation and the map.
        """
        if not self.map_set:
            return np.ones(len(particles))/len(particles)

        scans = self.scan_sim.scan(particles) / (self.resolution * self.lidar_scale_to_map_scale)
        observation = np.clip(observation / (self.resolution * self.lidar_scale_to_map_scale), 0, self.z_max)
        scans = np.clip(scans, 0, self.z_max)

        self.node.get_logger().info(f"scan shape:{scans.shape}")
        self.node.get_logger().info(f"observation shape:{observation.shape}")


        scan_indices = np.round(scans * (self.table_width - 1) / self.z_max).astype(int)
        obs_indices = np.round(observation * (self.table_width - 1) / self.z_max).astype(int)

        # new 2 lines from chat:
        scan_indices = np.clip(scan_indices, 0, self.table_width - 1)
        obs_indices = np.clip(obs_indices, 0, self.table_width - 1)

        self.node.get_logger().info(f"scan_indices shape:{scan_indices.shape}")
        self.node.get_logger().info(f"obs_indices shape:{obs_indices.shape}")

        probabilities = np.prod(self.sensor_model_table[obs_indices, scan_indices], axis=-1)

        # probabilities = np.zeros(particles.shape[0])

        # # Iterate over each particle
        # for i in range(particles.shape[0]):
        #     probabilities[i] = self.sensor_model_table[obs_indices[i], scan_indices[i]]

        self.node.get_logger().info(f"probabilities shape:{probabilities.shape}")

        return probabilities

    def map_callback(self, map_msg):
        self.map = np.array(map_msg.data, np.double) / 100.
        self.map = np.clip(self.map, 0, 1)
        self.resolution = map_msg.info.resolution

        origin_p = map_msg.info.origin.position
        origin_o = map_msg.info.origin.orientation
        origin_o = euler_from_quaternion((
            origin_o.x,
            origin_o.y,
            origin_o.z,
            origin_o.w))
        origin = (origin_p.x, origin_p.y, origin_o[2])

        self.scan_sim.set_map(
            self.map,
            map_msg.info.height,
            map_msg.info.width,
            map_msg.info.resolution,
            origin,
            0.5)
        self.map_set = True
        print("Map initialized")

    def sensor_model_plot(self):
        fig = plt.figure()
        z_vals = np.linspace(0, self.z_max, self.table_width)
        d_vals = np.linspace(0, self.z_max, self.table_width)
        Z, D = np.meshgrid(z_vals, d_vals)
        P = self.sensor_model_table


        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(D, Z, P, cmap='viridis')
        ax.set_xlabel('Expected Distance d')
        ax.set_ylabel('Measured Distance z')
        ax.set_zlabel('Probability p(z | d)')
        ax.set_title('Sensor Model 3D Plot')
        plt.show()
