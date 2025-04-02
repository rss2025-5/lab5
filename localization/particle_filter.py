from localization.sensor_model import SensorModel
from localization.motion_model import MotionModel

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped, Quaternion, PoseArray, Pose, Point

from rclpy.node import Node
import rclpy

assert rclpy

# new imports
from sensor_msgs.msg import LaserScan
import tf2_ros
from tf_transformations import euler_from_quaternion, quaternion_from_euler
import numpy as np

class ParticleFilter(Node):

    def __init__(self):
        super().__init__("particle_filter")

        self.declare_parameter('particle_filter_frame', "default")
        self.particle_filter_frame = self.get_parameter('particle_filter_frame').get_parameter_value().string_value

        #  *Important Note #1:* It is critical for your particle
        #     filter to obtain the following topic names from the
        #     parameters for the autograder to work correctly. Note
        #     that while the Odometry message contains both a pose and
        #     a twist component, you will only be provided with the
        #     twist component, so you should rely only on that
        #     information, and *not* use the pose component.

        self.declare_parameter('odom_topic', "/odom")
        self.declare_parameter('scan_topic', "/scan")

        scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value

        self.laser_sub = self.create_subscription(LaserScan, scan_topic,
                                                  self.laser_callback,
                                                  1)

        self.odom_sub = self.create_subscription(Odometry, odom_topic,
                                                 self.odom_callback,
                                                 1)
        self.pose_list = self.create_publisher(PoseArray, "/all_poses", 10)
        self.predicted_pose = self.create_publisher(Pose, "/predicted_pose", 10)
        #  *Important Note #2:* You must respond to pose
        #     initialization requests sent to the /initialpose
        #     topic. You can test that this works properly using the
        #     "Pose Estimate" feature in RViz, which publishes to
        #     /initialpose.

        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, "/initialpose",
                                                 self.pose_callback,
                                                 1)

        #  *Important Note #3:* You must publish your pose estimate to
        #     the following topic. In particular, you must use the
        #     pose field of the Odometry message. You do not need to
        #     provide the twist part of the Odometry message. The
        #     odometry you publish here should be with respect to the
        #     "/map" frame.

        self.odom_pub = self.create_publisher(Odometry, "/pf/pose/odom", 1)

        # Initialize the models
        self.motion_model = MotionModel(self)
        self.sensor_model = SensorModel(self)

        # Implement the MCL algorithm
        # using the sensor model and the motion model
        #
        # Make sure you include some way to initialize
        # your particles, ideally with some sort
        # of interactive interface in rviz
        #
        # Publish a transformation frame between the map
        # and the particle_filter_frame.

        # Particle filter settings
        self.num_particles = 100  # TODO: USE PARAM VALUE
        self.particles = np.random.uniform(low=-5, high=5, size=(self.num_particles, 3))  # Initial random particles
        self.particle_weights = np.ones(self.num_particles) / self.num_particles  # Equal weights initially

        # TF broadcaster for particle filter pose
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.get_logger().info("=============+READY+=============")

    def laser_callback(self, msg):
        """
        Callback for laser scan messages.
        """
        # Use the sensor model to evaluate likelihood
        probabilities = self.sensor_model.evaluate(self.particles, np.array(msg.ranges))
        probabilities = probabilities/np.sum(probabilities)


        # Resample particles based on probabilities
        self.resample_particles(probabilities)

        # Compute the average particle pose
        self.publish_particle_pose()

    def odom_callback(self, msg):
        """
        Callback for odometry messages.
        """
        # Get odometry delta (dx, dy, dtheta)
        dx = msg.twist.twist.linear.x
        dy = msg.twist.twist.linear.y
        dtheta = msg.twist.twist.angular.z

        # Use the motion model to update the particle positions
        odometry = np.array([dx, dy, dtheta])
        self.particles = self.motion_model.evaluate(self.particles, odometry)

    def pose_callback(self, msg):
        """
        Callback for pose initialization messages.
        """
        # Set initial particles around the guessed pose with random spread
        initial_pose = msg.pose.pose
        x, y, theta = initial_pose.position.x, initial_pose.position.y, euler_from_quaternion([initial_pose.orientation.x, initial_pose.orientation.y, initial_pose.orientation.z, initial_pose.orientation.w])[2]

        # Initialize particles with random spread around the initial pose
        self.particles = np.random.normal(loc=[x, y, theta], scale=0.5, size=(self.num_particles, 3))  # Standard deviation can be adjusted
        self.particle_weights = np.ones(self.num_particles) / self.num_particles  # Reset particle weights

    def resample_particles(self, probabilities):
        """
        Resample particles based on their probabilities.
        """
        indices = np.random.choice(self.num_particles, size=self.num_particles, p=probabilities)
        self.particles = self.particles[indices]
        self.particle_weights = self.particle_weights[indices]  # Re-weight based on the resampled particles

    def compute_average_pose(self):
        """
        Compute the average pose of the particles, considering angular wraparound.
        """
        angles = self.particles[:, 2]
        sin_sum = np.sum(np.sin(angles))
        cos_sum = np.sum(np.cos(angles))
        avg_angle = np.arctan2(sin_sum, cos_sum)

        avg_x = np.mean(self.particles[:, 0])
        avg_y = np.mean(self.particles[:, 1])

        return avg_x, avg_y, avg_angle

    def publish_particle_pose(self):
        """
        Publish the average particle pose as an Odometry message and a TF transform.
        """
        avg_x, avg_y, avg_angle = self.compute_average_pose()

        # Publish to /pf/pose/odom
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = "map"
        odom_msg.child_frame_id = "base_link"

        odom_msg.pose.pose.position.x = avg_x
        odom_msg.pose.pose.position.y = avg_y

        quat_msg = Quaternion()
        quat_msg.x, quat_msg.y, quat_msg.z, quat_msg.w = quaternion_from_euler(0, 0, avg_angle)
        odom_msg.pose.pose.orientation = quat_msg

        self.odom_pub.publish(odom_msg)

        # Publish TF transform between map and base_link_pf
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = "map"
        transform.child_frame_id = "base_link_pf"

        transform.transform.translation.x = avg_x
        transform.transform.translation.y = avg_y
        transform.transform.translation.z = 0.0
        transform.transform.rotation = quat_msg

        self.tf_broadcaster.sendTransform(transform)

        self.get_logger().info(f"particles: {self.particles}")

        pose_array_msg = PoseArray()
        pose_array_msg.header.frame_id = "map"
        pose_array_msg.poses = []

        for pt in self.particles:
            quat2_msg = Quaternion()
            quat2_msg.x, quat2_msg.y, quat2_msg.z, quat2_msg.w = quaternion_from_euler(0, 0, pt[2])
            point_msg = Point()
            point_msg.x = pt[0]
            point_msg.y = pt[1]
            point_msg.z = 0.0

            pose_msg = Pose()
            pose_msg.position = point_msg
            pose_msg.orientation = quat2_msg

            pose_array_msg.poses.append(pose_msg)
        self.get_logger().info(f"pose_array = {pose_array_msg}")
        self.pose_list.publish(pose_array_msg)




def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFilter()
    rclpy.spin(pf)
    rclpy.shutdown()
