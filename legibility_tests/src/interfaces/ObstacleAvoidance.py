#!/usr/bin/python3

import rospy
from geometry_msgs.msg import Twist, Point, Quaternion
from nav_msgs.msg import Odometry, Path
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, Float64
from derived_object_msgs.msg import Object, ObjectArray
from lmpcc_msgs.msg import obstacle_array
from std_srvs.srv import Empty as EmptySrv
from std_msgs.msg import Empty as EmptyMsg
from robot_localization.srv import SetPose
import numpy as np
from scipy.interpolate import CubicSpline
import torch
import copy

class DynamicObstacle(object):
    def __init__(self, id, disc_start_id, radius):
        self.obstacle_id = id
        self.disc_start_id = disc_start_id
        self.radius = radius
        self.pose = None


class JackalInterface:
    def __init__(self, cfg):
        rospy.init_node('jackal_controller', anonymous=True)

        x_ref = np.array([0, 3, 6, 9, 12, 15, 18, 21, 30, 34, 38])
        y_ref = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        self.spline_x, self.spline_y, self.x_fine, self.y_fine = self.generate_reference_spline(x_ref, y_ref)

        # Visualize robot's goal
        self.marker_publisher = rospy.Publisher('visualization_marker', Marker, queue_size=10)
        self.marker = Marker()
        self.marker.header.frame_id = "map"
        self.marker.header.stamp = rospy.Time.now()
        self.marker.type = Marker.ARROW
        self.marker.action = Marker.ADD
        self.marker.scale.x = 0.2  # Diameter of the circle
        self.marker.scale.y = 0.2
        self.marker.scale.z = 0.2
        self.marker.color.a = 1.0  # Fully opaque
        self.marker.color.r = 1.0  # Red
        self.marker.pose.position = Point(cfg.goal[0], cfg.goal[1], 0.0)

        # robot's plan visuals
        self.visual_plan_publisher = rospy.Publisher('/collision_space', MarkerArray, queue_size=10)
        # obstacles visuals
        self.obstacle_marker_publisher = rospy.Publisher('/received_obstacles', MarkerArray, queue_size=10)
        self.obstacle_markers = MarkerArray()
        self.dynamic_obstacles = []

        # reference path visualization
        self.spline_marker_publisher = rospy.Publisher('/spline_marker', Marker, queue_size=10)
        self.spline_marker = Marker()
        self.spline_marker.header.frame_id = "map"  # Assuming "map" is your frame_id
        self.spline_marker.header.stamp = rospy.Time.now()
        self.spline_marker.type = Marker.LINE_STRIP
        self.spline_marker.action = Marker.ADD
        self.spline_marker.scale.x = 0.05  # Adjust the line width as needed
        self.spline_marker.color.a = 1.0  # Fully opaque
        self.spline_marker.color.r = 0.0  # Red
        self.spline_marker.color.g = 1.0  # Green
        self.spline_marker.color.b = 0.0  # Blue

        # velocity commands publisher
        self.velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.vel_cmd = Twist()
        self.odom_msg = None

        # subscribe to robot's states
        rospy.Subscriber('/odometry/filtered', Odometry, self.odom_callback)
        self.odom_msg = None

        # subscribe to pedestrians
        rospy.Subscriber('/pedestrian_simulator/pedestrians', ObjectArray, self.obstacle_callback)

        # subscribe to pedestrians predictions
        rospy.Subscriber('/pedestrian_simulator/trajectory_predictions', obstacle_array,
                         self.obstacle_predictions_callback)
        self.obstacle_predictions_msg = None
        self.obstacle_predictions = None

        # service clients for resetting
        self.reset_simulation_client = rospy.ServiceProxy('/gazebo/reset_world', EmptySrv)
        self.reset_ekf_client = rospy.ServiceProxy('/set_pose', SetPose)
        self.reset_simulation_pub = rospy.Publisher('/lmpcc/reset_environment', EmptyMsg, queue_size=1)

        # Set the rate at which to publish commands (in Hz)
        self.rate = rospy.Rate(10)  # 10 Hz

    def generate_reference_spline(self, x_ref, y_ref):
        # Resample the spline_x and spline_y
        t = np.linspace(0, 1, len(x_ref))

        # Fit splines
        spline_x = CubicSpline(t, x_ref)
        spline_y = CubicSpline(t, y_ref)

        # Evaluate the splines at a finer resolution
        t_fine = np.linspace(0, 1, 1000)
        x_fine = spline_x(t_fine)
        y_fine = spline_y(t_fine)
        return spline_x, spline_y, x_fine, y_fine

    def visualize_spline(self):
        for i in range(len(self.x_fine)):
            point = Point()
            point.x = self.x_fine[i]
            point.y = self.y_fine[i]
            point.z = 0.0  # Assuming 2D path, adjust for 3D if needed
            self.spline_marker.points.append(point)

        self.spline_marker.header.stamp = rospy.Time.now()
        self.spline_marker_publisher.publish(self.spline_marker)

    def visualize_trajectory(self, traj):
        markers = []

        for i, point in enumerate(traj):
            x, y = point[0].item(), point[1].item()  # Extract (x, y) from the tensor
            marker = self.create_circle_marker(x, y, i)
            markers.append(marker)

        # Create a line strip marker
        line_strip_marker = Marker()
        line_strip_marker.header.frame_id = "map"
        line_strip_marker.header.stamp = rospy.Time.now()
        line_strip_marker.id = len(traj)  # Unique ID for the line strip
        line_strip_marker.type = Marker.LINE_STRIP
        line_strip_marker.action = Marker.ADD
        line_strip_marker.scale.x = 0.1  # Adjust as needed
        line_strip_marker.scale.z = 0.1e-3  # Adjust as needed

        line_strip_marker.color = ColorRGBA(0.0, 0.0, 1.0, 1.0)  # Blue color

        # Set the points for the line strip
        line_strip_marker.points = [Point(x=point[0].item(), y=point[1].item(), z=0.0) for point in traj]

        markers.append(line_strip_marker)
        marker_array = MarkerArray(markers=markers)
        self.visual_plan_publisher.publish(marker_array)

    def plot_obstacles(self):
        obstacle_radius = 0.3
        plot_height = 1.5

        for obstacle in self.dynamic_obstacles:
            obstacle_marker = self.create_obstacle_marker(obstacle)
            obstacle_marker.id = obstacle.obstacle_id
            obstacle_marker.scale.x = 2.0 * obstacle_radius
            obstacle_marker.scale.y = 2.0 * obstacle_radius
            obstacle_marker.pose.position.z = 1e-3 + plot_height / 2.0

            # Append the obstacle marker to the MarkerArray
            self.obstacle_markers.markers.append(obstacle_marker)

        # Publish the entire MarkerArray
        self.obstacle_marker_publisher.publish(self.obstacle_markers)

    def create_circle_marker(self, x, y, marker_id):
        radius = 0.3
        visual_plan_marker = Marker()
        visual_plan_marker.header.frame_id = "map"
        visual_plan_marker.header.stamp = rospy.Time.now()
        visual_plan_marker.id = marker_id  # Unique ID for each marker
        visual_plan_marker.type = Marker.CYLINDER
        visual_plan_marker.action = Marker.ADD
        visual_plan_marker.scale.x = radius * 2  # Diameter of the circle
        visual_plan_marker.scale.y = radius * 2
        visual_plan_marker.scale.z = 0.1e-3
        visual_plan_marker.color.a = 0.3  # Semi-transparent
        visual_plan_marker.color.r = 0.0
        visual_plan_marker.color.g = 0.0
        visual_plan_marker.color.b = 1.0  # Blue
        visual_plan_marker.pose.position = Point(x, y, 0.0)  # Set the z-coordinate to 0
        visual_plan_marker.pose.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)

        return visual_plan_marker

    def create_obstacle_marker(self, obstacle):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.color = ColorRGBA(0.0, 0.0, 0.0, 0.8)
        marker.scale.x = 0.6  # Diameter of the cylinder
        marker.scale.y = 0.6
        marker.scale.z = 1.5  # Height of the cylinder
        marker.pose.orientation.w = 1.0  # No rotation
        marker.pose = obstacle.pose  # Assuming obstacle.pose is of type Pose
        return marker

    ######################################### CALLBACK FUNCTIONS #########################################

    def odom_callback(self, msg):
        # store the received odometery message
        self.odom_msg = msg

    def obstacle_callback(self, obstacle_msg):
        self.dynamic_obstacles.clear()
        disc_id = 0
        obstacle_radius = 0.3
        for obstacle in obstacle_msg.objects:
            dynamic_obstacle = DynamicObstacle(obstacle.id, disc_id, obstacle_radius)
            dynamic_obstacle.pose = obstacle.pose
            self.dynamic_obstacles.append(dynamic_obstacle)
            disc_id += 1

        self.plot_obstacles()

    def obstacle_predictions_callback(self, obstacle_predictions_msg):
        self.obstacle_predictions_msg = obstacle_predictions_msg
        # print("Obstacle prediction callback is called")
        # print(len(obstacle_predictions_msg.obstacles), " obstacle predictions received")
        if len(obstacle_predictions_msg.obstacles) == 0 or obstacle_predictions_msg.obstacles[0].id == -1:
            rospy.logwarn('Predictions received when obstacle positions are unknown, not loading.')
            return

        # load predictions into the obstacles
        self.obstacle_predictions = obstacle_predictions_msg

    def reset_env(self):
        # Call the service to reset the simulation
        self.reset_simulation_client()
        reset_msg = EmptyMsg()

        # Publish the Empty message to the specified topic
        self.reset_simulation_pub.publish(reset_msg)

    def actuate(self, action):
        self.vel_cmd.linear.x  = action[0].item()
        self.vel_cmd.angular.z = action[1].item()

        # Publish the velocity command
        self.velocity_publisher.publish(self.vel_cmd)
        self.rate.sleep()
