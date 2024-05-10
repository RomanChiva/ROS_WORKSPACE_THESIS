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


class JackalInterfaceFreeNav:
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


        # service clients for resetting
        self.reset_simulation_client = rospy.ServiceProxy('/gazebo/reset_world', EmptySrv)
        self.reset_ekf_client = rospy.ServiceProxy('/set_pose', SetPose)
        #self.reset_simulation_pub = rospy.Publisher('/lmpcc/reset_environment', EmptyMsg, queue_size=1)


        # Visualize Robot Path
        self.path_publisher = rospy.Publisher('robot_trajectory', Path, queue_size=10)
        self.path = Path()
        self.path.header.frame_id = "map"
        self.path.header.stamp = rospy.Time.now()


        # Set the rate at which to publish commands (in Hz)
        self.frequency = 10
        self.rate = rospy.Rate(self.frequency)  # 10 Hz

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

        line_strip_marker.color = ColorRGBA(0.0, 1.0, 0.0, 1.0)  # Blue color

        # Set the points for the line strip
        line_strip_marker.points = [Point(x=point[0].item(), y=point[1].item(), z=0.0) for point in traj]

        markers.append(line_strip_marker)
        marker_array = MarkerArray(markers=markers)
        self.visual_plan_publisher.publish(marker_array)


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
        visual_plan_marker.color.a = 0.5  # Semi-transparent
        visual_plan_marker.color.r = 0.0
        visual_plan_marker.color.g = 0.0
        visual_plan_marker.color.b = 1.0  # Blue
        visual_plan_marker.pose.position = Point(x, y, 0.0)  # Set the z-coordinate to 0
        visual_plan_marker.pose.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)

        return visual_plan_marker


    ######################################### CALLBACK FUNCTIONS #########################################

    def odom_callback(self, msg):
        # store the received odometery message
        self.odom_msg = msg


    def reset_env(self):
        # Call the service to reset the simulation
        self.reset_simulation_client()
        reset_msg = EmptyMsg()

        # Publish the Empty message to the specified topic
        #self.reset_simulation_pub.publish(reset_msg)

    def actuate(self, action):
        self.vel_cmd.linear.x  = action[0].item()
        self.vel_cmd.angular.z = action[1].item()

        # Publish the velocity command
        self.velocity_publisher.publish(self.vel_cmd)
        self.rate.sleep()

    def create_goals(self, goal, separation, goal_index):

        separation_vector = np.array([0, separation])
        goal_1 = goal + separation_vector
        goal_2 = goal - separation_vector

        # Create a tensor with the two goals
        goals = torch.tensor([goal_1, goal_2], device=self.cfg.mppi.device)

        return goals, goals[goal_index]

