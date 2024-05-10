#!/usr/bin/python3

import rospy
from geometry_msgs.msg import Twist, Point, Quaternion, PoseStamped
from sensor_msgs.msg import Joy
from nav_msgs.msg import Odometry, Path
from tf.transformations import euler_from_quaternion
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from derived_object_msgs.msg import Object, ObjectArray
from lmpcc_msgs.msg import obstacle_array
from std_srvs.srv import Empty as EmptySrv
from std_msgs.msg import Empty as EmptyMsg
from robot_localization.srv import SetPose
import numpy as np
from scipy.interpolate import CubicSpline
import time
import sys
from planner.mppi_isaac import MPPIisaacPlanner
from utils.cubicspline import CubicSpline2D
from planner.mppi import MPPIPlanner
from utils.config_store import ExampleConfig
import hydra
from hydra.experimental import initialize, compose
import yaml
from yaml import SafeLoader
import torch
from torch.autograd import Variable
from scipy.optimize import minimize
from omegaconf import OmegaConf
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray


class DynamicObstacle(object):
    def __init__(self, id, disc_start_id, radius):
        self.obstacle_id = id
        self.disc_start_id = disc_start_id
        self.radius = radius
        self.pose = None


class Objective(object):
    def __init__(self, cfg, device):
        self.nav_goal = torch.tensor(cfg.goal, device=cfg.mppi.device)
        self.v_ref = torch.tensor(cfg.v_ref, device=cfg.mppi.device)
        self.x_ref = np.array([0, 3, 6, 9, 12, 15, 18, 21, 30, 34, 38])
        self.y_ref = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.reference_spline = CubicSpline2D(self.x_ref, self.y_ref)

    def generate_reference_spline(self, x_ref, y_ref):
        # Resample the spline_x and spline_y
        t = np.linspace(0, 1, len(x_ref))

        # Fit splines
        self.spline_x = CubicSpline(t, x_ref)
        self.spline_y = CubicSpline(t, y_ref)

    def cart2frt(self, fX, fY, fPsi, faRefX, faRefY):
        nClosestRefPoint, nClosest2ndRefPoint = self.closest_ref_point(fX, fY, faRefX, faRefY)

        if nClosestRefPoint > nClosest2ndRefPoint:
            nNextRefPoint = nClosestRefPoint
        else:
            nNextRefPoint = nClosest2ndRefPoint

        nPrevRefPoint = nNextRefPoint - 1
        if nNextRefPoint == 1:
            nPrevRefPoint = 1
            nNextRefPoint = 2

        fTangentX = faRefX[nNextRefPoint] - faRefX[nPrevRefPoint]
        fTangentY = faRefY[nNextRefPoint] - faRefY[nPrevRefPoint]

        fVecX = fX - faRefX[nPrevRefPoint]
        fVecY = fY - faRefY[nPrevRefPoint]

        fTangentLength = np.linalg.norm([fTangentX, fTangentY])
        fProjectedVecNorm = np.dot([fVecX, fVecY], [fTangentX, fTangentY]) / fTangentLength
        fProjectedVecX = fProjectedVecNorm * fTangentX / fTangentLength
        fProjectedVecY = fProjectedVecNorm * fTangentY / fTangentLength

        fD = self.distance(fVecX, fVecY, fProjectedVecX, fProjectedVecY)

        fX1, fY1 = faRefX[nPrevRefPoint], faRefY[nPrevRefPoint]
        fX2, fY2 = faRefX[nNextRefPoint], faRefY[nNextRefPoint]
        fd = (fX - fX1) * (fY2 - fY1) - (fY - fY1) * (fX2 - fX1)
        nSide = np.sign(fd)
        if nSide > 0:
            fD *= -1

        fS = 0
        for i in range(nPrevRefPoint):
            fS += self.distance(faRefX[i], faRefY[i], faRefX[i + 1], faRefY[i + 1])

        fS += fProjectedVecNorm

        return fS, fD

    def closest_ref_point(self, fX, fY, faRefX, faRefY):
        fClosestLen = np.inf
        nClosestRefPoint = 1

        for i in range(len(faRefX)):
            fRefX, fRefY = faRefX[i], faRefY[i]
            fDist = self.distance(fX, fY, fRefX, fRefY)

            if fDist < fClosestLen:
                fClosestLen = fDist
                nClosestRefPoint = i + 1
            else:
                break

        if nClosestRefPoint == len(faRefX):
            nClosest2ndRefPoint = nClosestRefPoint - 1
        elif nClosestRefPoint == 1:
            nClosest2ndRefPoint = nClosestRefPoint + 1
        else:
            fRefXp1, fRefYp1 = faRefX[nClosestRefPoint], faRefY[nClosestRefPoint]
            fDistp1 = self.distance(fX, fY, fRefXp1, fRefYp1)

            fRefXm1, fRefYm1 = faRefX[nClosestRefPoint - 1], faRefY[nClosestRefPoint - 1]
            fDistm1 = self.distance(fX, fY, fRefXm1, fRefYm1)

            if fDistm1 < fDistp1:
                nClosest2ndRefPoint = nClosestRefPoint - 1
            else:
                nClosest2ndRefPoint = nClosestRefPoint + 1

        return nClosestRefPoint, nClosest2ndRefPoint

    def distance(self, fX1, fY1, fX2, fY2):
        return np.sqrt((fX1 - fX2) ** 2 + (fY1 - fY2) ** 2)

    def frenet_s(self, external_points, reference_path):
        _, _, closest_point_indices, second_closest_point_indices = self.find_closest_points_vectorized(external_points,
                                                                                                        reference_path)
        # Adjust indices to get nPrevRefPoint and nNextRefPoint
        nPrevRefPoint = closest_point_indices
        nNextRefPoint = second_closest_point_indices

        # Handle boundary cases where nNextRefPoint exceeds the array size
        nNextRefPoint[nNextRefPoint == len(reference_path)] = 0

        # Swap indices if nClosest2ndRefPoint is greater than nClosestRefPoint
        swap_indices = nPrevRefPoint > nNextRefPoint
        nPrevRefPoint[swap_indices], nNextRefPoint[swap_indices] = nNextRefPoint[swap_indices], nPrevRefPoint[
            swap_indices]

        # Calculate tangent vectors
        fTangentX = reference_path[nNextRefPoint, 0] - reference_path[nPrevRefPoint, 0]
        fTangentY = reference_path[nNextRefPoint, 1] - reference_path[nPrevRefPoint, 1]

        # Calculate vectors to external points
        fVecX = external_points[:, 0] - reference_path[nPrevRefPoint, 0]
        fVecY = external_points[:, 1] - reference_path[nPrevRefPoint, 1]

        # Calculate lengths and dot products
        fTangentLength = np.linalg.norm([fTangentX, fTangentY], axis=0)
        fProjectedVecNorm = np.sum(np.stack([fVecX, fVecY], axis=1) * np.stack([fTangentX, fTangentY], axis=1),
                                   axis=1) / fTangentLength

        reference_path_x = reference_path[:, 0]
        reference_path_y = reference_path[:, 1]

        # Calculate distances between consecutive points
        distances = np.sqrt(np.diff(reference_path_x) ** 2 + np.diff(reference_path_y) ** 2)

        # Initialize an empty array to store the cumulative distances
        fS = np.zeros_like(nPrevRefPoint, dtype=float)

        # Loop through each entry in nPrevRefPoint and calculate cumulative distances
        for i, index in enumerate(nPrevRefPoint):
            fS[i] = np.sum(distances[:index])

        # Add the projected vectors to get the final S values
        fS += fProjectedVecNorm

        return fS

    def find_closest_points_vectorized(self, external_points, reference_path):
        # Calculate the Euclidean distance between each external point and all points in the reference path
        distances = np.linalg.norm(reference_path[:, np.newaxis, :] - external_points, axis=2)
        # Find the indices of the points with the minimum distances
        closest_point_indices = np.argmin(distances, axis=0)
        # Retrieve the closest points
        closest_points = reference_path[closest_point_indices]

        # Find the second-closest points
        n_closest_ref_point = closest_point_indices
        n_closest_2nd_ref_point = np.zeros_like(external_points[:, 0], dtype=np.int)

        # Cases where nClosestRefPoint is at the beginning or end
        mask_end = n_closest_ref_point == len(reference_path) - 1
        mask_start = n_closest_ref_point == 0

        n_closest_2nd_ref_point[mask_end] = n_closest_ref_point[mask_end] - 1
        n_closest_2nd_ref_point[mask_start] = n_closest_ref_point[mask_start] + 1

        # Cases where nClosestRefPoint is neither at the beginning nor end
        mask_mid = ~mask_end & ~mask_start

        fRefXp1 = reference_path[n_closest_ref_point[:] + 1, 0]
        fRefYp1 = reference_path[n_closest_ref_point[:] + 1, 1]
        fDistp1 = np.linalg.norm(
            external_points[:] - np.stack([fRefXp1, fRefYp1], axis=1), axis=1
        )

        fRefXm1 = reference_path[n_closest_ref_point[:] - 1, 0]
        fRefYm1 = reference_path[n_closest_ref_point[:] - 1, 1]
        fDistm1 = np.linalg.norm(
            external_points[:] - np.stack([fRefXm1, fRefYm1], axis=1), axis=1
        )

        # Update mask_choose_p1 based on distances
        mask_choose_p1 = fDistm1 < fDistp1
        n_closest_2nd_ref_point[mask_mid] = n_closest_ref_point[mask_mid]

        # Subtract 1 where mask_choose_p1 is True in the mask_mid region
        n_closest_2nd_ref_point[:][mask_mid & mask_choose_p1] -= 1
        # Add 1 where mask_choose_p1 is False in the mask_mid region
        n_closest_2nd_ref_point[:][mask_mid & ~mask_choose_p1] += 1

        # Retrieve the second-closest points
        second_closest_points = reference_path[n_closest_2nd_ref_point]
        return closest_points, second_closest_points, closest_point_indices, n_closest_2nd_ref_point

    '''
    def compute_cost(self, state, u):
        pos = state[:, 0:2]
        velocity_cost = torch.square(u[:, 0] - self.v_ref)

        pos_x = state[:, 0]
        pos_y = state[:, 1]

        pos_x_cpu = pos_x.cpu().numpy()
        pos_y_cpu = pos_y.cpu().numpy()

        external_points = np.array(list(zip(pos_x_cpu, pos_y_cpu)))
        reference_path = np.array(list(zip(self.x_ref, self.x_ref)))

        # closest_point, closest_point_index, _, _ = find_closest_points_vectorized(external_points, reference_path)
        fS = self.frenet_s(external_points, reference_path)
        d_values_traj = np.abs(
            [np.linalg.norm([x - self.reference_spline.sx(s), y - self.reference_spline.sy(s)]) for s, x, y
             in
             zip(fS, pos_x_cpu, pos_y_cpu)])

        return torch.tensor(d_values_traj, device="cuda:0")  # + 0.5 * velocity_cost

    '''

    def compute_cost(self, state, u):
        pos = state[:, 0:2]
        velocity_cost = torch.square(u[:, 0] - self.v_ref)

        return torch.clamp(
            torch.linalg.norm(pos - self.nav_goal, axis=1) - 0.05, min=0, max=1999
        )  # + 0.5 * velocity_cost


class Planner:
    def __init__(self):
        rospy.init_node('jackal_controller', anonymous=True)

        config = self.load_config()
        self.cfg = OmegaConf.to_object(config)

        x_ref = np.array([0, 3, 6, 9, 12, 15, 18, 21, 30, 34, 38])
        y_ref = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        self.spline_x, self.spline_y, self.x_fine, self.y_fine = self.generate_reference_spline(x_ref, y_ref)
        self.dynamic_obstacles = []
        self.static_obstacles = []
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
        self.marker.pose.position = Point(self.cfg.goal[0], self.cfg.goal[1], 0.0)

        self.visual_plan_publisher = rospy.Publisher('/collision_space', MarkerArray, queue_size=10)
        self.trajectory_publisher = rospy.Publisher('/trajectory_path', MarkerArray, queue_size=10)
        self.spline_marker_publisher = rospy.Publisher('/spline_marker', Marker, queue_size=10)

        self.obstacle_marker_publisher = rospy.Publisher('/received_obstacles', MarkerArray, queue_size=10)
        self.obstacle_markers = MarkerArray()

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

        self.velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        rospy.Subscriber('/robot_ekf/odometry', Odometry, self.odom_callback)
        rospy.Subscriber('/dynamic_object3', PoseStamped, self.static_obstacle_callback)
        self.vel_cmd = Twist()
        self.odom_msg = None

        # subscribe to robot's states
        '''
        rospy.Subscriber('/odometry/filtered', Odometry, self.odom_callback)
        '''
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

    def load_config(self):
        # Load Hydra configurations
        with initialize(config_path="../conf"):
            config = compose(config_name="config_jackal_robot")
        return config

    def set_planner(self):
        """
        Initializes the mppi planner for jackal robot.

        Params
        ----------
        goal_position: np.ndarray
            The goal to the motion planning problem.
        """
        objective = Objective(self.cfg, self.cfg.mppi.device)
        mppi_planner = MPPIisaacPlanner(self.cfg, objective)

        return mppi_planner

    def odom_callback(self, msg):
        # store the received odometry message
        self.odom_msg = msg

    def static_obstacle_callback(self, msg):
        self.static_obstacles.clear()
        self.static_obst = msg

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

    def run_jackal_robot(self):
        # Create a publisher for velocity commands
        mppi_planner = self.set_planner()
        step_num = 0
        computational_time = []

        while not rospy.is_shutdown():

            self.marker_publisher.publish(self.marker)
            self.visualize_spline()

            if self.odom_msg is not None:
                orientation_quaternion = self.odom_msg.pose.pose.orientation
                orientation_euler = euler_from_quaternion([
                    orientation_quaternion.x,
                    orientation_quaternion.y,
                    orientation_quaternion.z,
                    orientation_quaternion.w
                ])
                # Extract individual Euler angles
                roll, pitch, yaw = orientation_euler
                robot_position = [self.odom_msg.pose.pose.position.x, self.odom_msg.pose.pose.position.y,
                                  yaw, 0]
                robot_velocity = [self.odom_msg.twist.twist.linear.x, self.odom_msg.twist.twist.linear.y,
                                  self.odom_msg.twist.twist.angular.z]

                # print("The jackal position is: ", robot_position[0:2])
                start_time = time.time()
                action, plan, states = mppi_planner.compute_action(
                    q=robot_position,
                    qdot=robot_velocity,
                    obst=self.static_obst
                )
                end_time = time.time()
                if step_num > 0:
                    computational_time.append(end_time - start_time)
                step_num += 1
                # print(action)
                self.visualize_trajectory(plan)

                self.vel_cmd.linear.x = action[0].item()
                self.vel_cmd.angular.z = action[1].item()

                # Publish the velocity command
                self.velocity_publisher.publish(self.vel_cmd)

                # Sleep to control the frequency of the command
                self.rate.sleep()

                # check if the target goal is reached
                '''
                if np.linalg.norm(np.array(robot_position[0:2]) - np.array(self.cfg.goal)) < 0.7:
                    # Call the service to reset the simulation
                    self.reset_simulation_client()
                    reset_msg = EmptyMsg()

                    # Publish the Empty message to the specified topic
                    self.reset_simulation_pub.publish(reset_msg)
                    # print("the robot x-position is: ", self.odom_msg.pose.pose.position.x, " and the y-position is: ",
                    #      self.odom_msg.pose.pose.position.y)
                    print("planning computational time is: ", (np.sum(computational_time) / step_num) * 1000, " ms")
                    computational_time = []
                    step_num = 0
                    mppi_planner = self.set_planner()
                '''


if __name__ == '__main__':
    try:
        planner = Planner()
        planner.run_jackal_robot()
    except rospy.ROSInterruptException:
        pass
