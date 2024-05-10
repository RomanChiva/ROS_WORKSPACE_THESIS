#!/usr/bin/python3

import sys
import os
# Get the parent directory of the current script
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# Add the parent directory to the Python path
sys.path.append(parent_dir)



import rospy
from tf.transformations import euler_from_quaternion
import numpy as np
import time
from planner.MPPI_wrapper import MPPI_Wrapper
from hydra.experimental import initialize, compose
import torch
from omegaconf import OmegaConf
from obstacles.obstacle_class import DynamicObstacles
from interfaces.ObstacleAvoidance import JackalInterface
from CostFn.ObstacleAvoidance import Objective
from utils.plotter import LiveBarChart
from utils.config_store import *

class DynamicObstacle(object):
    def __init__(self, id, disc_start_id, radius):
        self.obstacle_id = id
        self.disc_start_id = disc_start_id
        self.radius = radius
        self.pose = None



class Planner:
    def __init__(self):
        config = self.load_config()
        self.cfg = OmegaConf.to_object(config)
        self.interface = JackalInterface(self.cfg)

    def load_config(self):
        # Load Hydra configurations
        with initialize(config_path="../../conf"):
            config = compose(config_name="ObstacleAvoidance")
        return config

    def set_planner(self, obstacles):
        """
        Initializes the mppi planner for jackal robot.

        Params
        ----------
        goal_position: np.ndarray
            The goal to the motion planning problem.
        """
        objective = Objective(self.cfg, obstacles)
        mppi_planner = MPPI_Wrapper(self.cfg, objective)

        return mppi_planner

    def init_obstacles(self):

        # Set velocities of the obstacles. Not very nice but it works for the example
        N_obstacles = len(self.interface.obstacle_predictions.obstacles)
        print("number of obstacles is: ", N_obstacles)

        # Create covariance matrix which consists of N_obstacles stacks of 2x2 covariance matrices
        cov = torch.eye(2, device=self.cfg.mppi.device) * self.cfg.obstacles.initial_covariance
        cov = cov.repeat(N_obstacles, 1, 1)

        obstacles = DynamicObstacles(self.cfg, cov, N_obstacles)

        predicted_coordinates = torch.zeros(
            (self.cfg.mppi.horizon, len(self.interface.obstacle_predictions.obstacles), 2),
            device=self.cfg.mppi.device)
        predicted_cov = torch.zeros((self.cfg.mppi.horizon, len(self.interface.obstacle_predictions.obstacles), 2, 2),
                                    device=self.cfg.mppi.device)

        if self.interface.obstacle_predictions is not None:
            # iterate over the obstacles in the workspace
            obst_num = 0
            for obstacle in self.interface.obstacle_predictions.obstacles:
                for mode in range(len(obstacle.gaussians)):  # iterate over the prediction modes
                    for index in range(len(obstacle.gaussians[mode].mean.poses)):
                        predicted_coordinates[index, obst_num, 0] = obstacle.gaussians[mode].mean.poses[
                            index].pose.position.x
                        predicted_coordinates[index, obst_num, 1] = obstacle.gaussians[mode].mean.poses[
                            index].pose.position.y
                        predicted_cov[index, obst_num, :, :] = torch.tensor(
                            [[obstacle.gaussians[mode].major_semiaxis[0], 0],
                             [0, obstacle.gaussians[mode].major_semiaxis[0]]], dtype=torch.float32)
                obst_num += 1

        obstacles.predicted_coordinates = predicted_coordinates
        obstacles.predicted_covs = predicted_cov
        return obstacles

    def run_jackal_robot(self):
        # # Create a publisher for velocity commands
        while self.interface.obstacle_predictions is None:
            # The code will keep looping until some_variable is not None
            pass

        # Create Obstacles Object
        obstacles = self.init_obstacles()
        # Initialize MPPI Planner
        mppi_planner = self.set_planner(obstacles)
        # MISC Variables
        step_num = 0
        computational_time = []

        # INITIALIZE LOOP
        while not rospy.is_shutdown():
            # visualize goal and reference path
            self.interface.visualize_spline()
            self.interface.marker_publisher.publish(self.interface.marker)

            
            if self.interface.odom_msg is not None:
                # Get Robot Pose
                orientation_quaternion = self.interface.odom_msg.pose.pose.orientation
                orientation_euler = euler_from_quaternion([
                    orientation_quaternion.x,
                    orientation_quaternion.y,
                    orientation_quaternion.z,
                    orientation_quaternion.w
                ])
                # Extract individual Euler angles
                roll, pitch, yaw = orientation_euler
                robot_position = [self.interface.odom_msg.pose.pose.position.x,
                                  self.interface.odom_msg.pose.pose.position.y,
                                  yaw, 0]
                robot_velocity = [self.interface.odom_msg.twist.twist.linear.x,
                                  self.interface.odom_msg.twist.twist.linear.y,
                                  self.interface.odom_msg.twist.twist.angular.z]

                # compute action
                start_time = time.time()
                action, plan, states = mppi_planner.compute_action(
                    q=robot_position,
                    qdot=robot_velocity,
                    obst=self.interface.obstacle_predictions
                )
                end_time = time.time()
                if step_num > 0:
                    computational_time.append(end_time - start_time)
                step_num += 1

                # visualize trajectory
                self.interface.visualize_trajectory(plan)

                # actuate robot and sleep
                self.interface.actuate(action)

                # check if the target goal is reached
                if np.linalg.norm(np.array(robot_position[0:2]) - np.array(self.cfg.goal)) < 0.7:
                    self.interface.reset_env()
                    print("planning computational time is: ", (np.sum(computational_time) / step_num) * 1000, " ms")
                    computational_time = []
                    step_num = 0
                    mppi_planner = self.set_planner(obstacles)



if __name__ == '__main__':
    try:
        planner = Planner()
        planner.run_jackal_robot()

    except rospy.ROSInterruptException:
        pass