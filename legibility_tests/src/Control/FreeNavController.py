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
from interfaces.FreeNav import JackalInterfaceFreeNav
from interfaces.InterfacePlus import JackalInterfacePlus
from CostFn.FreeNav import ObjectiveFreeNav
from utils.config_store import *
#from utils.plotter import LiveBarChart
from geometry_msgs.msg import PoseStamped



class Planner_Free_Nav:
    def __init__(self):
        config = self.load_config()
        self.cfg = OmegaConf.to_object(config)
        self.interface = JackalInterfaceFreeNav(self.cfg)

        self.objective = ObjectiveFreeNav(self.cfg, None, self.interface)
        self.mppi_planner = MPPI_Wrapper(self.cfg, self.objective)

    def load_config(self):
        # Load Hydra configurations
        with initialize(config_path="../../conf"):
            config = compose(config_name="FreeNav")
        return config



    def run_jackal_robot(self):
       
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


                pose = PoseStamped()
                pose.header.frame_id = "map"
                pose.pose.position.x = self.interface.odom_msg.pose.pose.position.x
                pose.pose.position.y = self.interface.odom_msg.pose.pose.position.y
                pose.pose.position.z = self.interface.odom_msg.pose.pose.position.z
                self.interface.path.poses.append(pose)
                self.interface.path_publisher.publish(self.interface.path)


                # compute action
                start_time = time.time()
                action, plan, states = self.mppi_planner.compute_action(
                    q=robot_position,
                    qdot=robot_velocity,
                    obst=None
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
                    self.mppi_planner = MPPI_Wrapper(self.cfg, self.objective)



if __name__ == '__main__':
    try:
        planner = Planner_Free_Nav()
        planner.run_jackal_robot()

    except rospy.ROSInterruptException:
        pass