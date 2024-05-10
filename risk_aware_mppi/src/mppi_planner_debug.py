#!/usr/bin/python3

import sys

import numpy as np

from planner.mppi_isaac import MPPIisaacPlanner
import numpy as np
from scipy.interpolate import CubicSpline
from planner.mppi import MPPIPlanner
from utils.config_store import ExampleConfig
from utils.cubicspline import CubicSpline2D

import hydra
from hydra.experimental import initialize, compose
import yaml
from yaml import SafeLoader
import torch
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import time


class Objective(object):
    def __init__(self, cfg, device):
        self.nav_goal = torch.tensor(cfg.goal, device=cfg.mppi.device)
        self.v_ref = torch.tensor(cfg.v_ref, device=cfg.mppi.device)
        '''
        self.x_ref = np.array(
            [-57.302836, -56.76062061, -56.21867672, -55.67730956, -55.13680794, -54.59744398, -54.05947308,
             -53.52313388, -52.98864826, -52.45622131, -51.9260413, -51.39828503, -50.87332091, -50.35126661,
             -49.83199876, -49.31538464, -48.8012829, -48.2895437, -47.78000885, -47.2725117, -46.76687705,
             -46.26295748, -45.76092968, -45.26110064, -44.7635905, -44.26843317, -43.7756416, -43.28520804,
             -42.79710418, -42.31128127, -41.8276704, -41.34618006, -40.86663801, -40.38879226, -39.9123719,
             -39.43711161, -38.96276781, -38.48908289, -38.01578122, -37.54256919])

        self.y_ref = np.array([-6.1525149, -6.00065601, -5.84904207, -5.69774416, -5.54682757, -5.39635279, -5.24637558, -5.09694709, -4.94811389, -4.79991805, -4.65239721, -4.50556022, -4.3584472, -4.21109957, -4.06455945, -3.91983343, -3.7778897, -3.63965718, -3.50602524, -3.37784378, -3.25592395, -3.14091405, -3.03228981, -2.92899179, -2.83038065, -2.73599131, -2.64537239, -2.55808547, -2.47370462, -2.39181575, -2.31201585, -2.23392463, -2.15746878, -2.08287715, -2.01038416, -1.94009578, -1.8718999, -1.80566504, -1.74126266, -1.67856688])
        '''
        self.x_ref = np.array(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
             29, 30, 31, 32, 33, 34, 35, 36, 37, 38])
        self.y_ref = np.array(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0])
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
        if nNextRefPoint == 0:
            nPrevRefPoint = 0
            nNextRefPoint = 1

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
        nClosestRefPoint = 0

        for i in range(len(faRefX)):
            fRefX, fRefY = faRefX[i], faRefY[i]
            fDist = self.distance(fX, fY, fRefX, fRefY)

            if fDist < fClosestLen:
                fClosestLen = fDist
                nClosestRefPoint = i  # + 1  # MATLAB indexing starts from 1
            else:
                break

        if nClosestRefPoint == len(faRefX) - 1:
            nClosest2ndRefPoint = nClosestRefPoint - 1
        elif nClosestRefPoint == 0:
            nClosest2ndRefPoint = nClosestRefPoint + 1
        else:
            fRefXp1, fRefYp1 = faRefX[nClosestRefPoint + 1], faRefY[nClosestRefPoint + 1]
            fDistp1 = self.distance(fX, fY, fRefXp1, fRefYp1)

            fRefXm1, fRefYm1 = faRefX[nClosestRefPoint - 1], faRefY[nClosestRefPoint - 1]
            fDistm1 = self.distance(fX, fY, fRefXm1, fRefYm1)

            if fDistm1 < fDistp1:
                nClosest2ndRefPoint = nClosestRefPoint - 1
            else:
                nClosest2ndRefPoint = nClosestRefPoint + 1

        return nClosestRefPoint, nClosest2ndRefPoint

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

    def distance(self, fX1, fY1, fX2, fY2):
        return np.sqrt((fX1 - fX2) ** 2 + (fY1 - fY2) ** 2)

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

        return torch.tensor(d_values_traj, device="cuda:0") + velocity_cost
        '''

        pos = state[:, 0:2]
        velocity_cost = torch.square(u[:, 0] - self.v_ref)

        return torch.clamp(
            torch.linalg.norm(pos - self.nav_goal, axis=1) - 0.05, min=0, max=1999
        ) # + 0.5 * velocity_cost
        '''


class Planner:
    def __init__(self):
        config = self.load_config()
        self.cfg = OmegaConf.to_object(config)

    def load_config(self):
        # Load Hydra configurations
        with initialize(config_path="../conf"):
            config = compose(config_name="config_jackal_robot")
        return config

    def set_planner(self, cfg):
        """
        Initializes the mppi planner for jackal robot.

        Params
        ----------
        goal_position: np.ndarray
            The goal to the motion planning problem.
        """
        objective = Objective(cfg, cfg.mppi.device)
        mppi_planner = MPPIisaacPlanner(cfg, objective)

        return mppi_planner

    def odom_callback(self, msg):
        # store the received odometry message
        self.odom_msg = msg

    def run_jackal_robot(self):
        mppi_planner = self.set_planner(self.cfg)
        # robot_position = [-57.302836, -6.1525149, 0.27319292, 5.6313483, 0]
        robot_position = [0, 0, 0, 0]
        robot_velocity = [0, 0, 0]

        print(mppi_planner)
        for _ in range(self.cfg.n_steps):
            start_time = time.time()
            action, opt_plan, states = mppi_planner.compute_action(
                q=robot_position,
                qdot=robot_velocity,
            )
            end_time = time.time()
            print('the computation time is: ', end_time - start_time)
            print(action)
            all_trajs = states[:, :, 0:2].cpu().numpy()
            '''
            plt.figure(figsize=(8, 6))



            for traj in all_trajs:
                x_vals = traj[:, 0]
                y_vals = traj[:, 1]
                plt.plot(x_vals, y_vals, alpha=0.5)

            plt.plot(opt_plan[:, 0], opt_plan[:, 1], alpha=0.5, linewidth=20)



            plt.title('Trajectories')
            plt.xlabel('X-coordinate')
            plt.ylabel('Y-coordinate')
            plt.grid(True)
            plt.show()
            '''

            robot_position = [robot_position[0] + 0.05 * action[0].item() * np.cos(robot_position[2]),
                              robot_position[1] + 0.05 * action[0].item() * np.sin(robot_position[2]),
                              robot_position[2] + 0.05 * action[1].item(),
                              0]
            robot_velocity = [action[0].item(), action[1].item(), 0]


if __name__ == '__main__':
    mppi_planner = Planner()
    mppi_planner.run_jackal_robot()
