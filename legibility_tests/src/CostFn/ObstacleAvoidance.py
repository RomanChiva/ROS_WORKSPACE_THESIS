#!/usr/bin/python3

import numpy as np
from scipy.interpolate import CubicSpline
from utils.cubicspline import CubicSpline2D
import torch
from interfaces.ObstacleAvoidance import JackalInterface
from sklearn.mixture import GaussianMixture


class Objective(object):
    def __init__(self, cfg, obstacles):
        self.nav_goal = torch.tensor(cfg.goal, device=cfg.mppi.device)
        self.v_ref = torch.tensor(cfg.v_ref, device=cfg.mppi.device)
        self.x_ref = np.array([0, 3, 6, 9, 12, 15, 18, 21, 30, 34, 38])
        self.y_ref = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.reference_spline = CubicSpline2D(self.x_ref, self.y_ref)
        self.obstacles = obstacles
        self.cfg = cfg
        self.interface = JackalInterface(cfg)

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

    ## THIS IS THE ONE WE USE!!!!!!!!!!!!!!!!
    def compute_cost(self, state, u, t, obst):
        velocity_cost = torch.square(u[:, 0] - self.v_ref)
        if self.cfg.mppi.calculate_cost_once:
            pos = state[:, :, 0:2]
            state_goal = state.permute(1, 0, 2)
            # Now reshape to (T*K, nx)
            state_goal = state_goal.reshape(-1, self.cfg.nx)
            pos_goal = state_goal[:, 0:2]

            goal_cost = torch.linalg.norm(pos_goal - self.nav_goal, axis=1)

            predicted_coordinates = torch.zeros((self.cfg.mppi.horizon, len(obst.obstacles), 2),
                                                device=self.cfg.mppi.device)
            predicted_cov = torch.zeros((self.cfg.mppi.horizon, len(obst.obstacles), 2, 2), device=self.cfg.mppi.device)

            if obst is not None:
                # iterate over the obstacles in the workspace
                obst_num = 0
                for obstacles in obst.obstacles:
                    for mode in range(len(obstacles.gaussians)):  # iterate over the prediction modes
                        for index in range(len(obstacles.gaussians[mode].mean.poses)):
                            predicted_coordinates[index, obst_num, 0] = obstacles.gaussians[mode].mean.poses[
                                index].pose.position.x
                            predicted_coordinates[index, obst_num, 1] = obstacles.gaussians[mode].mean.poses[
                                index].pose.position.y
                            predicted_cov[index, obst_num, :, :] = torch.tensor(
                                [[obstacles.gaussians[mode].major_semiaxis[0], 0],
                                 [0, obstacles.gaussians[mode].major_semiaxis[0]]], dtype=torch.float32)
                    obst_num += 1

            predicted_coordinates = predicted_coordinates.permute(1, 0, 2)

            distances = torch.norm(pos[:, :, :] - predicted_coordinates[:, None, :], dim=-1)
            distances_reshaped = distances.permute(0, 2, 1)

            distance_threshold = 1.0
            in_collision = torch.zeros_like(distances_reshaped)
            in_collision[distances_reshaped <= distance_threshold] = 10

            collision_cost = in_collision.sum(dim=0).view(-1)
            print("Collision cost: ", collision_cost)
            print(collision_cost.shape, goal_cost.shape, 'shapes')
            print("Goal cost: ", goal_cost)
            return goal_cost + collision_cost
        else:
            pos = state[:, 0:2]
            return torch.clamp(
                torch.linalg.norm(pos - self.nav_goal, axis=1) - 0.05, min=0, max=1999
            )