#!/usr/bin/python3

import numpy as np
from scipy.interpolate import CubicSpline
from utils.cubicspline import CubicSpline2D
import torch

from sklearn.mixture import GaussianMixture



class ObjectiveFreeNav(object):
    def __init__(self, cfg, obstacles, interface):
        self.nav_goal = torch.tensor(cfg.goal, device=cfg.mppi.device)
        self.v_ref = torch.tensor(cfg.v_ref, device=cfg.mppi.device)
        self.x_ref = np.array([0, 3, 6, 9, 12, 15, 18, 21, 30, 34, 38])
        self.y_ref = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.reference_spline = CubicSpline2D(self.x_ref, self.y_ref)
        self.obstacles = obstacles
        self.cfg = cfg
        self.interface = interface

    

    # Cost Function for free navigation (Straight to goal)
    def compute_cost(self, state, u, t, obst=None):
        
        ## The state coming in is a tensor with all the different trajectory samples

        velocity_cost = torch.square(u[:, 0] - self.v_ref)
        if self.cfg.mppi.calculate_cost_once:
            
            pos = state[:, :, 0:2] # Get only the positions
            state_goal = state.permute(1, 0, 2)
            # Now reshape to (T*K, nx)
            state_goal = state_goal.reshape(-1, self.cfg.nx)
            pos_goal = state_goal[:, 0:2]

            goal_cost = torch.linalg.norm(pos_goal - self.nav_goal, axis=1)

            
            return goal_cost
        else:
            pos = state[:, 0:2]
            return torch.clamp(
                torch.linalg.norm(pos - self.nav_goal, axis=1) - 0.05, min=0, max=1999


            )