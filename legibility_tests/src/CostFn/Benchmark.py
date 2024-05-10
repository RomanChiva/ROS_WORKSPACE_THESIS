#!/usr/bin/python3

import numpy as np
from scipy.interpolate import CubicSpline
from utils.cubicspline import CubicSpline2D
import torch
from sklearn.mixture import GaussianMixture



class ObjectiveBenchmark(object):

    def __init__(self, cfg, obstacles, interface):
        # Create two possible goals used in pred_model
        self.cfg = cfg
        self.goal_index = self.cfg.costfn.goal_index
        self.other_goal = 1 - self.cfg.costfn.goal_index
        self.goals = torch.tensor(self.cfg.costfn.goals)
        self.v_ref = torch.tensor(cfg.v_ref, device=cfg.mppi.device)
        self.x_ref = np.array([0, 3, 6, 9, 12, 15, 18, 21, 30, 34, 38])
        self.y_ref = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.reference_spline = CubicSpline2D(self.x_ref, self.y_ref)
        self.obstacles = obstacles
        self.interface = interface

    

    # Cost Function for free navigation (Straight to goal)
    def compute_cost(self, state, u, t, obst):
        ## The state coming in is a tensor with all the different trajectory samples
        print(state.shape)
        # Goal cost
        goal_cost = self.goal_cost(state, index=self.goal_index)
        # Other goal cost
        other_goal_cost = self.goal_cost(state, index = self.other_goal)
        
        total_cost =  (1+ self.alpha())*goal_cost - other_goal_cost
        return total_cost
     

    def alpha(self):

        grad = self.cfg.costfn.alpha_max/self.cfg.costfn.alpha_steps

        if self.interface.timesteps < self.cfg.costfn.alpha_steps:
            return self.interface.timesteps * grad
        else:
            return self.cfg.costfn.alpha_max

        

    def goal_cost(self, state, index=0):

        pos = state[:, :, 0:2] # Get only the positions
        state_goal = state.permute(1, 0, 2)
        # Now reshape to (T*K, nx)
        state_goal = state_goal.reshape(-1, self.cfg.nx)
        pos_goal = state_goal[:, 0:2]
        goal_cost = torch.linalg.norm(pos_goal - self.goals[index], axis=1)

        return goal_cost
    
    

    def pred_model(self, horizon):


        ## Get position XY
        position = [self.interface.odom_msg.pose.pose.position.x,
                                  self.interface.odom_msg.pose.pose.position.y]

        # COnvert position to torch tensor
        position = torch.tensor(position, device=self.cfg.mppi.device)
        # Get velocity as a scalar
        velocity =  np.linalg.norm(np.array([self.interface.odom_msg.twist.twist.linear.x,
                        self.interface.odom_msg.twist.twist.linear.y]))
        
        # Geometry poistion to goals
        goal_vectors = self.goals - position
        goal_magnitudes = torch.linalg.norm(goal_vectors, axis=1)
        goal_units = goal_vectors / goal_magnitudes.view(-1,1)

        # Create a 1D tensor with numbers from 1 to horizon including the horizon
        t = torch.arange(1, horizon + 1, device=self.cfg.mppi.device)*velocity*(1/self.interface.frequency)

        # Create a 3d tensor with the unit vector of the goals matrix repeated horizon times
        pred = goal_units.unsqueeze(2).repeat(1, 1,horizon)

        # Multiply the unit vectors by the time*velocity to get the predicted position
        pred = pred * t.view(1,1,-1)
        # Add current position to predictions to get the absolute position
        # Pred is 2x2x20 with 2 goals, 2 dimensions and 20 timesteps
        pred = pred + position.view(1,2,1) 
        # Swap the dimensions to 2x20x2
        pred = pred.permute(0,2,1)
        
     
        weights = torch.exp(-goal_magnitudes)
        weights = weights / weights.sum()       

        # Return these as np arrays
        pred = pred.cpu().detach().numpy()
        weights = weights.cpu().detach().numpy()
        
        return pred, weights
    

    def pred_original(self, trajectory):
        # (Assume Equal Priors)

        trajectory = np.array(trajectory)
        pathlen = self.compute_path_length(trajectory)

        ## Get position XY
        position = np.array([self.interface.odom_msg.pose.pose.position.x,
                                  self.interface.odom_msg.pose.pose.position.y])
        
        # Compute distance from goals
        distance_goals = np.linalg.norm(self.goals - position, axis=1)

        VG_0 = np.linalg.norm(self.goals, axis=1)

        # Compute the weights
        weight_1 = np.exp(VG_0[0] - pathlen - distance_goals[0])
        weight_2 = np.exp(VG_0[1] - pathlen - distance_goals[1])

        # Normalize the weights
        weights = np.array([weight_1, weight_2])
        weights = weights / weights.sum()

        return weights


        




    
    def compute_path_length(self, path):
        path_diff = np.diff(path, axis=0)
        path_length = np.linalg.norm(path_diff, axis=1).sum()
        return path_length




