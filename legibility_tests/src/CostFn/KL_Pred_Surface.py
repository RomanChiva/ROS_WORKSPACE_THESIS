#!/usr/bin/python3

import numpy as np
from scipy.interpolate import CubicSpline
from utils.cubicspline import CubicSpline2D
import torch
from sklearn.mixture import GaussianMixture
import time


class ObjectiveLegibility(object):

    def __init__(self, cfg, obstacles, interface):
        # Create two possible goals used in pred_model
        self.cfg = cfg
        self.goal_index = cfg.costfn.goal_index
        self.other_goal = 1 - self.goal_index
        self.goals = torch.tensor(self.cfg.costfn.goals).float()
        self.v_ref = torch.tensor(cfg.v_ref, device=cfg.mppi.device)
        self.x_ref = np.array([0, 3, 6, 9, 12, 15, 18, 21, 30, 34, 38])
        self.y_ref = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.reference_spline = CubicSpline2D(self.x_ref, self.y_ref)
        self.obstacles = obstacles
        self.interface = interface

    

    # Cost Function for free navigation (Straight to goal)
    def compute_cost(self, state, u, t, obst):
        ## The state coming in is a tensor with all the different trajectory samples
        ## Goal Cost
        goal_cost = self.goal_cost(state) 
        # Legibility Cost

        legibility_cost = self.legibility_cost(state)
        legibility_cost = legibility_cost.reshape(-1)  
        

        # Add them
        return goal_cost + 0.2*legibility_cost
     

    def goal_cost(self, state):
        
        state_goal = state.permute(1, 0, 2)
        # Now reshape to (T*K, nx)
        state_goal = state_goal.reshape(-1, self.cfg.nx)
        pos_goal = state_goal[:, 0:2]
        goal_cost = torch.linalg.norm(pos_goal - self.goals[self.goal_index], axis=1)

        return goal_cost
    

    def legibility_cost(self, state):

        ## Specify Distributions
        plan_means = state[:, :, 0:2] # Get only the positions from state    
        pred_means1, pred_means2, weights = self.pred_model(state)
        # Generate Samples
        samples_pred1 = self.GenerateSamplesReparametrizationTrick(pred_means1, self.cfg.costfn.sigma_pred, self.cfg.costfn.monte_carlo_samples)
        samples_pred2 = self.GenerateSamplesReparametrizationTrick(pred_means2, self.cfg.costfn.sigma_pred, self.cfg.costfn.monte_carlo_samples)
        samples_pred = self.merge_gmm(samples_pred1, samples_pred2, weights, self.cfg.costfn.monte_carlo_samples)

        # Score
        score_pred = self.score_GMM(samples_pred, pred_means1, self.cfg.costfn.sigma_pred, pred_means2, self.cfg.costfn.sigma_pred, weights)
        score_plan = self.multivariate_normal_log_prob(samples_pred, plan_means, self.cfg.costfn.sigma_plan)
        # Compute KL Divergence
        kl_div = torch.mean(score_pred - score_plan, dim=0)
        return kl_div


    def pred_model(self, state):

        # Get velocity as a scalar
        velocity =  np.linalg.norm(np.array([self.interface.odom_msg.twist.twist.linear.x,
                        self.interface.odom_msg.twist.twist.linear.y]))

        # Get position XY and make it tensor
        position =  torch.tensor([self.interface.odom_msg.pose.pose.position.x,
                                  self.interface.odom_msg.pose.pose.position.y], device=self.cfg.mppi.device)
        


        #### All these steps are to make sure the predictions align, and avoid being 1 timestep ahead#####
        pos = state[:, :, 0:2]
        
        # Create a tensor with the current position repeated K times
        current_position = position.repeat(pos.shape[1], 1)
        ## Stack this at first index of pos
        pos = torch.cat([current_position.unsqueeze(0), pos], dim=0)
        # Remove the last element of each trajectory to make it a T-1 trajectory
        pos = pos[:-1, :, :]
        ####################################################################################################
        

        # Find vectors to goals 1 and 2 (Goal[-] shape[1,2] pos shape [T, K, 2])
        vector_goal1 = self.goals[0] - pos
        vector_goal2 = self.goals[1] - pos

        # Find magnitudes of vectors
        magnitude_goal1 = torch.linalg.norm(vector_goal1, axis=2)
        magnitude_goal2 = torch.linalg.norm(vector_goal2, axis=2)

        # Find unit vectors
        unit_goal1 = vector_goal1 / magnitude_goal1.unsqueeze(-1)
        unit_goal2 = vector_goal2 / magnitude_goal2.unsqueeze(-1)

        # Predictions in constant velocity for goal 1 and 2
        pred_goal1 = pos + unit_goal1 * velocity
        pred_goal2 = pos + unit_goal2 * velocity

        # Find weights

        weights = self.Compute_Weights_Goals(pos)

        # Convert all to floar32
        pred_goal1 = pred_goal1.float()
        pred_goal2 = pred_goal2.float()
        weights = weights.float()

        return pred_goal1, pred_goal2, weights


    def Compute_Weights_Goals(self, pos):

        pos = pos.permute(1,0,2)

        # Find vectors to goals 1 and 2 (Goal[-] shape[1,2] pos shape [T, K, 2])
        vector_goal1 = self.goals[0] - pos
        vector_goal2 = self.goals[1] - pos

        # Find magnitudes of vectors
        magnitude_goal1 = torch.linalg.norm(vector_goal1, axis=2)
        magnitude_goal2 = torch.linalg.norm(vector_goal2, axis=2)

        # Find distance travelled so far
        distance = self.compute_path_length(self.interface.trajectory)
        

        path_lengths = self.compute_path_lengths(pos)
        
        ## Add length of path so far to path_lengths
        path_lengths = path_lengths + distance

    
        # FInd magnitude of goals from [0,0]
        V_g1_0 = torch.linalg.norm(self.goals[0])
        V_g2_0 = torch.linalg.norm(self.goals[1])
     

        # Compute weights
        weight_1 = torch.exp(V_g1_0 - path_lengths - magnitude_goal1)
        weight_2 = torch.exp(V_g2_0 - path_lengths - magnitude_goal2)
        

        # Normalize the weights
        weights = torch.stack([weight_1, weight_2], dim=2)
        weights = weights / weights.sum(dim=2).unsqueeze(-1)
       
        weights = weights.permute(1,0,2)

        return weights


    def create_covariance_matrix_tensor(self, means_shape, sigma):
        # Create a tensor of shape (n_distributions, n_components, n_dimensions, n_dimensions)
        n_distributions, n_components, n_dimensions = means_shape
        covariances = torch.zeros(n_distributions, n_components, n_dimensions, n_dimensions)
        
        # Set the diagonal elements to sigma
        covariances[..., torch.arange(n_dimensions), torch.arange(n_dimensions)] = sigma
        # Ensure covariance matrices are positive-definite
        covariances = torch.einsum('...ij,...jk->...ik', covariances, covariances.transpose(-2, -1))

        # Add a small constant to the diagonal of each covariance matrix
        jitter = torch.eye(n_dimensions) * 1e-6
        covariances = covariances + jitter.unsqueeze(0).unsqueeze(0)
        
        return covariances
        
    def GenerateSamplesReparametrizationTrick(self, means, sigma, n_samples):
        # means: tensor of shape (n_distributions, n_components, n_dimensions)
        # covariances: tensor of shape (n_distributions, n_components, n_dimensions, n_dimensions)
        # n_samples: int, number of samples to generate

        # Expand the means tensor to match the number of samples
        expanded_means = means.unsqueeze(0).expand(n_samples, -1, -1, -1)

        # Sample noise from a standard normal distribution
        noise = torch.randn_like(expanded_means)

        # Scale the noise by the Cholesky decomposition and add the mean
        samples = expanded_means + sigma * noise

        return samples


    def score_GMM(self, sample, means1, covariances1, means2, covariances2, weights):
        # means1, means2: tensors of shape (n_distributions, n_components, n_dimensions)
        # covariances1, covariances2: tensors of shape (n_distributions, n_components, n_dimensions, n_dimensions)
        # weights: tensor of shape (timestep, path_sample, 2)

        # Compute the log probability for each distribution
        score1 = self.multivariate_normal_log_prob(sample, means1, covariances1)
        score2 = self.multivariate_normal_log_prob(sample, means2, covariances2)

        # Weigh the scores based on the weights tensor
        weighted_score = weights[:,:,0] * score1 + weights[:,:,1] * score2

        return weighted_score

    def multivariate_normal_log_prob(self, x, means, sigma):
        # x: tensor of shape (..., n_dimensions)
        # means: tensor of shape (..., n_dimensions)
        # sigma: float, standard deviation of the distributions

        # Convert sigma to a tensor
        sigma = torch.tensor(sigma, dtype=torch.float32)

        # Compute the log probability of a multivariate normal distribution
        diff = x - means
        exponent = -0.5 * (diff / sigma) ** 2
        log_det = -x.shape[-1] * torch.log(sigma)
        log_2pi = -0.5 * x.shape[-1] * torch.log(torch.tensor(2.0 * 3.1415))

        # Sum the log probabilities over the last dimension
        log_prob = torch.sum(exponent + log_det + log_2pi, dim=-1)

        return log_prob


    def merge_gmm(self, samples_1, samples_2, weights_tensor, n_samples):


        samples_1 = samples_1.permute(1,2,0,3)
        samples_2 = samples_2.permute(1,2,0,3)

        select_distribution = torch.distributions.Categorical(weights_tensor).sample((n_samples,))
        
        # Select distribution number of samples should be on last dimension
        select_distribution = select_distribution.permute(1,2,0)
        
        # Combine both tensors adding a new dinemsion at the penultimatew position
        samples = torch.stack([samples_1, samples_2], dim=-2)
        select_distribution = select_distribution.unsqueeze(-1).unsqueeze(-1)
        select_distribution = select_distribution.expand(-1, -1, -1, -1, 2)
        # Gather the samples
        samples = samples.gather(-2, select_distribution)
        # Remove penultimate dimension
        samples = samples.squeeze(-2)
        samples = samples.permute(2,0,1,3)
        
        return samples

    
    def pred_model_viz(self, horizon, mode_weights='linear'):


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
        
        
        # Find Mode Weights
        if mode_weights == 'linear':
            weights = goal_magnitudes / goal_magnitudes.sum()
        elif mode_weights == 'exponential':
            weights = torch.exp(-goal_magnitudes)
            weights = weights / weights.sum()       

        # Return these as np arrays
        pred = pred.cpu().detach().numpy()
        weights = weights.cpu().detach().numpy()
        
        return pred, weights
    

    def compute_path_length(self, path):
        path_diff = np.diff(path, axis=0)
        path_length = np.linalg.norm(path_diff, axis=1).sum()
        return path_length
    

    def compute_path_lengths(self, paths):


        # Get position XY and make it tensor
        position =  torch.tensor([self.interface.odom_msg.pose.pose.position.x,
                                self.interface.odom_msg.pose.pose.position.y], device=self.cfg.mppi.device)
        # Subtract the current position from the rest of the path
        paths = paths - position.unsqueeze(0).unsqueeze(0)

        # append row of zeros to the beginning of the path
        paths = torch.cat([torch.zeros_like(paths[0:1]), paths], dim=0)

        # Compute the differences between consecutive points in each path
        path_diffs = torch.diff(paths, dim=0)
        
        # Compute the length of each segment in each path
        segment_lengths = torch.norm(path_diffs, dim=-1)

        # Compute the cumulative sum of segment lengths along the timesteps dimension
        path_lengths = torch.cumsum(segment_lengths, dim=0)

        return path_lengths