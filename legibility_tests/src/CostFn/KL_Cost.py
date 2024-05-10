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
        self.goals, self.nav_goal = self.create_goals(cfg.goal, cfg.costfn.goal_separation, cfg.costfn.goal_index)
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
        return goal_cost + legibility_cost
     


    def goal_cost(self, state):
        
        state_goal = state.permute(1, 0, 2)
        # Now reshape to (T*K, nx)
        state_goal = state_goal.reshape(-1, self.cfg.nx)
        pos_goal = state_goal[:, 0:2]
        goal_cost = torch.linalg.norm(pos_goal - self.nav_goal, axis=1)

        return goal_cost
    

    def legibility_cost(self, state):

        # Create distributions
        # PLAN
        # Begin timer
        start = time.time()
        plan_means = state[:, :, 0:2] # Get only the positions from state
        print(f"Plan means shape: {plan_means.shape}")
        plan_covarances = self.create_covariance_matrix_tensor(plan_means.shape, self.cfg.costfn.sigma_plan)
        plan_distribution = self.DistributionTensor(plan_means, plan_covarances)
        # End timer
        end = time.time()
        print(f"Time to create plan distribution: {end - start} seconds")

        # PRED (Gaussian Mixture)
        # Begin timer
        start = time.time()
        pred1, pred2, weights = self.pred_model(state)
        # End timer
        end = time.time()
        print(f"Time to create pred model: {end - start} seconds")
        # Begin timer
        start = time.time()
        pred1_covariance = self.create_covariance_matrix_tensor(pred1.shape, self.cfg.costfn.sigma_pred)
        pred2_covariance = self.create_covariance_matrix_tensor(pred2.shape, self.cfg.costfn.sigma_pred)
        pred1_distribution = self.DistributionTensor(pred1, pred1_covariance)
        pred2_distribution = self.DistributionTensor(pred2, pred2_covariance)
        # End timer
        end = time.time()
        print(f"Time to create pred distribution: {end - start} seconds")

        # Begin timer
        start = time.time()
        # Compute KLDiv (pred||plan)
        ## SAMPLE FROM PLAN
        pred_samples = self.sample_gaussian_mixture(pred1_distribution, pred2_distribution, weights, self.cfg.costfn.monte_carlo_samples)
        # Compute the scores
        pred_score = self.score_GMM(pred_samples, pred1_distribution,pred2_distribution, weights)
        plan_score = plan_distribution.log_prob(pred_samples)
        # Compute Kl divegrence montecarlo (Mean of log_prob_pred - log_prob_plan) Mean on the 1st dimension
        # Should return a tensor with shape (T, K)
        kl_div = torch.mean(pred_score - plan_score, dim=0)
        # End timer
        end = time.time()
        print(f"Time to compute KL Divergence: {end - start} seconds")

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

        # stack the magnitudes of the goals
        magnitudes = torch.stack([magnitude_goal1, magnitude_goal2], dim=2)
        # Find the weights
        weights = magnitudes / magnitudes.sum(dim=2).unsqueeze(-1)

        # COmpute inverse squared of weights
        weights = 1 / weights**2

        # Convert all to floar32
        pred_goal1 = pred_goal1.float()
        pred_goal2 = pred_goal2.float()
        weights = weights.float()

        return pred_goal1, pred_goal2, weights



    def DistributionTensor(self, means, covariances):
        # means: tensor of shape (n_distributions, n_components, n_dimensions)
        # covariances: tensor of shape (n_distributions, n_components, n_dimensions, n_dimensions)

        # Create a MultivariateNormal for each distribution
        distributions = torch.distributions.MultivariateNormal(means, covariances)

        return distributions


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
        
    def sample_gaussian_mixture(self, distribution_tensor_1, distribution_tensor_2, weights_tensor, num_samples):


        # Sample from either distribution tensor 1 or 2 with probability based on weights tensor
        samples_1 = distribution_tensor_1.sample((num_samples,))
        samples_1 = samples_1.permute(1,2,0,3)
        samples_2 = distribution_tensor_2.sample((num_samples,))
        samples_2 = samples_2.permute(1,2,0,3)
        select_distribution = torch.distributions.Categorical(weights_tensor).sample((num_samples,))
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
        
    def score_GMM(self, sample, distribution1, distribution2, weights):
        # Calculate the score of the sample for each distribution
        score1 = distribution1.log_prob(sample)
        score2 = distribution2.log_prob(sample)
        
        # Weigh the scores based on the weights tensor(Weights tensor has shape (timestep, path_sample, 2))
        weighted_score = weights[:,:,0] * score1 + weights[:,:,1] * score2

        return weighted_score


    def create_goals(self, goal, separation, goal_index):

        separation_vector = np.array([0, separation])
        goal_1 = goal + separation_vector
        goal_2 = goal - separation_vector

        # Create a tensor with the two goals
        goals = torch.tensor([goal_1, goal_2], device=self.cfg.mppi.device)

        return goals, goals[goal_index]
    

    
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
    

