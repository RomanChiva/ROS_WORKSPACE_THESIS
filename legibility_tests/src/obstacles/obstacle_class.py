import time
import torch
import numpy as np
import matplotlib.pyplot as plt

from omegaconf import OmegaConf
from scipy.integrate import nquad, dblquad
from scipy.stats import multivariate_normal
from multiprocessing import Pool, set_start_method
from typing import List


class DynamicObstacles(object):

    def __init__(self, cfg, cov, num_obstacles) -> None:

        # Set meta parameters
        self.print_time = cfg.obstacles.print_time  # Set to True to print the time it takes to perform certain operations
        self.use_batch_gaussian = cfg.obstacles.use_gaussian_batch  # True is faster

        if cov.ndim == 2:
            cov = cov.unsqueeze(0)

        # Save the inputs as the actual current state of the obstacle
        self.cfg = cfg
        self.state_cov = cov
        # Initialise the predicted states of the obstacle
        self.predicted_coordinates = None
        self.predicted_covs = None

        # Save mppi parameters
        self.N_obstacles = num_obstacles
        self.N_rollouts = cfg.mppi.num_samples
        self.t = cfg.mppi.horizon

        # Set values used for monte carlo integration
        self.N_monte_carlo = cfg.obstacles.N_monte_carlo  # NOTE: Has large influence on the runtime of the cost calculation
        self.integral_radius = cfg.obstacles.integral_radius
        sample_bound = cfg.obstacles.sample_bound
        self.map_x0 = self.map_y0 = -sample_bound
        self.map_x1 = self.map_y1 = sample_bound

        # Sample a grid of points of shape (self.N_monte_carlo, 2) with x and y ranging from -5 to 5
        samples_x = torch.rand(self.N_monte_carlo, device=self.cfg.mppi.device) * (
                self.map_x1 - self.map_x0) + self.map_x0
        samples_y = torch.rand(self.N_monte_carlo, device=self.cfg.mppi.device) * (
                self.map_y1 - self.map_y0) + self.map_y0
        self.samples = torch.stack((samples_x, samples_y), dim=1)

    def update_monte_carlo_samples(self, robot_x, robot_y):
        samples_x = torch.rand(self.N_monte_carlo, device=self.cfg.mppi.device) * (
                    self.map_x1 - self.map_x0) + self.map_x0
        samples_y = torch.rand(self.N_monte_carlo, device=self.cfg.mppi.device) * (
                    self.map_y1 - self.map_y0) + self.map_y0
        samples_x += robot_x
        samples_y += robot_y
        self.samples = torch.stack((samples_x, samples_y), dim=1)

    def update_predicted_states(self, coordinates, covs):
        self.predicted_coordinates = coordinates
        self.predicted_covs = covs

    # Function that can be called to create all Gaussians or only the most efficient, batch Gaussians
    def create_gaussians(self, x, y, cov, use_only_batch_gaussian=True):

        # Save x, y and cov in the correct format
        if isinstance(x, int) or isinstance(x, float):
            x = torch.tensor([x], device=self.cfg.mppi.device)

        if isinstance(y, int) or isinstance(y, float):
            y = torch.tensor([y], device=self.cfg.mppi.device)

        if cov.ndim == 2:
            cov = cov.unsqueeze(0)

        # Set initial values for the gaussian
        self.coordinates = torch.stack((x, y), dim=1)
        self.cov = cov

        # Create only batch Gaussian
        if use_only_batch_gaussian:
            self.update_gaussian_batch(self.coordinates, self.cov)
        # Otherwise create all Gaussians
        else:
            self.update_gaussian_scipy(self.coordinates, self.cov)
            self.update_gaussian(self.coordinates, self.cov)
            self.update_gaussian_batch(self.coordinates, self.cov)

    ########## SCIPY VERSION ##########

    def update_gaussian_scipy(self, coordinates, cov):

        # Create Gaussian used by the scipy merhod
        coordinates = coordinates.cpu()[0]
        cov = cov.cpu()[0]

        # Create the Gaussian
        self.gaussian_scipy = multivariate_normal(mean=coordinates, cov=cov)

    def integrand_scipy(self, x, y):
        point = np.array([x, y])
        return self.gaussian_scipy.pdf(point)

    def integrate_gaussian_scipy(self, x0, x1, y0, y1):
        # Define the limits for the integration
        limits = [(x0, x1), (y0, y1)]

        # Perform the integration using nquad and the wrapper function
        integral, _ = nquad(self.integrand_scipy, limits)
        return integral

    ########## UPDATE TORCH GAUSSIANS ##########

    # Function that creates a list of torch Gaussians
    def update_gaussian(self, coordinates, cov):

        t = time.time()

        # Create Gaussian used by the torch and Monte Carlo method
        self.torch_gaussians = []
        for i in range(len(coordinates)):
            self.torch_gaussians.append(
                torch.distributions.multivariate_normal.MultivariateNormal(coordinates[i], cov[i]))

        if self.print_time:
            print(f"Time to create torch gaussians: {time.time() - t}")

        # Sample the torch Gaussian for Monte Carlo integration
        self.monte_carlo_sample()

        # Function that samples N points from the Gaussian distributions

    def monte_carlo_sample(self):

        t = time.time()

        # Calculate the log probabilities of the samples and convert to probabilities
        self.pdf_values = torch.zeros(self.N_monte_carlo, device=self.cfg.mppi.device)

        for gaussian in self.torch_gaussians:
            log_probs = gaussian.log_prob(self.samples)
            self.pdf_values += torch.exp(log_probs)

        if self.print_time:
            print(f"Time to calculate pdf values: {time.time() - t}")

    ########## UPDATE TORCH GAUSSIANS BATCH ##########

    # Function that creates a single batch of torch Gaussians
    def update_gaussian_batch(self, coordinates, cov):

        # Create Gaussian batch used by the torch and Monte Carlo method
        t = time.time()

        self.torch_gaussian_batch = torch.distributions.multivariate_normal.MultivariateNormal(coordinates, cov)

        if self.print_time:
            print(f"Time to create torch gaussians batch: {time.time() - t}")

        # Sample the torch Gaussian for Monte Carlo integration
        self.monte_carlo_sample_batch()

    def monte_carlo_sample_batch(self):

        t = time.time()

        # Expand points to match the batch size and compute log_prob
        expanded_points = self.samples.unsqueeze(1).expand(-1, self.torch_gaussian_batch.batch_shape[0], -1)
        log_probs = self.torch_gaussian_batch.log_prob(expanded_points)

        # Calculate the sum of the pdf values for each sample
        # Split the calculation into the timesteps
        if self.cfg.mppi.calculate_cost_once and self.cfg.obstacles.split_calculation:  # NOTE: This should fix the problem of the sum being calculated over all timesteps

            self.sum_pdf = torch.zeros((self.t, self.N_monte_carlo), device=self.cfg.mppi.device)
            for i in range(self.t):
                self.sum_pdf[i] = torch.exp(log_probs[:, i * self.N_obstacles:(i + 1) * self.N_obstacles]).sum(dim=1)

            # # CHAT REWRITE: The above code can be replaced by the following code
            # # Reshape the log_probs tensor to have dimensions (N_monte_carlo, t, N_obstacles)
            # reshaped_log_probs = log_probs.view(-1, self.t, self.N_obstacles)
            # # Calculate the exponential of log_probs and sum over the last dimension
            # exp_log_probs_sum = torch.exp(reshaped_log_probs).sum(dim=2)
            # # The sum_pdf tensor is now ready without the use of a for loop
            # self.sum_pdf = exp_log_probs_sum.transpose(0, 1)

        # Calculate for all inputs at once
        else:
            self.sum_pdf = torch.exp(log_probs).sum(dim=1)
            # self.sum_pdf = torch.exp(log_probs).max(dim=1).values  # Take the max rather than sum over all obstacles

        if self.print_time:
            print(f"Time to calculate pdf values batch: {time.time() - t}")

    ########## MONTE CARLO VERSION ##########

    # Calculate the integral of the gaussian over a rectangular area using Monte Carlo integration
    # This version takes in a single value for all bounds
    '''
    def integrate_monte_carlo(self, x0, x1, y0, y1):

        # Check which samples are within the specified bounds
        within_bounds = ((self.samples[:, 0] >= x0) & (self.samples[:, 0] <= x1) &
                         (self.samples[:, 1] >= y0) & (self.samples[:, 1] <= y1))

        if self.use_batch_gaussian:
            mean_within_bounds = torch.mean(self.sum_pdf[within_bounds])
        else:
            mean_within_bounds = torch.mean(self.pdf_values[within_bounds])

        # The integral is approximated as the proportion of points within the bounds multiplied by the area of the
        # rectangular region
        area = (x1 - x0) * (y1 - y0)
        integral_estimate = mean_within_bounds * area

        return integral_estimate
    '''

    # Calculate the integral of the gaussian over a rectangular area using Monte Carlo integration
    # This version takes in tensors for all bounds
    def integrate_one_shot_monte_carlo(self, x0, x1, y0, y1):

        if not isinstance(x0, torch.Tensor):
            x0, x1, y0, y1 = map(lambda v: torch.as_tensor(v, device=self.cfg.mppi.device), (x0, x1, y0, y1))

        # Check which samples are within the specified bounds
        within_x_bounds = (self.samples[:, 0, None] >= x0) & (self.samples[:, 0, None] <= x1)
        within_y_bounds = (self.samples[:, 1, None] >= y0) & (self.samples[:, 1, None] <= y1)
        within_bounds = within_x_bounds & within_y_bounds

        # Mask the values of the pdf_values tensor with the within_bounds tensor and calculate the column sums
        if self.use_batch_gaussian:
            masked_values = self.sum_pdf[:, None] * within_bounds
        else:
            masked_values = self.pdf_values[:, None] * within_bounds

        column_sums = torch.sum(masked_values, dim=0)
        true_counts = torch.sum(within_bounds, dim=0)

        # Calculate the mean by dividing the sum by the count and avoid division by zero by using torch.where
        means_within_bounds = torch.where(true_counts > 0, column_sums / true_counts, torch.tensor(0.0))

        # The integral is approximated as the proportion of points within the bounds multiplied by the area of the
        # rectangular region
        area = (x1 - x0) * (y1 - y0)
        integral_estimate = means_within_bounds * area

        return integral_estimate

    # Create a function that does the same as integrate_one_shot_monte_carlo but it takes in x and y which are the
    # centers of circles The within bounds check has to be done on all samples if they are within a circle with radius r
    def integrate_one_shot_monte_carlo_circles(self, x, y):

        # # Create the tensors for x, y and r
        # x, y = map(lambda v: torch.as_tensor(v, device=self.cfg.mppi.device), (x, y))
        # r = torch.ones((len(x)), device=self.cfg.mppi.device) * self.integral_radius

        # Check which samples are within the specified bounds
        within_bounds = ((self.samples[:, 0, None] - x) ** 2 + (
                self.samples[:, 1, None] - y) ** 2 <= self.integral_radius ** 2)

        # Calculate per timestep
        if self.cfg.mppi.calculate_cost_once and self.cfg.obstacles.split_calculation:  # NOTE: This should fix the problem of the sum being calculated over all timesteps
            # Within bounds is of shape (N_monte_carlo, T*N_rollouts). Here reshape to (N_mounte_carlo, T, N_rollouts)
            within_bounds = within_bounds.reshape(self.N_monte_carlo, self.t, self.N_rollouts)
            within_bounds = within_bounds.permute(1, 0, 2)

            if self.use_batch_gaussian:
                masked_values = self.sum_pdf[:, :, None] * within_bounds
            else:
                masked_values = self.pdf_values[:, :, None] * within_bounds

            column_sums = torch.sum(masked_values, dim=1)
            true_counts = torch.sum(within_bounds, dim=1)

            # Reshape column_sums and true_counts to (T*N_rollouts)
            column_sums = column_sums.reshape(self.t * self.N_rollouts)
            true_counts = true_counts.reshape(self.t * self.N_rollouts)

        # Calculate for all inputs at once
        else:
            if self.use_batch_gaussian:
                masked_values = self.sum_pdf[:,
                                None] * within_bounds  # Change self.pdf_values to self.sum_pdf to use batch version
            else:
                masked_values = self.pdf_values[:, None] * within_bounds

            column_sums = torch.sum(masked_values, dim=0)
            true_counts = torch.sum(within_bounds, dim=0)

        means_within_bounds = torch.where(true_counts > 0, column_sums / true_counts, torch.tensor(0.0))

        # The integral is approximated as the proportion of points within the bounds multiplied by the area of the
        # rectangular region
        integral_estimate = means_within_bounds * np.pi * self.integral_radius ** 2

        return integral_estimate
