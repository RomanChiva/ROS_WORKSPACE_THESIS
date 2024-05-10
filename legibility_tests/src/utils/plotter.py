import time

import matplotlib.pyplot as plt

import sys
import os
# Get the parent directory of the current script
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# Add the parent directory to the Python path
sys.path.append(parent_dir)

class LiveBarChart:
    def __init__(self, goals):
        self.goals = goals
        self.probabilities = [0] * len(goals)
        self.fig, self.ax = plt.subplots()
        self.ax.bar(range(len(goals)), self.probabilities, color='C0')  # Set color to 'C0' for the first bar
        self.ax.set_xticks(range(len(goals)))
        self.ax.set_xticklabels([f'Goal {i+1}' for i in range(len(goals))])
        plt.ion()
        plt.show()

    def update(self, probabilities):
        self.probabilities = probabilities
        self.ax.clear()
        # Set different colors for each bar
        for i, prob in enumerate(self.probabilities):
            self.ax.bar(i, prob, color=f'C{i}')
        self.ax.set_xticks(range(len(self.goals)))
        self.ax.set_xticklabels([f'Goal {i+1}' for i in range(len(self.goals))])
        # Set y axis range 0 to 1
        self.ax.set_ylim(0, 1)
        # Add grid
        self.ax.grid()
        plt.draw()
        plt.pause(0.2)



# Show trajectory

def show_trajectory(goals, trajectory):

    # Show origin
    plt.plot(0, 0, 'ro')
    # Plot trajectory with red dotted line
    plt.plot([x[0] for x in trajectory], [x[1] for x in trajectory], 'r--')        
    # Show goal locations in different colors
    for i, goal in enumerate(goals):
        plt.plot(goal[0], goal[1], 'o', color='C'+str(i))
    # Add title with trajectory duration
    plt.title(f'Trajectory duration: {len(trajectory)}')
    plt.show()




# Make two 3D plots 
import numpy as np

def weights235(goals, position):
    distances = np.array([np.linalg.norm(np.array(goal) - np.array(position)) for goal in goals])
    numerator = np.exp(-distances)
    denominator = np.exp(-distances).sum()
    return numerator / denominator

def weights(goals, positions):
    # Goals is a list of goal positions
    # Positions is a np array of positions shape 10000x2

    # Find distance of each point to goal 1
    distances1 = np.linalg.norm(positions - goals[0], axis=1)
    # Find distance of each point to goal 2
    distances2 = np.linalg.norm(positions - goals[1], axis=1)

    # Calculate the weights for each goal
    weights1 = np.exp(-distances1)
    print(weights1, 'weights1')
    weights2 = np.exp(-distances2)
    print(weights2, 'weights2')

    # Combine both weight vectors into a 10000x2 matrix
    weights = np.array([weights1, weights2]).T  # Transpose to get 10000x2 shape

    # Normalize weights on axis 1
    weights = weights / weights.sum(axis=1, keepdims=True)  # Normalize across goals for each position

    return weights

def weights_lin(goals, positions):
    # Goals is a list of goal positions
    # Positions is a np array of positions shape 10000x2

    # Find distance of each point to goal 1
    distances1 = np.linalg.norm(positions - goals[0], axis=1)
    # Find distance of each point to goal 2
    distances2 = np.linalg.norm(positions - goals[1], axis=1)

    # Calculate the weights for each goal
    weights1 = 1 / distances1**2
    weights2 = 1 / distances2**2

    # Combine both weight vectors into a 10000x2 matrix
    weights = np.array([weights1, weights2]).T  # Transpose to get 10000x2 shape

    # Normalize weights on axis 1
    weights = weights / weights.sum(axis=1, keepdims=True)  # Normalize across goals for each position

    return weights


def plot_weights(goals, weight_func, x_range, y_range):
    fig, axs = plt.subplots(1, len(goals), figsize=(12, 6))

    # Create a grid of points in the x-y plane
    x = np.linspace(*x_range, 500)
    y = np.linspace(*y_range, 500)
    X, Y = np.meshgrid(x, y)
    positions = np.array([X, Y]).T.reshape(-1, 2)

    # Calculate weights for each position
    weights = weight_func(goals, positions)

    # Reshape weights to match the grid
    weights = weights.reshape(X.shape[0], X.shape[1], len(goals))

    # Transpose weights to match x-y convention in plotting
    weights = weights.transpose(1, 0, 2)

    # Make Heat Map for each goal
    for i in range(len(goals)):
        im = axs[i].imshow(weights[:, :, i], cmap='coolwarm', extent=[x_range[0], x_range[1], y_range[0], y_range[1]], origin='lower')

        # Add goal locations
        axs[i].plot(goals[i][0], goals[i][1], 'o', color='C'+str(i))

        # Add colorbars
        cbar = fig.colorbar(im, ax=axs[i])
        cbar.set_label(f'Goal {i+1} Weight')

        # Add titles
        axs[i].set_title(f'Goal {i+1}')

    plt.show()







# Write unit test for weights function
def test_weights():
    goals = [[-2, 5], [2, 5]]
    positions = np.array([[100, 100], [0, 1], [0, 2]])
    expected_result = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    assert np.allclose(weights(goals, positions), expected_result), 'Test Failed!'


# goals = [[-2, 5], [2, 5]]
# positions = np.array([[100, 100], [0, 1], [0, 2]])
# print(weights_lin(goals, positions))

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import multivariate_normal

# # Define the parameters for the three Gaussians
# mu1 = np.array([0, 0])
# cov1 = np.array([[1, 0], [0, 1]])

# mu2 = np.array([1, 1])
# cov2 = np.array([[1, 0], [0, 1]])

# mu3 = np.array([-1, -1])
# cov3 = np.array([[1, 0], [0, 1]])

# # Create a grid of points at which to evaluate the Gaussians
# x = np.linspace(-3, 3, 100)
# y = np.linspace(-3, 3, 100)
# X, Y = np.meshgrid(x, y)
# pos = np.dstack((X, Y))

# # Evaluate the Gaussians at the grid points
# Z1 = multivariate_normal(mu1, cov1).pdf(pos)
# Z2 = multivariate_normal(mu2, cov2).pdf(pos)
# Z3 = multivariate_normal(mu3, cov3).pdf(pos)

# # Create the surface plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, Y, Z1, color='red', alpha=0.5)
# ax.plot_surface(X, Y, Z2, color='red', alpha=0.5)
# ax.plot_surface(X, Y, Z3, color='blue', alpha=0.5)
# plt.show()