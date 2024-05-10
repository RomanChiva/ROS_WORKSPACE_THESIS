import torch
import time



def p_distrib(goals, position, traj_len, states, rationality):

    ## Mode 1

    optimal1 = torch.linalg.norm(goals[0])
    position_to_goal_magnitude1 = torch.linalg.norm(position - goals[0])
    dist_to_goal = states + position - goals[0]
    dist_to_goal = torch.linalg.norm(dist_to_goal, axis=2)

    ## Weight Mode 1
    w1 = torch.exp(rationality*(optimal1 - traj_len -position_to_goal_magnitude1))

    ## Distribution Mode 1 
    p1 = torch.exp(rationality*(optimal1 - traj_len - dist_to_goal))
    

    ## Mode 2

    optimal2 = torch.linalg.norm(goals[1])
    position_to_goal_magnitude2 = torch.linalg.norm(position - goals[1])
    dist_to_goal = states + position - goals[1]
    dist_to_goal = torch.linalg.norm(dist_to_goal, axis=2)

    ## Weight Mode 2
    w2 = torch.exp(rationality*(optimal2 - traj_len -position_to_goal_magnitude2))

    ## Distribution Mode 2
    p2 = torch.exp(rationality*(optimal2 - traj_len - dist_to_goal))


    ## Nomralize the distribution

    p = w1*p1 + w2*p2
    p = p/p.sum()


    return p



range_x = [-3, 3]
range_y = [0, 5]

## Create a 3D tensor where the first two dimensions are the grid and the third dimension is the position x, y
## The grid is a 2D grid that represents the environment
grid = torch.zeros(100, 100, 2)
grid[:,:,0] = torch.linspace(range_x[0], range_x[1], 100).view(1, -1).repeat(100, 1)
grid[:,:,1] = torch.linspace(range_y[0], range_y[1], 100).view(-1, 1).repeat(1, 100)

goals = torch.tensor([[-5, 10], [5, 10]]).float()

position = torch.tensor([0, 1]).float()

traj_len = 1

rationality = 1

p = p_distrib(goals, position, traj_len, grid, rationality)


# Calculate entropy of P 
entropy = -torch.sum(p*torch.log2(p))
print(entropy)


# Plot the distribution as a 3D surface

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(grid[:,:,0].numpy(), grid[:,:,1].numpy(), p.numpy())

plt.show()
