import torch

# Given tensors
#predicted_coordinates = torch.randn((5, 4, 2))
#obstacle_coordinates = torch.randn((2, 4, 2))

predicted_coordinates = torch.tensor([[[1, 2], [2, 2], [3, 2], [4, 2]], [[0, 0], [0, 1], [0, 2], [0, 3]], [[4, 1], [3, 1], [2, 1], [1, 1]]], dtype=torch.float32)
obstacle_coordinates = torch.tensor([[[1, 1], [1, 2], [1, 3], [1, 4]], [[0, 0], [1, 1], [2, 2], [3, 3]]], dtype=torch.float32)

distances = torch.norm(predicted_coordinates[:, :, :] - obstacle_coordinates[:, None, :], dim=-1)
distances_reshaped = distances.permute(0, 2, 1)
# Threshold for distance
distance_threshold = 1.0

# Compute distances between all pairs of trajectory points and obstacles

within_bounds = torch.zeros_like(distances_reshaped)
within_bounds[distances_reshaped < 3] = -10

within_bounds_summed = within_bounds.sum(dim=0)
within_bounds_summed_reshaped = within_bounds_summed.view(-1)
# Check if any distance is below the threshold


print(within_bounds)
'''

predicted_coordinates = torch.tensor([[[1, 2], [0, 0], [4, 1]], [[2, 2], [0, 1], [3, 1]], [[3, 2], [0, 2], [2, 1]], [[4, 2], [0, 3], [1, 1]]], dtype=torch.float32)
obstacle_coordinates = torch.tensor([[[1, 1]], [[1, 2]], [[1, 3]], [[1, 4]]], dtype=torch.float32)
distances = torch.norm(predicted_coordinates[:, :, :] - obstacle_coordinates[:, None, :], dim=-1)
# Threshold for distance
distance_threshold = 1.0

# Compute distances between all pairs of trajectory points and obstacles

within_bounds = torch.zeros_like(distances)
within_bounds[distances <= 3] = -10
'''