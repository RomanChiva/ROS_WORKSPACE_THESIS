import numpy as np

# Given data
losses = np.array([10, 5, 8, 15, 20])
probabilities = np.array([0.1, 0.2, 0.3, 0.2, 0.2])

# Sort the losses and probabilities
sorted_indices = np.argsort(losses)
sorted_losses = losses[sorted_indices]
sorted_probabilities = probabilities[sorted_indices]

# Calculate cumulative probabilities
cumulative_probabilities = np.cumsum(sorted_probabilities)

# Define the confidence level (1 - alpha)
confidence_level = 0.3

# Find the VaR threshold
var_threshold = sorted_losses[np.searchsorted(cumulative_probabilities, confidence_level)]

# Find the index of the threshold for CVaR
cvar_index = np.searchsorted(cumulative_probabilities, confidence_level)

# Calculate CVaR as the average of losses beyond the threshold
cvar_threshold = np.mean(sorted_losses[cvar_index:])

print("VaR at {}% confidence level: {}".format(confidence_level * 100, var_threshold))
print("CVaR at {}% confidence level: {}".format(confidence_level * 100, cvar_threshold))
