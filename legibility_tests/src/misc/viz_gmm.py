from sklearn.mixture import GaussianMixture
import numpy as np

from sklearn.mixture import GaussianMixture
import numpy as np

def GM(components, means, covariance, weights):

        distribution = GaussianMixture(n_components=components)

        # Set the parameters directly
        distribution.weights_ = weights
        distribution.means_ = means
        distribution.covariances_ = covariance
        distribution.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(covariance))
        # Im not sure why this is needed?

        return distribution


def kl_divergence_monte_carlo(gmm_p, gmm_q, num_samples=10**3):
        # Draw samples from p
        samples, _ = gmm_p.sample(num_samples)

        # Compute log likelihood of samples under p and q
        log_p = gmm_p.score_samples(samples)
        log_q = gmm_q.score_samples(samples)

        # Compute and return KL divergence
        return np.mean(log_p - log_q)

# Create two GMMs
# Create two GMMs
gmm_p = GM(2, np.array([[0, 0], [2, 0]]), np.array([[[1, 0], [0, 1]], [[1, 0], [0, 1]]]), np.array([0.5, 0.5]))
gmm_q = GM(2, np.array([[1, 1], [2, 1]]), np.array([[[1, 0], [0, 1]], [[1, 0], [0, 1]]]), np.array([0.5, 0.5]))

# Visualize the GMMs
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def plot_gmm(gmm, ax, color='k'):
        for i in range(gmm.n_components):
                mean = gmm.means_[i]
                cov = gmm.covariances_[i]
                lambda_, v = np.linalg.eig(cov)
                lambda_ = np.sqrt(lambda_)
                ell = Ellipse(
                xy=mean, width=lambda_[0]*2, height=lambda_[1]*2,
                angle=np.rad2deg(np.arccos(v[0, 0])), color=color
                )
                ell.set_facecolor('none')
                ax.add_artist(ell)
                ax.plot(mean[0], mean[1], 'o', markersize=10, color=color)

fig, ax = plt.subplots()
plot_gmm(gmm_p, ax, color='r')
plot_gmm(gmm_q, ax, color='b')
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
plt.show()


# Comp[ute KL
print(kl_divergence_monte_carlo(gmm_p, gmm_q))