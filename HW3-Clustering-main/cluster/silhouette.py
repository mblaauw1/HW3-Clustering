import numpy as np
from scipy.spatial.distance import cdist

import numpy as np
from scipy.spatial.distance import cdist

class Silhouette:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        silhouette_scores = np.zeros(X.shape[0])

        for i in range(X.shape[0]):
            a_o = np.zeros(self.k)
            b_o = np.zeros(self.k)

            # Calculate a_o for each cluster
            for k in range(self.k):
                indices_same_cluster = np.where(y == k)[0]
                cluster_points = X[indices_same_cluster]
                a_o[k] = np.mean(cdist([X[i]], cluster_points))

            # Calculate b_o for each cluster
            for k in range(self.k):
                if k != y[i]:  # Skip the cluster to which point o belongs
                    indices_other_cluster = np.where(y == k)[0]
                    other_cluster_points = X[indices_other_cluster]
                    b_o[k] = np.mean(cdist([X[i]], other_cluster_points))
            y=y-1
            # Calculate silhouette score for point o
            s_o = (np.min(b_o[y[i]]) - a_o[y[i]]) / max(np.min(b_o[y[i]]), a_o[y[i]])
            silhouette_scores[i] = s_o

        return silhouette_scores