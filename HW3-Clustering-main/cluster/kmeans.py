import numpy as np
from scipy.spatial.distance import cdist
import random

class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.centroids = None
        self.mat = None  # Added attribute to store input data
        self.error = None  # Initialize error attribute

    def fit(self, mat: np.ndarray):
        if not isinstance(mat, np.ndarray):
            raise ValueError("Input matrix must be a numpy array.")

        if mat.shape[0] < self.k:
            raise ValueError("Number of clusters (k) should be less than the number of data points.")

        self.mat = mat  # Store the input data

        # k-means++ initialization
        centroids = np.zeros((self.k, mat.shape[1]))
        centroids[0] = mat[random.choice(range(len(mat)))]

        for i in range(1, self.k):
            squared_distances = cdist(mat, centroids[:i]) ** 2
            min_squared_distances = np.min(squared_distances, axis=1)
            probs = min_squared_distances / np.sum(min_squared_distances)
            centroids[i] = mat[np.random.choice(range(len(mat)), p=probs)]

        self.centroids = centroids

        for _ in range(self.max_iter):
            distances = cdist(mat, self.centroids)
            estimated_groups = np.argmin(distances, axis=1)

            new_centroids = np.array([mat[estimated_groups == j].mean(axis=0) for j in range(self.k)])

            # Check convergence
            self.error = np.sum((new_centroids - self.centroids) ** 2)
            if self.error < self.tol:
                break

            self.centroids = new_centroids

        self.estimated_groups = estimated_groups
        

    def predict(self, mat: np.ndarray) -> np.ndarray:
        if self.centroids is None:
            raise ValueError("Model has not been fitted. Call fit() before predict().")
        distances = cdist(mat, self.centroids)
        return np.argmin(distances, axis=1)

    def get_error(self) -> float:
        if self.error is None:
            raise ValueError("Error not available. Fit the model first.")
        return self.error

    def get_centroids(self) -> np.ndarray:
        if self.centroids is None:
            raise ValueError("Centroids not available. Fit the model first.")
        return self.centroids