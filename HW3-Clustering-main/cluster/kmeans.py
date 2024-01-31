import numpy as np
from scipy.spatial.distance import cdist
import utils
import random
import pandas as pd

class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.centroids = None
        self.mat = None  # Added attribute to store input data

    def fit(self, mat: np.ndarray):
        self.mat = mat  # Store the input data

        # k-means++ initialization
        centroids = np.zeros((self.k, mat.shape[1]))
        centroids[0] = mat[random.choice(range(len(mat)))]
        
        for i in range(1, self.k):
            squared_distances = cdist(mat, centroids[:i])**2
            min_squared_distances = np.min(squared_distances, axis=1)
            probs = min_squared_distances / np.sum(min_squared_distances)
            centroids[i] = mat[np.random.choice(range(len(mat)), p=probs)]

        self.centroids = centroids

        for _ in range(self.max_iter):
            distances = cdist(mat, self.centroids)
            estimated_groups = np.argmin(distances, axis=1)

            new_centroids = np.array([mat[estimated_groups == j].mean(axis=0) for j in range(self.k)])

            # Check convergence
            if np.sum((new_centroids - self.centroids) ** 2) < self.tol:
                break

            self.centroids = new_centroids

    def predict(self, mat: np.ndarray) -> np.ndarray:
        distances = cdist(mat, self.centroids)
        return np.argmin(distances, axis=1)

    def get_error(self) -> float:
        distances = cdist(self.mat, self.centroids)
        return np.sum(np.min(distances**2, axis=1))

    def get_centroids(self) -> np.ndarray:
        return self.centroids