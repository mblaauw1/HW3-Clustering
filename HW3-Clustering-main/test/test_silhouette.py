

# write your silhouette score unit tests here
import sys
import os
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Add the path to the main project directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from cluster.kmeans import KMeans
from cluster.utils import make_clusters, plot_clusters, plot_multipanel
from cluster.silhouette import Silhouette

# Generate synthetic data
mat, true_labels = make_clusters(n=1500, m=100, k=6)
centers=6
kmeans = KMeans(centers)
transformed_mat = StandardScaler().fit_transform(mat)
centroids, error, sorted_points=kmeans.fit(transformed_mat)
silhouette_score=Silhouette.score(n=1500, m=1, k=6)

#silhouette_scores = silhouette.score(mat, sorted_points)

# Plot the clusters
#plot_clusters(mat, sorted_points, "test_clusters.png")

# Plot multipanel visualization
#plot_multipanel(mat, true_labels, sorted_points[0], sorted_points[1], "multipanel.png")
#print("Silhouette Scores:", silhouette_scores)

for i in range(3,11):
    assert silhouette_score[i] >= -1
    assert silhouette_score[i] <= 1

