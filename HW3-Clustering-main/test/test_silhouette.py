# write your silhouette score unit tests here
import sys
import os


# Add the path to the main project directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from cluster.kmeans import KMeans
from cluster.utils import make_clusters, plot_clusters, plot_multipanel
from cluster.silhouette import Silhouette

# Generate synthetic data
data, true_labels = make_clusters(n=1500, m=100, k=5)
kmeans = KMeans(k=5, tol=50000000)
kmeans.fit(data)
predicted_labels = kmeans.predict(data)
silhouette = Silhouette(k=5)
silhouette_scores = silhouette.score(data, predicted_labels)

# Plot the clusters
plot_clusters(data, predicted_labels, "test_clusters.png")

# Plot multipanel visualization
plot_multipanel(data, true_labels, predicted_labels, silhouette_scores, "multipanel.png")
print("Silhouette Scores:", silhouette_scores)

assert silhouette_scores >= -1
assert silhouette_scores <= 1

