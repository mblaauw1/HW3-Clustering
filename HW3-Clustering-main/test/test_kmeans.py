

import sys
import os


# Add the path to the main project directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from cluster.kmeans import KMeans
from cluster.utils import make_clusters, plot_clusters, plot_multipanel
from cluster.silhouette import Silhouette


np.random.seed(45)
assert k <= n
# Generate synthetic data
data, true_labels = make_clusters(n=1500, m=100, k=5)
kmeans = KMeans(k=5, tol=1E-6)
kmeans.fit(data)
print(kmeans.estimated_groups)
predicted_labels = kmeans.predict(data)
#silhouette = Silhouette(k=5)
#silhouette_scores = silhouette.score(data, kmeans.estimated_groups)

# Plot the clusters
plot_clusters(data, kmeans.estimated_groups, "test_clusters3.png")

# Plot multipanel visualization
#kmeaplot_multipanel(data, true_labels, kmeans.estimated_groups, silhouette_scores, "multipanel.png")

error=kmeans.get_error()
print(error)
assert error < kmeans.tol
