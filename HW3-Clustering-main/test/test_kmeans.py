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


#np.random.seed(45)

# Generate synthetic data
#data, true_labels = make_clusters(n=1500, m=100, k=6)
#kmeans = KMeans(k=4)
#KMeans.fit(4, 1E-4)
#print(KMeans.working_df)
#predicted_labels = KMeans.predict(data)
#silhouette = Silhouette(k=5)
#silhouette_scores = silhouette.score(data, kmeans.estimated_groups)

# Plot the clusters
#plot_clusters(data, KMeans.working_df[2], "test_clusters3.png")

# Plot multipanel visualization
#kmeaplot_multipanel(data, true_labels, kmeans.estimated_groups, silhouette_scores, "multipanel.png")

#error=kmeans.get_error()
#print(kmeans.error)
#assert KMeans.error < KMeans.tol




centers = 6
tol=1E-6
mat, true_labels = make_clusters(n=1500, m=100, k=6)
transformed_mat = StandardScaler().fit_transform(mat)
# Fit centroids to dataset
kmeans = KMeans(centers)
centroids, error, sorted_points=kmeans.fit(transformed_mat)
# View results
#class_centers, classification = kmeans.evaluate(transformed_mat)
#sns.scatterplot(x=[X[0] for X in transformed_mat],
 #               y=[X[1] for X in transformed_mat],
#                hue=true_labels,
#                style=classification,
#               palette="deep",
 #               legend=None
 #               )
#plt.plot([x for x, _ in kmeans.centroids],
#         [y for _, y in kmeans.centroids],
#         '+',
#        markersize=10,
#        )
#plt.show()
print(error)

assert error < tol

