import numpy as np
from scipy.spatial.distance import cdist

import numpy as np
from scipy.spatial.distance import cdist

"""class Silhouette:
   def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):
       self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        silhouette_scores = np.zeros(X.shape[0])

        for i in range(X.shape[0]):
            #make number of observations in the cluster -1
            a_o = np.zeros(self.k)
            #make number of observations in the next closest cluster -1
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
            # Calculate silhouette score for point o
                    s_o = (np.min(b_o[y[i]]) - a_o[y[i]]) / max(np.min(b_o[y[i]]), a_o[y[i]])
                    silhouette_scores[i] = s_o

        return silhouette_scores"""
    
    
    
class Silhouette:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def score(self, n=1500, m=1, k=6): 
        import sklearn.metrics as metrics
        from sklearn.cluster import KMeans
        import pandas as pd
        import numpy as np
        import seaborn as sns
        import matplotlib.pyplot as plt
        import sys
        import os
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        sys.path.insert(0, project_root)
        from cluster.utils import make_clusters, plot_clusters, plot_multipanel


        wss = []
        data, true_labels = make_clusters(n=1500, m=1, k=6)
        """array = np.concatenate(data[0],data[1])
        index_values = list(range(n))
        column_values = ['x-value', 'y-value']
        df = pd.DataFrame(data=array,
                index=index_values,
                columns=column_values)"""
        """for k in K:
            kmeans=KMeans(n_clusters=k,init="k-means++")
            kmeans=kmeans.fit(data)
            wss_iter = kmeans.inertia_
            wss.append(wss_iter)"""

        SK = range(3,13)
        sil_score = []
        for i in SK:
            labels=KMeans(n_clusters=i,init="k-means++",random_state=45).fit(data).labels_
            score = metrics.silhouette_score(data,labels,metric="euclidean",sample_size=1500,random_state=45)
            sil_score.append(score)
            print ("Silhouette score for k(clusters) = "+str(i)+" is "
                +str(metrics.silhouette_score(data,labels,metric="euclidean",sample_size=1500,random_state=45)))
            return metrics.silhouette_score