import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from numpy.random import uniform
from sklearn.datasets import make_blobs
import seaborn as sns
import random

def euclidean(point, data):
    """
    Euclidean distance between point & data.
    Point has dimensions (m,), data has dimensions (n,m), and output will be of size (n,).
    """
    return np.sqrt(np.sum((point - data)**2, axis=1))
class KMeans:
    def __init__(self, n_clusters=8, max_iter=300, tol: float = 1e-6):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.max_iter = max_iter
        self.error=0
    def fit(self, mat: np.ndarray):
        # Initialize the centroids, using the "k-means++" method, where a random datapoint is selected as the first,
        # then the rest are initialized w/ probabilities proportional to their distances to the first
        # Pick a random point from train data for first centroid
        self.centroids = [random.choice(mat)]
        for _ in range(self.n_clusters-1):
            # Calculate distances from points to the centroids
            dists = np.sum([euclidean(centroid, mat) for centroid in self.centroids], axis=0)
            # Normalize the distances
            dists /= np.sum(dists)
            # Choose remaining points based on their distances
            new_centroid_idx, = np.random.choice(range(len(mat)), size=1, p=dists)
            self.centroids += [mat[new_centroid_idx]]
        # This initial method of randomly selecting centroid starts is less effective
        # min_, max_ = np.min(mat, axis=0), np.max(mat, axis=0)
        # self.centroids = [uniform(min_, max_) for _ in range(self.n_clusters)]
        # Iterate, adjusting centroids until converged or until passed max_iter
        iteration = 0
        prev_centroids = None
        while np.not_equal(self.centroids, prev_centroids).any() and iteration < self.max_iter:
            # Sort each datapoint, assigning to nearest centroid
            sorted_points = [[] for _ in range(self.n_clusters)]
            for x in mat:
                dists = euclidean(x, self.centroids)
                centroid_idx = np.argmin(dists)
                sorted_points[centroid_idx].append(x)
            # Push current centroids to previous, reassign centroids as mean of the points belonging to them
            prev_centroids = self.centroids
            self.centroids = [np.mean(cluster, axis=0) for cluster in sorted_points]
            for i, centroid in enumerate(self.centroids):
                if np.isnan(centroid).any():  # Catch any np.nans, resulting from a centroid having no points
                    self.centroids[i] = prev_centroids[i]
                    self.error = np.sum((self.centroids[i] - prev_centroids[i]) ** 2)
                    if self.error<self.tol:
                        break
                    iteration += 1

            return self.centroids, self.error, sorted_points 
# Create a dataset of 2D distributions
centers = 5
mat, true_labels = make_blobs(n_samples=100, centers=centers, random_state=42)
mat = StandardScaler().fit_transform(mat)
# Fit centroids to dataset
kmeans = KMeans(n_clusters=centers)
kmeans.fit(mat)
# View results
#class_centers, classification = kmeans.evaluate(mat)
'''sns.scatterplot(x=[X[0] for X in mat],
                y=[X[1] for X in mat],
                hue=true_labels,
                style=classification,
                palette="deep",
                legend=None
                )'''
#plt.plot([x for x in kmeans.centroids],
#         [y for y in kmeans.centroids],
#         'k+',
#         markersize=10,
#         )
#plt.show()



""" import numpy as np
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
        distances = cdist(mat, self.centroids)
        estimated_groups = np.argmin(distances, axis=1)

        for _ in range(self.max_iter):

            new_centroids = np.array([mat[estimated_groups == j].mean(axis=0) for j in range(self.k)])

            # Check convergence
            self.error = np.sum((new_centroids - self.centroids) ** 2)
            self.estimated_groups = estimated_groups
            self.centroids = new_centroids
            distances = cdist(mat, self.centroids)
            estimated_groups = np.argmin(distances, axis=1)
            if self.error < self.tol:
                break



        




class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):
        self.tol = tol
        self.max_iter = max_iter
        self.centroids = None
        self.mat = None  # Added attribute to store input data
        self.error = None  # Initialize error attribute
    
    def fit(self, k, tol=1E-4):

        data, true_labels =make_clusters(n=1500, m=100, k=6)
        array=np.concatenate(data[:,:2], true_labels)
        index_values=[0, range(len(array))-1]
        column_values=['x-value','y-value','centroid']
        df = pd.DataFrame(data = array,  
            index = index_values,  
            columns = column_values) 
        working_df=df.copy
        self.working_df=working_df
        err = []
        goahead = True
        j = 0
        m=0

        working_df.head()
        '''
        Select k data points as centroids
        k: number of centroids
        df: pandas dataframe
        '''
        centroids = working_df.sample(k)

        '''
        Given a dataframe `dset` and a set of `centroids`, we assign each
        data point in `dset` to a centroid. 
        - dset - pandas dataframe with observations
        - centroids - pa das dataframe with centroids
        '''
        k = centroids.shape[0]
        n = working_df.shape[0]
        assignation = []
        assign_errors = []
        
        while(goahead):
            for obs in range(n):
                # Estimate error
                all_errors = np.array([])
                for centroid in range(k):
                    err = rsserr(centroids.iloc[centroid, :], working_df.iloc[obs,:])
                    all_errors = np.append(all_errors, err)

                # Get the nearest centroid and the error
                    nearest_centroid =  np.where(all_errors==np.amin(all_errors))[0].tolist()[0]
                    nearest_centroid_error = np.amin(all_errors)

                # Add values to corresponding lists
                    assignation.append(nearest_centroid)
                    assign_errors.append(nearest_centroid_error)




        #df['centroid'], df['error'] = centroid_assignation(df, centroids)
        #   df.head()
        colnames=column_values
        centroids = working_df.groupby('centroid').agg('mean').loc[:, colnames].reset_index(drop = True)
        centroids
        
        self.working_df=working_df
        
        if j>0:
            # Is the error less than a tolerance (1E-4)
            error=err[j-1]-err[j]
            self.error=error
            if err[j-1]-err[j]<=tol:
                goahead = False
            j+=1
        if m>0:
            # Is the iteration # than a tolerance
            if m>max_iter:
                goahead = False
            m=1

#utils.plot_clusters(array[:2], kmeans.estimated_groups, "test_clusters3.png")"""



""" for i, centroid in enumerate(range(centroids.shape[0])):
        err = rsserr(centroids.iloc[centroid,:], df.iloc[36,:])
        print('Error for centroid {0}: {1:.2f}'.format(i, err))










        
        
def kmeans(dset, k=2, tol=1e-4):
    '''
   K-means implementationd for a 
    `dset`:  DataFrame with observations
    `k`: number of clusters, default k=2
    `tol`: tolerance=1E-4
    '''
    # Let us work in a copy, so we don't mess the original
    working_dset = dset.copy()
    # We define some variables to hold the error, the 
    # stopping signal and a counter for the iterations
    err = []
    goahead = True
    j = 0
    
    # Step 2: Initiate clusters by defining centroids 
    centroids = initiate_centroids(k, dset)

    while(goahead):
        # Step 3 and 4 - Assign centroids and calculate error
        working_dset['centroid'], j_err = centroid_assignation(working_dset, centroids) 
        err.append(sum(j_err))
        
        # Step 5 - Update centroid position
        centroids = working_dset.groupby('centroid').agg('mean').reset_index(drop = True)

        # Step 6 - Restart the iteration
        if j>0:
            # Is the error less than a tolerance (1E-4)
            if err[j-1]-err[j]<=tol:
                goahead = False
            j+=1

    working_dset['centroid'], j_err = centroid_assignation(working_dset, centroids)
    centroids = working_dset.groupby('centroid').agg('mean').reset_index(drop = True)
    return working_dset['centroid'], j_err, centroids





np.random.seed(42)
df['centroid'], df['error'], centroids =  kmeans(df[['x','y']], 3)
df.head()



err_total = []
n = 10

df_elbow = blobs[['x','y']]"""


"""fig, ax = plt.subplots(figsize=(8, 6))
plt.scatter(df.iloc[:,0], df.iloc[:,1],  marker = 'o', 
        c=df['centroid'].astype('category'), 
        cmap = customcmap, s=80, alpha=0.5)
plt.scatter(centroids.iloc[:,0], centroids.iloc[:,1],  
        marker = 's', s=200, c=[0, 1, 2], 
        cmap = customcmap)
ax.set_xlabel(r'x', fontsize=14)
ax.set_ylabel(r'y', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()"""



"""class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):
        self.tol = tol
        self.max_iter = max_iter
        self.centroids = None
        self.mat = None  # Added attribute to store input data
        self.error = None  # Initialize error attribute
        self.working_df = None  # Initialize working_df attribute

    def fit(self, k, tol=1E-4):
        n=1500
        m=100
        k=6
        data, true_labels = make_clusters(n=1500, m=100, k=6)
        array = np.concatenate((data[:,:2], true_labels.reshape(-1, 1)), axis=1)
        index_values = list(range(n))
        column_values = ['x-value', 'y-value', 'centroid']
        df = pd.DataFrame(data=array,
                          index=index_values,
                          columns=column_values)
        working_df = df.copy()
     #   self.working_df = working_df
        centroids = working_df.sample(k)
        assignation = []  # Initialize assignation
        assign_errors=[]
        #self.data=data

        err = []
        go_ahead = True
        j = 0
        m = 0

        while go_ahead:
            for obs in range(n):
                # Estimate error
                all_errors = np.array([])
                for centroid in range(k):
                    err = rsserr(centroids.iloc[centroid, :], working_df.iloc[obs, :])
                    all_errors = np.append(all_errors, err)

                    # Get the nearest centroid and the error
                    nearest_centroid = np.where(all_errors == np.amin(all_errors))[0].tolist()[0]
                    nearest_centroid_error = np.amin(all_errors)

                    # Add values to corresponding lists
                    assignation.append(nearest_centroid)
                    assign_errors.append(nearest_centroid_error)

            # df['centroid'], df['error'] = centroid_assignation(df, centroids)
            # df.head()
            colnames = column_values
            centroids = working_df.groupby('centroid').agg('mean').loc[:, colnames].reset_index(drop=True)
            centroids

            self.working_df = working_df

            if j > 0:
                # Is the error less than a tolerance (1E-4)
                error = err[j - 1] - err[j]
                self.error = error
                if err[j - 1] - err[j] <= tol:
                    go_ahead = False
                j += 1
            if m > 0:
                # Is the iteration # than a tolerance
                if m > self.max_iter:
                    go_ahead = False
                m += 1"""