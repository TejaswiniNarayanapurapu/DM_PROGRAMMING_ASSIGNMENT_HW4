import myplots as myplt
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from sklearn.datasets import make_blobs
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
import scipy.io as io
from scipy.cluster.hierarchy import dendrogram, linkage  #
from sklearn.datasets import make_circles, make_moons, make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score



# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u


# ----------------------------------------------------------------------
"""
Part 1: 
Evaluation of k-Means over Diverse Datasets: 
In the first task, you will explore how k-Means perform on datasets with diverse structure.
"""

# Fill this function with code at this location. Do NOT move it. 
# Change the arguments and return according to 
# the question asked. 

def fit_kmeans(data, clusters, random_state=42):
    features, _ = data
    features_normalized = normalizer.fit_transform(features)
    means_model = KMeans(n_clusters=clusters, init='random', random_state=random_state)
    means_model.fit(features_normalized)
    
    return means_model.labels_



def compute():
    answers = {}
    np.random.seed(42)

    noisy_circles = make_circles(n_samples=100, factor=0.5, noise=0.05)
    noisy_moons = make_moons(n_samples=100, noise=0.05)
    blobs_varied = make_blobs(n_samples=100, cluster_std=[1.0, 2.5, 0.5], random_state=170)
    aniso = (np.dot(blobs_varied[0], [[0.6, -0.6], [-0.4, 0.8]]), blobs_varied[1])
    blobs = make_blobs(n_samples=100, random_state=8)
    datasets = {'nc': noisy_circles, 'nm': noisy_moons, 'bvv': blobs_varied, 'add': aniso, 'b': blobs}

    dct = answers["1A: datasets"] = {'nc': [noisy_circles[0],noisy_circles[1]],
                                     'nm': [noisy_moons[0],noisy_moons[1]],
                                     'bvv': [blobs_varied[0],blobs_varied[1]],
                                     'add': [aniso[0],aniso[1]],
                                     'b': [blobs[0],blobs[1]]}

    """
    A.	Load the following 5 datasets with 100 samples each: noisy_circles (nc), noisy_moons (nm), blobs with varied variances (bvv), Anisotropicly distributed data (add), blobs (b). Use the parameters from (https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html), with any random state. (with random_state = 42). Not setting the correct random_state will prevent me from checking your results.
    """

    # Dictionary of 5 datasets. e.g., dct["nc"] = [data, labels]
    # 'nc', 'nm', 'bvv', 'add', 'b'. keys: 'nc', 'nm', 'bvv', 'add', 'b' (abbreviated datasets)
   

    """
   B. Write a function called fit_kmeans that takes dataset (before any processing on it), i.e., pair of (data, label) Numpy arrays, and the number of clusters as arguments, and returns the predicted labels from k-means clustering. Use the init='random' argument and make sure to standardize the data (see StandardScaler transform), prior to fitting the KMeans estimator. This is the function you will use in the following questions. 
    """

    # dct value:  the `fit_kmeans` function
    dct = answers["1B: fit_kmeans"] = fit_kmeans
    result = dct


    """
    C.	Make a big figure (4 rows x 5 columns) of scatter plots (where points are colored by predicted label) with each column corresponding to the datasets generated in part 1.A, and each row being k=[2,3,5,10] different number of clusters. For which datasets does k-means seem to produce correct clusters for (assuming the right number of k is specified) and for which datasets does k-means fail for all values of k? 
    
    Create a pdf of the plots and return in your report. 
    """

    # dct value: return a dictionary of one or more abbreviated dataset names (zero or more elements) 
    # and associated k-values with correct clusters.  key abbreviations: 'nc', 'nm', 'bvv', 'add', 'b'. 
    # The values are the list of k for which there is success. Only return datasets where the list of cluster size k is non-empty.
    cluster_successes = {}
    cluster_failures = []
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    for i, (dataset_id, (X, y)) in enumerate(datasets.items()):
        for j, cluster_size in enumerate([2, 3, 5, 10]):
            axis = axes[j, i]
            cluster_labels = fit_kmeans((X, y), cluster_size)
            axis.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='viridis', s=50, alpha=0.7)
            axis.set_title(f"{dataset_id}, k={cluster_size}")
            axis.set_xticks(())
            axis.set_yticks(())
        
           # Evaluate clustering performance with silhouette score
            silhouette_avg = silhouette_score(X, cluster_labels)
            if silhouette_avg > 0.5:
                if dataset_id not in cluster_successes:
                    cluster_successes[dataset_id] = []
                    cluster_successes[dataset_id].append(cluster_size)
            else:
                cluster_failures.append(dataset_id)
    plt.tight_layout()
    plt.savefig("clustering_analysis_report.pdf")

# These lines remain unchanged as per your request
    dct = answers["1C: cluster successes"] = {"xy": [3,4], "zx": [2]} 
    dct = answers["1C: cluster failures"] = ["xy"]

    """
    D. Repeat 1.C a few times and comment on which (if any) datasets seem to be sensitive to the choice of initialization for the k=2,3 cases. You do not need to add the additional plots to your report.

    Create a pdf of the plots and return in your report. 
    """

    initialization_sensitivity = []


    for _ in range(5):  # Iterating a set number of time
        for dataset_key, (data_points, _) in datasets.items():
            for num_clusters in [2, 3]:
            
              labels_first_try = fit_kmeans((data_points, None), num_clusters, random_state=42)
              labels_second_try = fit_kmeans((data_points, None), num_clusters, random_state=0)
            
            
              if not np.array_equal(labels_first_try, labels_second_try):
                  initialization_sensitivity.append(dataset_key)
                  break  
              else:
            
                continue
        
              break


    answers["1D: datasets sensitive to initialization"] = initialization_sensitivity

    return answers



# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part1.pkl", "wb") as f:
        pickle.dump(answers, f)
