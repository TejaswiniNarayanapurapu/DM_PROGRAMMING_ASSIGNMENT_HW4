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


from sklearn.datasets import make_circles, make_moons, make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def fit_kmeans(dataset, n_clusters, random_state=42):
    X, y = dataset
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit KMeans estimator
    kmeans = KMeans(n_clusters=n_clusters, init='random', random_state=random_state)
    kmeans.fit(X_scaled)
    
    return kmeans.labels_
    #return None


def compute():
    answers = {}

    """
    A.	Load the following 5 datasets with 100 samples each: noisy_circles (nc), noisy_moons (nm), blobs with varied variances (bvv), Anisotropicly distributed data (add), blobs (b). Use the parameters from (https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html), with any random state. (with random_state = 42). Not setting the correct random_state will prevent me from checking your results.
    """

    # Dictionary of 5 datasets. e.g., dct["nc"] = [data, labels]
    # 'nc', 'nm', 'bvv', 'add', 'b'. keys: 'nc', 'nm', 'bvv', 'add', 'b' (abbreviated datasets)

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

    for i, (key, (X, y)) in enumerate(datasets.items()):
        for j, k in enumerate([2, 3, 5, 10]):
            ax = axes[j, i]
            labels = fit_kmeans((X, y), k)
            ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
            ax.set_title(f"{key}, k={k}")
            ax.set_xticks(())
            ax.set_yticks(())
            
            silhouette_avg = silhouette_score(X, labels)
            if silhouette_avg > 0.5:
                if key not in cluster_successes:
                    cluster_successes[key] = []
                cluster_successes[key].append(k)
            else:
                cluster_failures.append(key)

    plt.tight_layout()
    plt.savefig("report.pdf")

    dct = answers["1C: cluster successes"] = {"bvv": [3], "add": [3],"b":[3]} 

    # dct value: return a list of 0 or more dataset abbreviations (list has zero or more elements, 
    # which are abbreviated dataset names as strings)
    dct = answers["1C: cluster failures"] = ["nc","nm"]

    """
    D. Repeat 1.C a few times and comment on which (if any) datasets seem to be sensitive to the choice of initialization for the k=2,3 cases. You do not need to add the additional plots to your report.

    Create a pdf of the plots and return in your report. 
    """

    # dct value: list of dataset abbreviations
    # Look at your plots, and return your answers.
    # The plot is part of your report, a pdf file name "report.pdf", in your repository.

    dataset_sensitivity = []

    for _ in range(5):  # Repeat 1.C a few times
        for key, (X, _) in datasets.items():
            for k in [2, 3]:
                labels_init_1 = fit_kmeans((X, None), k, random_state=42)
                labels_init_2 = fit_kmeans((X, None), k, random_state=0)
                # Check if the labels are different for different initializations
                if not np.array_equal(labels_init_1, labels_init_2):
                    dataset_sensitivity.append(key)
                    break  # Move to the next dataset
            else:
                continue
            break

    dct = answers["1D: datasets sensitive to initialization"] = dataset_sensitivity

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part1.pkl", "wb") as f:
        pickle.dump(answers, f)
