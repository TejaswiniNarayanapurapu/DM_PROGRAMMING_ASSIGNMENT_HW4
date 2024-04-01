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

def fit_kmeans():
    X, _ = data  # Unpack the data; labels are not used in k-means clustering.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # Standardize the features
    kmeans = cluster.KMeans(n_clusters=n_clusters, init='random', random_state=42)
    kmeans.fit(X_scaled)
    return kmeans.labels_

def compute():
    answers = {}
    
    # Load the datasets
    noisy_circles = datasets.make_circles(n_samples=100, factor=.5, noise=.05, random_state=42)
    noisy_moons = datasets.make_moons(n_samples=100, noise=.05, random_state=42)
    blobs = datasets.make_blobs(n_samples=100, random_state=8)
    varied = datasets.make_blobs(n_samples=100, cluster_std=[1.0, 2.5, 0.5], random_state=42)
    random_state = 170
    X, y = datasets.make_blobs(n_samples=100, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)

    """
    A.	Load the following 5 datasets with 100 samples each: noisy_circles (nc), noisy_moons (nm), blobs with varied variances (bvv), Anisotropicly distributed data (add), blobs (b). Use the parameters from (https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html), with any random state. (with random_state = 42). Not setting the correct random_state will prevent me from checking your results.
    """

    # Dictionary of 5 datasets. e.g., dct["nc"] = [data, labels]
    # 'nc', 'nm', 'bvv', 'add', 'b'. keys: 'nc', 'nm', 'bvv', 'add', 'b' (abbreviated datasets)
    dct = answers["1A: datasets"] = {
        "nc": noisy_circles,
        "nm": noisy_moons,
        "b": blobs,
        "bvv": varied,
        "add": aniso
    }

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
    plt.figure(figsize=(20, 16))
    plot_num = 1
    
    for dataset_label, dataset in datasets_list.items():
        for n_clusters in [2, 3, 5, 10]:
            plt.subplot(4, 5, plot_num)
            labels = fit_kmeans(dataset, n_clusters)
            plt.scatter(dataset[0][:, 0], dataset[0][:, 1], c=labels, cmap='viridis')
            plt.title(f"{dataset_label} (k={n_clusters})")
            plot_num += 1
    
    plt.tight_layout()
    plt.savefig("kmeans_clusters.pdf")
    answers["1C: Scatter Plots"] = "Generated and saved to kmeans_clusters.pdf"

    # dct value: return a dictionary of one or more abbreviated dataset names (zero or more elements) 
    # and associated k-values with correct clusters.  key abbreviations: 'nc', 'nm', 'bvv', 'add', 'b'. 
    # The values are the list of k for which there is success. Only return datasets where the list of cluster size k is non-empty.
    dct = answers["1C: cluster successes"] = {"xy": [3,4], "zx": [2]} 

    # dct value: return a list of 0 or more dataset abbreviations (list has zero or more elements, 
    # which are abbreviated dataset names as strings)
    dct = answers["1C: cluster failures"] = ["xy"]

    """
    D. Repeat 1.C a few times and comment on which (if any) datasets seem to be sensitive to the choice of initialization for the k=2,3 cases. You do not need to add the additional plots to your report.

    Create a pdf of the plots and return in your report. 
    """
    sensitivity = {}
    for dataset_label, dataset in datasets_list.items():
        labels_list = []
        for seed in range(42, 47):  # Using five different seeds for initialization
            kmeans = KMeans(n_clusters=3, init='random', random_state=seed)
            labels = kmeans.fit_predict(StandardScaler().fit_transform(dataset[0]))
            labels_list.append(labels)
        # Check if all elements are the same in labels_list (i.e., no sensitivity)
        sensitivity[dataset_label] = all(np.array_equal(labels_list[0], l) for l in labels_list[1:])
    
    answers["1D: Sensitivity Analysis"] = sensitivity

    return answers

    # dct value: list of dataset abbreviations
    # Look at your plots, and return your answers.
    # The plot is part of your report, a pdf file name "report.pdf", in your repository.
    dct = answers["1D: datasets sensitive to initialization"] = [""]

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    start_time = time.time()
    answers = compute()
    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")
    
    # Saving the computed answers to a file
    with open("assignment2_answers.pkl", "wb") as f:
        pickle.dump(answers, f)
