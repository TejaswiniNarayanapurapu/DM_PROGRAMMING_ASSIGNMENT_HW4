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
    features, _ = dataset  # Corrected variable name from 'data' to 'dataset'
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=n_clusters, init='random', random_state=random_state)  # Corrected 'clusters' to 'n_clusters'
    kmeans.fit(features_normalized)
    return kmeans.labels_



def compute():
    answers = {}

    """
    A.	Load the following 5 datasets with 100 samples each: noisy_circles (nc), noisy_moons (nm), blobs with varied variances (bvv), Anisotropicly distributed data (add), blobs (b). Use the parameters from (https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html), with any random state. (with random_state = 42). Not setting the correct random_state will prevent me from checking your results.
    """

    # Dictionary of 5 datasets. e.g., dct["nc"] = [data, labels]
    # 'nc', 'nm', 'bvv', 'add', 'b'. keys: 'nc', 'nm', 'bvv', 'add', 'b' (abbreviated datasets)

    np.random.seed(42)

    circles_with_noise = make_circles(n_samples=100, factor=0.5, noise=0.05)
    moons_with_noise = make_moons(n_samples=100, noise=0.05)
    varied_blobs = make_blobs(n_samples=100, cluster_std=[1.0, 2.5, 0.5], random_state=170)
    transformed_blobs = (np.dot(varied_blobs[0], [[0.6, -0.6], [-0.4, 0.8]]), varied_blobs[1])
    regular_blobs = make_blobs(n_samples=100, random_state=8)

    datasets = {
        'nc': circles_with_noise, 
        'nm': moons_with_noise, 
        'bvv': varied_blobs,  
        'add': transformed_blobs, 
        'b': regular_blobs
    }

# Structuring the datasets for further analysis and assignment to 'dct'
    answers["1A: datasets"] = {
         'nc': [circles_with_noise[0], circles_with_noise[1]],
         'nm': [moons_with_noise[0], moons_with_noise[1]],
         'bvv': [varied_blobs[0], varied_blobs[1]],
         'add': [transformed_blobs[0], transformed_blobs[1]],
         'b': [regular_blobs[0], regular_blobs[1]]
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

    # dct value: return a dictionary of one or more abbreviated dataset names (zero or more elements) 
    # and associated k-values with correct clusters.  key abbreviations: 'nc', 'nm', 'bvv', 'add', 'b'. 
    # The values are the list of k for which there is success. Only return datasets where the list of cluster size k is non-empty.

    # Initialize dictionaries to record clustering outcomes
    successful_clusters = {}
    failed_clusters = []

# Create a subplot grid for visualizing the clusters
    fig, grid_axes = plt.subplots(4, 5, figsize=(20, 16))

# Iterate over each dataset and their respective cluster counts
    for dataset_idx, (dataset_name, (features, labels)) in enumerate(datasets.items()):
        for row_idx, num_clusters in enumerate([2, 3, 5, 10]):
            current_ax = grid_axes[row_idx, dataset_idx]
            predicted_labels = fit_kmeans((features, labels), num_clusters)
        # Scatter plot for visualizing the clusters
            current_ax.scatter(features[:, 0], features[:, 1], c=predicted_labels, cmap='viridis', s=50, alpha=0.7)
            current_ax.set_title(f"{dataset_name}, k={num_clusters}")
            current_ax.set_xticks([])  # Remove x-axis tick marks
            current_ax.set_yticks([])  # Remove y-axis tick marks
        
        # Calculate and evaluate the silhouette score for each clustering
            avg_silhouette_score = silhouette_score(features, predicted_labels)
            if avg_silhouette_score > 0.5:
               successful_clusters.setdefault(dataset_name, []).append(num_clusters)
            else:
                if dataset_name not in failed_clusters:
                   failed_clusters.append(dataset_name)




    plt.tight_layout()
    plt.savefig("cluster_evaluation_report.pdf")

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

    # Initialize a list to track which datasets show sensitivity to initialization
    sensitivity_to_initialization = []

# Perform multiple initializations to test sensitivity across datasets
    for _ in range(5):  # Iterate a few times to ensure consistency
        for dataset_name, (data_features, _) in datasets.items():
            for cluster_count in [2, 3]:
            # Generate cluster labels with two distinct random state values
                first_trial_labels = fit_kmeans((data_features, None), cluster_count, random_state=42)
                second_trial_labels = fit_kmeans((data_features, None), cluster_count, random_state=0)
            
            # Evaluate if differing initializations lead to different clustering results
                if not np.array_equal(first_trial_labels, second_trial_labels):
                # If results vary, note the dataset as sensitive to initialization
                    if dataset_name not in sensitivity_to_initialization:
                        sensitivity_to_initialization.append(dataset_name)
                        break  # Exit the loop after finding sensitivity
        else:
            # The 'else' here is paired with 'for', executing if the loop completes normally (no breaks)
              continue  # Proceed to the next iteration of the outer loop if no sensitivity detected
        break  # If sensitivity is detected, no further checks on this dataset are needed

# This rewritten block maintains the original functionality but with clearer naming and structure.


    dct = answers["1D: datasets sensitive to initialization"] = sensitivity_to_initialization

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part1.pkl", "wb") as f:
        pickle.dump(answers, f)
