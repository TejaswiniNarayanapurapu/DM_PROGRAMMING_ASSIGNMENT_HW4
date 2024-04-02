import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering
import pickle
from matplotlib.backends.backend_pdf import PdfPages

# Assuming utils.py is a custom module you've created for this project
# import utils as u

def fit_hierarchical_cluster(data_pair, number_of_clusters, method='ward'):
    data_points, categories = data_pair
    standardizer = StandardScaler()
    data_points_standardized = standardizer.fit_transform(data_points)
    agglomerative = AgglomerativeClustering(n_clusters=number_of_clusters, linkage=method)
    agglomerative.fit(data_points_standardized)
    return agglomerative.labels_

def fit_modified(data_pair, threshold_distance, method_linkage):
    features, targets = data_pair
    normalizer = StandardScaler()
    normalized_features = normalizer.fit_transform(features)
    clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold_distance, linkage=method_linkage)
    clustering_model.fit(normalized_features)
    return clustering_model.labels_

def determine_optimal_threshold(input_data, method_linkage):
    normalizer = StandardScaler()
    normalized_data = normalizer.fit_transform(input_data)
    linkage_matrix = linkage(normalized_data, method=method_linkage)
    distances_between_merges = np.diff(linkage_matrix[:, 2])
    significant_change_index = np.argmax(distances_between_merges)
    optimal_threshold = linkage_matrix[significant_change_index, 2]
    return optimal_threshold

def compute():
    results = {}
    datasets_dict = results["4A: datasets"] = {}
    
    sample_size = 100
    random_seed = 42
    
    datasets_dict['nc'] = list(datasets.make_circles(n_samples=sample_size, factor=0.5, noise=0.05, random_state=random_seed))
    datasets_dict['nm'] = list(datasets.make_moons(n_samples=sample_size, noise=0.05, random_state=random_seed))
    datasets_dict['bvv'] = list(datasets.make_blobs(n_samples=sample_size, cluster_std=[1.0, 2.5, 0.5], random_state=random_seed))
    
    X_blob, y_blob = datasets.make_blobs(n_samples=sample_size, random_state=random_seed)
    transformation_matrix = [[0.6, -0.6], [-0.4, 0.8]]
    X_transformed = np.dot(X_blob, transformation_matrix)
    datasets_dict['add'] = [X_transformed, y_blob]
    
    datasets_dict['b'] = list(datasets.make_blobs(n_samples=sample_size, random_state=random_seed))
    
    results["4A: fit_hierarchical_cluster"] = fit_hierarchical_cluster

    # Proceed with evaluation
    dataset_identifiers = ['nc', 'nm', 'bvv', 'add', 'b']
    linkage_options = ['single', 'complete', 'ward', 'average']
    report_filename = "report_4B.pdf"
    report_pages = []
    optimal_thresholds = {}

    fig, axes = plt.subplots(len(linkage_options), len(dataset_identifiers), figsize=(20, 16))
    fig.suptitle('Scatter plots for different datasets and linkage types', fontsize=16)
    
    for i, linkage_type in enumerate(linkage_options):
        for j, dataset_key in enumerate(dataset_identifiers):
            data, labels = datasets_dict[dataset_key]
            
            distance_threshold = determine_optimal_threshold(data, linkage_type)
            optimal_thresholds[(dataset_key, linkage_type)] = distance_threshold
            
            predicted_labels = fit_modified((data, labels), distance_threshold, linkage_type)
    
            ax = axes[i, j]
            ax.scatter(data[:, 0], data[:, 1], c=predicted_labels, cmap='viridis')
            ax.set_title(f'{linkage_type.capitalize()} Linkage\n{dataset_key}')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    report_pages.append(fig)
    plt.close(fig)
    
    with PdfPages(report_filename) as pdf:
        for page in report_pages:
            pdf.savefig(page)
    
    results["4C: clustering function"] = fit_modified

    return results

if __name__ == "__main__":
    answers = compute()
    with open("part4.pkl", "wb") as f:
        pickle.dump(answers, f)
