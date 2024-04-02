import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets as skdatasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib.backends.backend_pdf import PdfPages
import pickle

# Adjusted the import statement for datasets to avoid naming conflict

# Hierarchical Clustering Function
def fit_hierarchical_cluster(dataset, n_clusters, linkage_type='ward'):
    data, _ = dataset
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_type)
    model.fit(data_scaled)
    return model.labels_

# Modified Hierarchical Clustering with Cut-off Distance
def fit_modified(dataset, distance_threshold, linkage_method='ward'):
    data, _ = dataset
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    linkage_matrix = linkage(data_scaled, method=linkage_method)
    labels = fcluster(linkage_matrix, distance_threshold, criterion='distance')
    return labels

# Compute Cut-off Distance
def compute_distance_threshold(data, linkage_method='ward'):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    linkage_matrix = linkage(data_scaled, method=linkage_method)
    distances = linkage_matrix[:, 2]
    distance_diff = np.diff(distances)
    max_diff_index = np.argmax(distance_diff)
    cutoff_distance = (distances[max_diff_index] + distances[max_diff_index + 1]) / 2
    return cutoff_distance

# Generate Datasets
def generate_datasets():
    seed = 42
    n_samples = 100
    datasets_dict = {
        'Circles': skdatasets.make_circles(n_samples=n_samples, factor=.5, noise=.05, random_state=seed),
        'Moons': skdatasets.make_moons(n_samples=n_samples, noise=.05, random_state=seed),
        'Varied': skdatasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=seed),
        'Anisotropic': (np.dot(skdatasets.make_blobs(n_samples=n_samples, random_state=seed)[0], [[0.6, -0.6], [-0.4, 0.8]]), skdatasets.make_blobs(n_samples=n_samples, random_state=seed)[1]),
        'Blobs': skdatasets.make_blobs(n_samples=n_samples, random_state=seed)
    }
    return datasets_dict

# Plot Clusters
def plot_clusters(datasets, pdf_filename, is_modified=False, linkage_types=['ward', 'complete', 'average', 'single']):
    with PdfPages(pdf_filename) as pdf:
        for name, dataset in datasets.items():
            fig, axes = plt.subplots(1, len(linkage_types), figsize=(20, 4), squeeze=False)
            fig.suptitle(f'{name} dataset', fontsize=16)
            for i, linkage_type in enumerate(linkage_types):
                if is_modified:
                    distance_threshold = compute_distance_threshold(dataset[0], linkage_type)
                    labels = fit_modified(dataset, distance_threshold, linkage_type)
                else:
                    labels = fit_hierarchical_cluster(dataset, n_clusters=2, linkage=linkage_type)
                axes[0, i].scatter(dataset[0][:, 0], dataset[0][:, 1], c=labels, cmap='viridis', edgecolor='k')
                axes[0, i].set_title(f'Linkage: {linkage_type}')
            pdf.savefig(fig)
            plt.close(fig)



if __name__ == "__main__":
    answers = compute()

    with open("part4.pkl", "wb") as f:
        pickle.dump(answers, f)
