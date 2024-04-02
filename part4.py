import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering
from matplotlib.backends.backend_pdf import PdfPages
import pickle

def fit_hierarchical_cluster(data_pair, n_clusters=None, linkage='ward', distance_threshold=None):
    data_points, _ = data_pair
    scaler = StandardScaler()
    data_points_standardized = scaler.fit_transform(data_points)
    if distance_threshold is not None:
        hierarchical_cluster = AgglomerativeClustering(n_clusters=None, linkage=linkage, distance_threshold=distance_threshold)
    else:
        hierarchical_cluster = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    hierarchical_cluster.fit(data_points_standardized)
    return hierarchical_cluster.labels_

def calculate_distance_threshold(data, linkage_type='ward'):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    Z = linkage(data_scaled, method=linkage_type)
    merge_distances = np.diff(Z[:, 2])
    max_rate_change_idx = np.argmax(merge_distances)
    distance_threshold = Z[max_rate_change_idx + 1, 2] # +1 to get the next merge level's distance
    return distance_threshold

def generate_pdf_report(datasets_dict, linkage_types, pdf_filename):
    num_clusters = [2] # Part 4B: Using 2 clusters
    with PdfPages(pdf_filename) as pdf:
        for linkage_type in linkage_types:
            fig, axes = plt.subplots(1, len(datasets_dict), figsize=(20, 4))
            fig.suptitle(f'Linkage: {linkage_type}', fontsize=16)
            for j, (dataset_key, (data, _)) in enumerate(datasets_dict.items()):
                ax = axes[j]
                if '4C' in pdf_filename:
                    distance_threshold = calculate_distance_threshold(data, linkage_type)
                    labels = fit_hierarchical_cluster((data, np.zeros(data.shape[0])), linkage=linkage_type, distance_threshold=distance_threshold)
                else: # '4B' in pdf_filename
                    labels = fit_hierarchical_cluster((data, np.zeros(data.shape[0])), n_clusters=2, linkage=linkage_type)
                ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=10)
                ax.set_title(f'Dataset: {dataset_key}')
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

def compute():
    answers = {}
    seed, n_samples = 42, 100

    # Create datasets
    datasets_info = {
        "nc": datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05, random_state=seed),
        "nm": datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=seed),
        "bvv": datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=seed),
        "add": datasets.make_blobs(n_samples=n_samples, random_state=seed),  # Will transform
        "b": datasets.make_blobs(n_samples=n_samples, random_state=seed)
    }

    # Transform 'add' dataset
    X, y = datasets_info["add"]
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    datasets_info["add"] = (np.dot(X, transformation), y)

    answers["4A: datasets"] = {key: (data[0], data[1]) for key, data in datasets_info.items()}
    answers["4A: fit_hierarchical_cluster"] = fit_hierarchical_cluster
    answers["4B: cluster successes"] = list(datasets_info.keys())
    answers["4C: modified function"] = calculate_distance_threshold

    # Generate reports for 4B and 4C
    linkage_types = ['single', 'complete', 'ward', 'average']
    generate_pdf_report(answers["4A: datasets"], linkage_types, "report_4B.pdf")
    generate_pdf_report(answers["4A: datasets"], linkage_types, "report_4C.pdf")

    return answers

if __name__ == "__main__":
    answers = compute()
    with open("results_part4.pkl", "wb") as file:
        pickle.dump(answers, file)
