import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering
from matplotlib.backends.backend_pdf import PdfPages
import pickle

def fit_hierarchical_cluster(data_pair, n_clusters, linkage='ward'):
    data_points, _ = data_pair
    scaler = StandardScaler()
    data_points_standardized = scaler.fit_transform(data_points)
    hierarchical_cluster = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    hierarchical_cluster.fit(data_points_standardized)
    return hierarchical_cluster.labels_

def calculate_distance_threshold(data, linkage_type):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    Z = linkage(data_scaled, method=linkage_type)
    merge_distances = np.diff(Z[:, 2])
    max_rate_change_idx = np.argmax(merge_distances)
    distance_threshold = Z[max_rate_change_idx, 2]
    return distance_threshold

def generate_plots_for_datasets(given_datasets, linkage_types, num_clusters, plot_title, filename):
    fig, axes = plt.subplots(len(linkage_types), len(given_datasets.keys()), figsize=(20, 16))
    fig.suptitle(plot_title, fontsize=16)

    for i, linkage_type in enumerate(linkage_types):
        for j, (dataset_key, (data, labels)) in enumerate(given_datasets.items()):
            for k in num_clusters:
                if plot_title == "Report 4B":
                    predicted_labels = fit_hierarchical_cluster((data, labels), n_clusters=k, linkage=linkage_type)
                else: # For "Report 4C"
                    threshold = calculate_distance_threshold(data, linkage_type)
                    predicted_labels = fit_hierarchical_cluster((data, labels), n_clusters=None, linkage=linkage_type, distance_threshold=threshold)
                ax = axes[i, j]
                ax.scatter(data[:, 0], data[:, 1], c=predicted_labels, cmap='viridis')
                ax.set_title(f'{linkage_type.capitalize()} Linkage\n{dataset_key}, k={k}')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    with PdfPages(filename) as pdf:
        pdf.savefig(fig)
    plt.close(fig)

def compute():
    answers = {}
    seed, n_samples = 42, 100
    datasets_info = {
        "nc": datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05, random_state=seed),
        "nm": datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=seed),
        "bvv": datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=seed),
        "add": datasets.make_blobs(n_samples=n_samples, random_state=seed),
        "b": datasets.make_blobs(n_samples=n_samples, random_state=seed)
    }
    # Applying transformation to 'add'
    X, y = datasets_info["add"]
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    datasets_info["add"] = (np.dot(X, transformation), y)

    answers["4A: datasets"] = {k: (v[0], v[1]) for k, v in datasets_info.items()}
    answers["4A: fit_hierarchical_cluster"] = fit_hierarchical_cluster
    answers["4B: cluster successes"] = list(datasets_info.keys())
    answers["4C: modified function"] = calculate_distance_threshold

    # Generate and save plots for Part 4B
    linkage_types = ['single', 'complete', 'ward', 'average']
    generate_plots_for_datasets(answers["4A: datasets"], linkage_types, [2], "Report 4B", "report_4B.pdf")

    # Assuming dynamic calculation for Report 4C similar to 4B but using calculated thresholds
    generate_plots_for_datasets(answers["4A: datasets"], linkage_types, [None], "Report 4C", "report_4C.pdf")

    return answers

if __name__ == "__main__":
    answers = compute()
    with open("results_part4.pkl", "wb") as file:
        pickle.dump(answers, file)
