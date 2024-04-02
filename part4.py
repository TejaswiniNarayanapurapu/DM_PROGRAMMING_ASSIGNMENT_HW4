import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering
from matplotlib.backends.backend_pdf import PdfPages


def fit_hierarchical_cluster_linkage(dataset, n_clusters, linkage):
    data, labels = dataset
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    hierarchical_cluster = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    hierarchical_cluster.fit(data_scaled)
    return hierarchical_cluster.labels_

def calculate_distance_threshold(datasets, linkage_type):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(datasets)
    Z = linkage(data_scaled, method=linkage_type)
    merge_distances = np.diff(Z[:, 2])
    max_rate_change_idx = np.argmax(merge_distances)
    distance_threshold = Z[max_rate_change_idx, 2]
    return distance_threshold

def compute():
    answers = {}
    dct = answers["4A: datasets"] = {}
    seed = 42
    n_samples = 100

    
    nc = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05, random_state=seed)
    nm = datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=seed)
    bvv = datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=seed)
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=seed)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    add = (X_aniso, y)
    b = datasets.make_blobs(n_samples=n_samples, random_state=seed)

   
    dct['nc'] = list(nc)
    dct['nm'] = list(nm)
    dct['bvv'] = list(bvv)
    dct['add'] = list(add)
    dct['b'] = list(b)

    
    given_datasets = {"nc": nc, "nm": nm, "bvv": bvv, "add": add, "b": b}
    dataset_keys = ['nc', 'nm', 'bvv', 'add', 'b']
    linkage_types = ['single', 'complete', 'ward', 'average']
    num_clusters = [2]

   
    pdf_filename_4B = "report_4B.pdf"
    with PdfPages(pdf_filename_4B) as pdf:
        fig, axes = plt.subplots(len(linkage_types), len(dataset_keys), figsize=(20, 16))
        fig.suptitle('Scatter plots for different datasets and linkage types (2 clusters)', fontsize=16)
        for i, linkage_type in enumerate(linkage_types):
            for j, dataset_key in enumerate(dataset_keys):
                data, labels = given_datasets[dataset_key]
                for k in num_clusters:
                    predicted_labels = fit_hierarchical_cluster_linkage(given_datasets[dataset_key], n_clusters=k, linkage=linkage_type)
                    ax = axes[i, j]
                    ax.scatter(data[:, 0], data[:, 1], c=predicted_labels, cmap='viridis')
                    ax.set_title(f'{linkage_type.capitalize()} Linkage\n{dataset_key}, k={k}')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)

    answers["4B: cluster successes"] = dataset_keys

   
    pdf_filename_4C = "report_4C.pdf"
    with PdfPages(pdf_filename_4C) as pdf:
        fig, axes = plt.subplots(len(linkage_types), len(dataset_keys), figsize=(20, 16))
        fig.suptitle('Scatter plots for different datasets and linkage types', fontsize=16)
        for i, linkage_type in enumerate(linkage_types):
            for j, dataset_key in enumerate(dataset_keys):
                data, labels = given_datasets[dataset_key]
                distance_threshold = calculate_distance_threshold(data, linkage_type)
                predicted_labels = fit_hierarchical_cluster_linkage(given_datasets[dataset_key], n_clusters=None, linkage=linkage_type, distance_threshold=distance_threshold)
                ax = axes[i, j]
                ax.scatter(data[:, 0], data[:, 1], c=predicted_labels, cmap='viridis')
                ax.set_title(f'{linkage_type.capitalize()} Linkage\n{dataset_key}')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)

    
    answers["4C: modified function"] = calculate_distance_threshold

    return answers

if __name__ == "__main__":
    answers = compute()
    with open("part4.pkl", "wb") as f:
        pickle.dump(answers, f)
