import pickle

import numpy as np
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

import datasets


def get_all_scores(dataset, centers):

    data, y = dataset.data, dataset.y

    assignment = get_labels(data, centers)

    scores = [
        j_score(data, centers),
        silh_score(data, assignment),
        accuracy(assignment, y),
        norm_kmeans_dist(dataset.name, centers),
        len(centers),
    ]

    return scores


def j_score(data, centers):
    inertia = 0

    pred_labels = get_labels(data, centers)

    for i, center in enumerate(centers):
        inertia += ((data[pred_labels == i] - center) ** 2).sum()

    return inertia


def silh_score(data, assignment):
    if len(np.unique(assignment)) > 1:
        try:
            silh_score = silhouette_score(data, assignment, sample_size=10000, n_jobs=2)
        except:
            silh_score = (
                -1.0
            )  # sometimes, #uniqe_labels > 1 but not for the sampled set
        return silh_score
    else:
        return -1


def accuracy(preds, true_labels):
    predicted_labels = np.zeros_like(true_labels)

    for label in np.unique(preds):
        indices = preds == label

        counts = np.bincount(
            true_labels[indices], minlength=len(np.unique(true_labels))
        )

        cluster_label = np.argmax(counts)

        predicted_labels[indices] = cluster_label

    return (predicted_labels == true_labels).sum() / len(true_labels)


def norm_kmeans_dist(dataset_name, centers):

    distance_avg = 0

    with open(f"./data/opt/{dataset_name}_opt.pkl", "rb") as f_:
        opt_clusterings = pickle.load(f_)

    for opt_clustering in opt_clusterings:
        dists = cdist(centers, opt_clustering, metric="euclidean")
        distance = np.min(dists, axis=1).sum() / centers.shape[0]
        distance_avg += distance

    distance_avg /= len(opt_clusterings)

    dataset = datasets.by_name(dataset_name)

    distance_avg /= dataset.radius * 2

    return distance_avg


def get_labels(data, centers):
    dists = cdist(data, centers, metric="euclidean")

    return np.argmin(dists, axis=1)
