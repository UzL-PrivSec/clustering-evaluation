from sklearn.cluster import KMeans
import numpy as np

import metrics

from diffprivlib.models import KMeans as DPLLyodImpr
from diffprivlib.mechanisms import GaussianAnalytic, Laplace

from clustering_algorithms.lshsplits.learning.clustering import (
    clustering_algorithm as lshsplits_repo,
)

from clustering_algorithms.emmc.emmc import EMMC
from clustering_algorithms.dpm.dpm import DPM


def kmeans(dataset, k, with_scores=True, **unused_kwargs):

    algo = KMeans(
        n_clusters=k,
        n_init=1,
        max_iter=300,
        random_state=None,
        verbose=0,
    )

    _ = algo.fit(dataset.data, dataset.y)

    if with_scores:
        return metrics.get_all_scores(dataset, algo.cluster_centers_)
    else:
        return algo.cluster_centers_


def dplloyd(dataset, k, eps, with_scores=True, **unused_kwargs):
    algo = DPLLyodImpr(n_clusters=k, bounds=dataset.bounds, epsilon=eps)

    _ = algo.fit(dataset.data)

    if with_scores:
        return metrics.get_all_scores(dataset, algo.cluster_centers_)
    else:
        return algo.cluster_centers_


def emmc(dataset, k, eps, delta, with_scores=True, **unused_kwargs):
    algo = EMMC()

    centers = algo.cluster(
        data=dataset.data,
        diameter=dataset.radius * 2,
        k=k,
        epsilon=eps,
        delta=delta,
    )

    if with_scores:
        return metrics.get_all_scores(dataset, centers)
    else:
        return centers


def lshsplits(dataset, k, eps, delta, with_scores=True, **unused_kwargs):

    average_prop = 0.1
    clustering_prop = 1 - average_prop

    eps_average, delta_average = average_prop * eps, average_prop * delta
    eps_clustering, delta_clustering = clustering_prop * eps, clustering_prop * delta

    noisy_count = Laplace(epsilon=eps_average * 0.2, sensitivity=1).randomise(
        dataset.num_points
    )

    gaussian_mech = GaussianAnalytic(
        epsilon=eps_average * 0.8, delta=delta_average, sensitivity=dataset.max_norm
    )
    noisy_sum = np.array(
        [gaussian_mech.randomise(sum_dim) for sum_dim in np.sum(dataset.data, axis=0)]
    )

    priv_avg = noisy_sum / noisy_count

    data = dataset.data - priv_avg

    lw_bound = np.min(np.repeat(dataset.bounds[0], data.shape[1]) - priv_avg)
    up_bound = np.max(np.repeat(dataset.bounds[1], data.shape[1]) - priv_avg)
    cap = up_bound if np.abs(up_bound) > np.abs(lw_bound) else -lw_bound
    max_norm = cap * np.sqrt(data.shape[1])

    data_class = lshsplits_repo.clustering_params.Data(data, max_norm)

    dp_params = lshsplits_repo.clustering_params.DifferentialPrivacyParam(
        eps_clustering,
        delta_clustering,
    )

    result = lshsplits_repo.private_lsh_clustering(k, data_class, dp_params)

    centers = result.centers + priv_avg

    if with_scores:
        return metrics.get_all_scores(dataset, centers)
    else:
        return centers


def dpm(
    dataset,
    eps,
    delta,
    with_scores=True,
    **kwargs,
):

    dpm = DPM(
        data=dataset.data.copy(),
        bounds=dataset.bounds,
        epsilon=eps,
        delta=delta,
        **kwargs,
    )

    centers, _ = dpm.perform_clustering()

    if with_scores:
        return metrics.get_all_scores(dataset, centers)
    else:
        return centers
