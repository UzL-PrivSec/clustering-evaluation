import os
import argparse
import time
from datetime import datetime
import itertools

import pandas as pd
import numpy as np
from tqdm import tqdm

import algorithms
import datasets
import hypersettings


def SAVE(data_frame, type_, tag, algo, dataset_name):
    base_path = "./data/exps/"

    # current date as string
    today = datetime.today().strftime("%Y-%m-%d")
    tag = MAIN_TAG + "_" + tag + "_" + today
    path = os.path.join(base_path, type_, tag)

    os.makedirs(path, exist_ok=True)

    path = os.path.join(path, algo + "_" + dataset_name + ".pkl")

    data_frame.to_pickle(path)


def POSTPROCESS(
    exp_type,
    tag,
    dt_name,
    algo_name,
    algo,
    iterations,
    k_target,
    iter_param,
    num_data_args,
    base_frame,
    save,
    iter_out_param=None,
):
    if iter_out_param is None:
        iter_out_param = iter_param

    result_data = iterate_repeat_and_pack(
        algo,
        iterations,
        iter_param,
        num_data_args=num_data_args,
    )

    print(result_data)

    result_frame = results_to_pandas(base_frame, k_target, iter_out_param, result_data)

    print(result_frame)

    if save:
        SAVE(result_frame, exp_type, tag, algo_name, dt_name)


def iterate_repeat_and_pack(algo, iterations, *iter_param_list, num_data_args):
    assert (
        len(set(map(len, iter_param_list))) == 1
    ), "param lists have to be of equal length"

    result_data = []

    for param_tuple in tqdm(zip(*iter_param_list)):

        result_data.append([])

        if num_data_args == 0:
            func_tup = lambda: (algo(*param_tuple)(),)
        else:
            func_tup = algo(*param_tuple)

        for _ in range(iterations):
            result_data[-1].append(func_tup())

    return result_data


def results_to_pandas(base_frame, k_targets, hyper_param_list, data):

    if np.array(k_targets).shape == ():  # single value
        k_targets = itertools.repeat(itertools.repeat(k_targets))
    else:
        k_targets = np.repeat(k_targets, repeats=len(data[0])).reshape(
            (len(k_targets), len(data[0]))
        )

    for k_target, hyper_param, hp_data in zip(k_targets, hyper_param_list, data):
        hyper_param = np.atleast_1d(hyper_param)
        for iteration, (kt, iter_hp_data) in enumerate(zip(k_target, hp_data)):
            print(iteration, kt, hyper_param, iter_hp_data)
            base_frame.loc[len(base_frame)] = [
                iteration,
                kt,
                *hyper_param,
            ] + list(iter_hp_data)

    return base_frame


EXPERIMENTS_BY_NAME = {}


def register(class_):
    class_name = class_.__name__
    if class_name in EXPERIMENTS_BY_NAME:
        raise ValueError(f"Duplicate experiment name: {class_name}")
    EXPERIMENTS_BY_NAME[class_name] = class_
    return class_


@register
class KOpt:
    def __init__(self, algo_name):
        self.num_data_args = 7
        self.algo_name = algo_name
        self.algo_func = getattr(algorithms, algo_name)

    base_frame = pd.DataFrame(
        [],
        columns=[
            "iteration",
            "k_target",
            "_",
            "inertia",
            "silh_score",
            "accuracy",
            "norm_kmeans_dist",
            "k_result",
        ],
    )

    def run(self):

        def algo(_):
            return lambda: self.algo_func(
                dataset, eps=setting.eps, delta=setting.delta, k=dataset.num_clusters
            )

        POSTPROCESS(
            "kopt",
            setting.tag,
            dataset.name,
            self.algo_name,
            algo,
            setting.num_iterations,
            dataset.num_clusters,
            "?",
            self.num_data_args,
            self.base_frame,
            save=setting.save,
        )


@register
class EpsDist:
    def __init__(self, algo_name):
        self.num_data_args = 7

        if algo_name != "dpm":
            raise ValueError("Only DPM is supported for Experiment 'EpsDist'")

        self.algo_name = algo_name
        self.algo_func = getattr(algorithms, algo_name)

    base_frame = pd.DataFrame(
        [],
        columns=[
            "iteration",
            "k_target",
            "eps_bandwidth",
            "eps_split",
            "eps_count",
            "eps_average",
            "inertia",
            "silh_score",
            "accuracy",
            "norm_kmeans_dist",
            "k_result",
        ],
    )

    def run(self):

        def algo(eps_dist):
            return lambda: self.algo_func(
                dataset,
                eps=setting.eps,
                delta=setting.delta,
                eps_dist=eps_dist,
            )

        POSTPROCESS(
            "epsDist",
            setting.tag,
            dataset.name,
            self.algo_name,
            algo,
            setting.num_iterations,
            "?",
            setting.eps_dist_range,
            self.num_data_args,
            self.base_frame,
            save=setting.save,
        )


@register
class Centreness:
    def __init__(self, algo_name):
        self.num_data_args = 7

        if algo_name != "dpm":
            raise ValueError("Only DPM is supported for Experiment 'Centreness'")

        self.algo_name = algo_name
        self.algo_func = getattr(algorithms, algo_name)

    base_frame = pd.DataFrame(
        [],
        columns=[
            "iteration",
            "k_target",
            "t",
            "q",
            "inertia",
            "silh_score",
            "accuracy",
            "norm_kmeans_dist",
            "k_result",
        ],
    )

    def run(self):

        def algo(tq):
            return lambda: self.algo_func(
                dataset,
                eps=setting.eps,
                delta=setting.delta,
                cent_params=tq,
            )

        POSTPROCESS(
            "centreness",
            setting.tag,
            dataset.name,
            self.algo_name,
            algo,
            setting.num_iterations,
            "?",
            setting.tq_range,
            self.num_data_args,
            self.base_frame,
            save=setting.save,
        )


@register
class Timing:
    def __init__(self, algo_name):
        self.num_data_args = 1
        self.algo_name = algo_name
        self.algo_func = getattr(algorithms, algo_name)

    base_frame = pd.DataFrame(
        [],
        columns=["iteration", "k_target", "_", "elapsed_time"],
    )

    def run(self):

        def algo(_):
            def func():
                t = time.time()
                _ = self.algo_func(
                    dataset,
                    eps=setting.eps,
                    delta=setting.delta,
                    k=dataset.num_clusters,
                    with_scores=False,
                )
                delta_t = time.time() - t
                return [delta_t]

            return func

        POSTPROCESS(
            "timing",
            setting.tag,
            dataset.name,
            self.algo_name,
            algo,
            setting.num_iterations,
            dataset.num_clusters,
            "?",
            self.num_data_args,
            self.base_frame,
            save=setting.save,
        )


####################
#      MAIN        #
####################


def handle_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--setting",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        required=True,
    )
    parser.add_argument("--dataset", type=str, required=True)

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = handle_args()

    dataset = datasets.datasets_by_name[args.dataset]()
    setting = hypersettings.get(args.experiment, args.setting, dataset)

    print(setting)

    MAIN_TAG = "ccs24"

    experiment = EXPERIMENTS_BY_NAME[args.experiment](args.algorithm)
    experiment.run()
