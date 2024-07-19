from dataclasses import dataclass
from itertools import product

import numpy as np


@dataclass
class HyperSetting:
    name: str
    num_iterations: int
    save: bool
    eps: float
    delta: float

    @property
    def tag(self):
        return f"{self.name}_reps{self.num_iterations}_eps{self.eps}"


@dataclass
class KOptSetting(HyperSetting):
    pass


@dataclass
class CentrenessSetting(HyperSetting):
    tq_range: np.ndarray

    @property
    def tag(self):
        return super().tag + f"_tq{self.tq_range[0]}-{self.tq_range[-1]}"


@dataclass
class EpsDistSetting(HyperSetting):
    eps_dist_range: np.ndarray

    @property
    def tag(self):
        return super().tag


@dataclass
class TimingSetting(HyperSetting):

    @property
    def tag(self):
        return super().tag


def get(experiment: str, flavor: str, dataset):

    setting = None

    match experiment:
        case "KOpt":
            match flavor:
                case "test":
                    setting = KOptSetting(
                        name=flavor,
                        num_iterations=5,
                        eps=1.0,
                        delta=1 / (dataset.num_points * np.sqrt(dataset.num_points)),
                        save=False,
                    )
                case "paper":
                    setting = KOptSetting(
                        name=flavor,
                        num_iterations=20,
                        eps=1.0,
                        delta=1 / (dataset.num_points * np.sqrt(dataset.num_points)),
                        save=True,
                    )
        case "Centreness":
            match flavor:
                case "test":
                    setting = CentrenessSetting(
                        name=flavor,
                        num_iterations=1,
                        eps=1,
                        delta=1 / (dataset.num_points * np.sqrt(dataset.num_points)),
                        tq_range=np.asarray([[1, 1], [0.3, 1 / 12]]),
                        save=False,
                    )
                case "paper":
                    setting = CentrenessSetting(
                        name=flavor,
                        num_iterations=10,
                        eps=1,
                        delta=1 / (dataset.num_points * np.sqrt(dataset.num_points)),
                        tq_range=np.asarray(
                            [
                                [0.3, 0.1],
                                [0.4, 0.1],
                                [0.5, 0.1],
                                [0.6, 0.1],
                                [0.7, 0.1],
                                [0.8, 0.1],
                                [0.9, 0.1],
                                [0.5, 0.2],
                                [0.6, 0.2],
                                [0.7, 0.2],
                                [0.8, 0.2],
                                [0.9, 0.2],
                                [0.7, 0.3],
                                [0.8, 0.3],
                                [0.9, 0.3],
                                [0.9, 0.4],
                            ]
                        ),
                        save=True,
                    )
        case "EpsDist":
            match flavor:
                case "test":
                    setting = EpsDistSetting(
                        name=flavor,
                        num_iterations=1,
                        eps=1,
                        delta=1 / (dataset.num_points * np.sqrt(dataset.num_points)),
                        eps_dist_range=np.asarray(
                            [[0.04, 0.18, 0.18, 0.6], [0.4, 0.3, 0.2, 0.1]]
                        ),
                        save=False,
                    )
                case "paper":
                    eps_bandwidth_range = np.arange(0.1, 1, 0.1)
                    eps_split_range = np.arange(0.1, 1, 0.1)
                    eps_count_range = np.arange(0.1, 1, 0.1)
                    eps_average_range = np.arange(0.1, 1, 0.1)
                    eps_dist_range = np.asarray(
                        [
                            vels
                            for vels in product(
                                eps_bandwidth_range,
                                eps_split_range,
                                eps_count_range,
                                eps_average_range,
                            )
                            if np.isclose(sum(vels), 1, atol=0.05)
                        ]
                    )
                    setting = EpsDistSetting(
                        name=flavor,
                        num_iterations=10,
                        eps=1,
                        delta=1 / (dataset.num_points * np.sqrt(dataset.num_points)),
                        eps_dist_range=eps_dist_range,
                        save=True,
                    )
        case "Timing":
            match flavor:
                case "test":
                    setting = TimingSetting(
                        name=flavor,
                        num_iterations=2,
                        eps=1.0,
                        delta=1 / (dataset.num_points * np.sqrt(dataset.num_points)),
                        save=False,
                    )
                case "paper":
                    setting = TimingSetting(
                        name=flavor,
                        num_iterations=10,
                        eps=1.0,
                        delta=1 / (dataset.num_points * np.sqrt(dataset.num_points)),
                        save=True,
                    )

    if setting is None:
        raise ValueError(f"Invalid experiment.flavor: {experiment}.{flavor}")

    return setting
