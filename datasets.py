from dataclasses import dataclass
import numpy as np
import pickle

from sklearn.datasets import make_blobs


@dataclass
class _Dataset:
    data: np.ndarray
    y: np.ndarray
    bounds: tuple
    num_clusters: int
    num_points: int
    num_dims: int
    name: str

    @property
    def range(self):
        return self.bounds[1] - self.bounds[0]

    @property
    def num_labels(self):
        return len(np.unique(self.y))

    @property
    def max_norm(self):
        cap = (
            self.bounds[1]
            if np.abs(self.bounds[1]) > np.abs(self.bounds[0])
            else -self.bounds[0]
        )
        return cap * np.sqrt(self.num_dims)

    @property
    def radius(self):
        return (self.bounds[1] - self.bounds[0]) * np.sqrt(self.num_dims) / 2


def _create_synth(
    num_points, num_dims, num_clusters, bounds, seed, name, cluster_std=1
):
    data, y = make_blobs(
        n_samples=num_points,
        n_features=num_dims,
        centers=num_clusters,
        random_state=seed,
        center_box=bounds,
        cluster_std=cluster_std,
    )

    return _Dataset(data, y, bounds, num_clusters, num_points, num_dims, name)


def Synth10D():
    return _create_synth(
        num_points=100000,
        num_dims=10,
        num_clusters=64,
        bounds=(-100, 100),
        seed=42,
        name="synth_10d",
    )


def SynthBig1():
    return _create_synth(
        num_points=100000,
        num_dims=100,
        num_clusters=64,
        bounds=(-100, 100),
        seed=42,
        name="synth_big1",
    )


def uciLetters():

    # https://archive.ics.uci.edu/dataset/59/letter+recognition

    with open(f"./data/dts/uciLetters/uciLetters.pkl", "rb") as f_:
        x, y = pickle.load(f_)

    num_per_class = 720
    num_classes = 26
    emb_size = 16

    generator = np.random.default_rng(seed=42)

    x_new = np.zeros((num_classes * num_per_class, emb_size))
    y_new = np.zeros(num_classes * num_per_class).astype(int)

    for i in range(num_classes):
        x_label_i = x[y == i]
        indices = generator.integers(0, len(x_label_i), size=num_per_class)
        x_new[i * num_per_class : (i + 1) * num_per_class] = x_label_i[indices]
        y_new[i * num_per_class : (i + 1) * num_per_class] = i

    perm = generator.permutation(num_per_class * num_classes)
    x_new = x_new[perm]
    y_new = y_new[perm]

    bounds = (0, 15)

    dt = _Dataset(
        x_new,
        y_new,
        bounds,
        num_classes,
        num_per_class * num_classes,
        emb_size,
        "uci_letters",
    )

    return dt


def uciGasEmissions():

    # https://archive.ics.uci.edu/dataset/551/gas+turbine+co+and+nox+emission+data+set

    with open(f"./data/dts/uciGasEmissions/uciGasEmissions.pkl", "rb") as f_:
        x, y = pickle.load(f_)

    y = y.astype(int)

    emb_size = 11

    bounds = (-6.23, 1100.9)

    return _Dataset(
        x,
        y,
        bounds,
        2,
        len(x),
        emb_size,
        "uci_gas_emissions",
    )


def mnistEmbs():

    with open(f"./data/dts/mnistEmbs/mnist_embs.pkl", "rb") as f_:
        x, y = pickle.load(f_)

    y = y.astype(int)

    emb_size = 40

    bounds = (0, 13)

    dt = _Dataset(x, y, bounds, 10, len(x), emb_size, "mnist_embs")

    return dt


def fashionEmbs():

    with open(f"./data/dts/mnistFashion/fashion_embs.pkl", "rb") as f_:
        x, y = pickle.load(f_)

    y = y.astype(int)

    emb_size = 40

    bounds = (0, 13)

    return _Dataset(
        x,
        y,
        bounds,
        10,
        len(x),
        emb_size,
        "fashion_embs",
    )


datasets_by_name = {
    "mnist_embs": mnistEmbs,
    "fashion_embs": fashionEmbs,
    "synth_10d": Synth10D,
    "synth_big1": SynthBig1,
    "uci_letters": uciLetters,
    "uci_gas_emissions": uciGasEmissions,
}
