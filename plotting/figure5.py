import seaborn as sns
import pandas as pd
import pickle as pkl
from matplotlib import pyplot as plt
import os
import matplotlib
from matplotlib.patches import Patch

dts = [
    "fashion_embs",
    "mnist_embs",
    "synth_big1",
    "synth_10d",
    "uci_letters",
    "uci_gas_emissions",
]

dfs = []
for dt_name in dts:

    file_path = {
        "dpm": f"./data/exps/timing/ccs24_fast_reps2_eps1.0/dpm_{dt_name}.pkl",
        "dplloyd": f"./data/exps/timing/ccs24_fast_reps2_eps1.0/dplloyd_{dt_name}.pkl",
        "kmeans": f"./data/exps/timing/ccs24_fast_reps2_eps1.0/kmeans_{dt_name}.pkl",
        "emmc": f"./data/exps/timing/ccs24_fast_reps2_eps1.0/emmc_{dt_name}.pkl",
        "lshsplits": f"./data/exps/timing/ccs24_fast_reps2_eps1.0/lshsplits_{dt_name}.pkl",
    }

    for algo_name, path in file_path.items():

        if not os.path.exists(path):
            continue

        with open(path, "rb") as _file:
            df = pkl.load(_file)
            df["dataset"] = dt_name
            df["algo"] = algo_name
            dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

sns.set_palette("colorblind")

matplotlib.rcParams.update({"font.size": 38})
matplotlib.rc("figure", figsize=(18, 8))

df = df.drop(df.loc[(df["dataset"] == "synth_big1") & (df["algo"] == "emmc")].index)

df.loc[df["dataset"] == "fashion_embs", "dataset"] = "MNIST Embs."
df.loc[df["dataset"] == "mnist_embs", "dataset"] = "Fashion Embs."
df.loc[df["dataset"] == "uci_letters", "dataset"] = "UCI Letters"
df.loc[df["dataset"] == "uci_gas_emissions", "dataset"] = "UCI Gas"
df.loc[df["dataset"] == "synth_10d", "dataset"] = "Synth-10d"
df.loc[df["dataset"] == "synth_big1", "dataset"] = "Synth-100d"

df.loc[df["algo"] == "dplloyd", "algo"] = "DP-Lloyd"
df.loc[df["algo"] == "dpm", "algo"] = "DPM (ours)"
df.loc[df["algo"] == "lshsplits", "algo"] = "LSH-Splits"
df.loc[df["algo"] == "emmc", "algo"] = "EM-MC"
df.loc[df["algo"] == "kmeans", "algo"] = "KMeans (non-private)"

hue_order = ["EM-MC", "DP-Lloyd", "LSH-Splits", "DPM (ours)", "KMeans (non-private)"]

ax = sns.barplot(
    df,
    x="dataset",
    y="elapsed_time",
    hue="algo",
    hue_order=hue_order,
    fill=False,
    linewidth=5,
    width=0.9,
    gap=0.2,
)

ax.text(1.65, 0.08, "X", fontsize=35, color="#0173B2", ha="center")

ax.tick_params(axis="x", which="major", labelsize=26)

hatches = ["//", "|", "-", "o", "x", "--"]

for bars, hatch in zip(ax.containers, hatches):
    for bar in bars:
        bar.set_hatch(hatch)

ax.set_xticklabels(ax.get_xticklabels(), rotation=0, horizontalalignment="center")

plt.ylabel("Running Time (s)")
plt.xlabel("")

plt.yscale("log")

handles = []
for hatch, color, label in zip(
    hatches, sns.color_palette("colorblind")[: len(hatches)], hue_order
):
    handles.append(
        Patch(facecolor="white", edgecolor=color, hatch=hatch, label=label, linewidth=5)
    )

plt.legend(title="Algorithm", handles=handles, loc="upper left", fontsize=26)

plt.tight_layout()

plt.savefig(
    f"./plots/figure5.pdf",
    transparent=True,
)
