import seaborn as sns
import pandas as pd
import pickle as pkl
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
import os
import matplotlib

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

    sources = {
        "dplloyd": f"./data/exps/kopt/ccs24_fast_reps2_eps1.0/dplloyd_{dt_name}.pkl",
        "lshsplits": f"./data/exps/kopt/ccs24_fast_reps2_eps1.0/lshsplits_{dt_name}.pkl",
        "emmc": f"./data/exps/kopt/ccs24_fast_reps2_eps1.0/emmc_{dt_name}.pkl",
        "dpm": f"./data/exps/kopt/ccs24_fast_reps2_eps1.0/dpm_{dt_name}.pkl",
    }

    for algo_name, file_path in sources.items():

        if not os.path.exists(file_path):
            continue

        with open(file_path, "rb") as _file:
            df = pkl.load(_file)
            df["algo"] = algo_name
            df["dataset"] = dt_name
            dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

sns.set_palette("colorblind")

matplotlib.rcParams.update({"font.size": 38})
matplotlib.rc("figure", figsize=(18, 10))

df.loc[df["algo"] == "dplloyd", "algo"] = "DP-Lloyd"
df.loc[df["algo"] == "dpm", "algo"] = "DPM (ours)"
df.loc[df["algo"] == "lshsplits", "algo"] = "LSH-Splits"
df.loc[df["algo"] == "emmc", "algo"] = "EM-MC"

df.loc[df["dataset"] == "fashion_embs", "dataset"] = "MNIST Embs."
df.loc[df["dataset"] == "mnist_embs", "dataset"] = "Fashion Embs."
df.loc[df["dataset"] == "uci_letters", "dataset"] = "UCI Letters"
df.loc[df["dataset"] == "uci_gas_emissions", "dataset"] = "UCI Gas Emissions"
df.loc[df["dataset"] == "synth_10d", "dataset"] = "Synth-10d"
df.loc[df["dataset"] == "synth_big1", "dataset"] = "Synth-100d"

# remove rows with dataset in uci
df = df[~df["dataset"].str.contains("UCI")]

hue_order = ["EM-MC", "DP-Lloyd", "LSH-Splits", "DPM (ours)"]

datasets = ["MNIST Embs.", "Fashion Embs", "Synth-10d", "Synth-100d"]

ax = sns.barplot(
    df,
    x="dataset",
    y="norm_kmeans_dist",
    hue="algo",
    hue_order=hue_order,
    fill=False,
    linewidth=5,
    width=0.9,
    gap=0.2,
)

ax.text(1.71, 0.008, "X", fontsize=35, color="#0173B2", ha="center")

hatches = ["//", "|", "-", "o"]
for bars, hatch in zip(ax.containers, hatches):
    for bar in bars:
        bar.set_hatch(hatch)

ax.set_xticklabels(ax.get_xticklabels(), rotation=0, horizontalalignment="center")

handles = []
for hatch, color, label in zip(
    hatches, sns.color_palette("colorblind")[: len(hatches)], hue_order
):
    handles.append(
        Patch(facecolor="white", edgecolor=color, hatch=hatch, label=label, linewidth=5)
    )

plt.legend(
    loc="upper left",
    title="Algorithm",
    handles=handles,
)

plt.yscale("log")
plt.ylabel("Norm. KMeans Distance")
plt.xlabel("")

plt.tight_layout()

plt.savefig(
    f"./plots/figure1.pdf",
    transparent=True,
)
