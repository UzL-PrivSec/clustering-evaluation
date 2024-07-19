import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
import pickle as pkl
import matplotlib
import numpy as np

with open(
    f"./data/exps/centreness/ccs24_paper_reps10_eps1.0/dpm_synth_10d.pkl", "rb"
) as _file:
    df = pkl.load(_file)

df["tq"] = list(zip(df["t"], df["q"]))
df = df.groupby("tq").mean().reset_index().round(2)

for metric in ["accuracy", "silh_score", "inertia", "norm_kmeans_dist"]:

    df_metric = pd.pivot(data=df, index="t", columns="q", values=metric)

    matplotlib.rcParams.update({"font.size": 45})
    matplotlib.rc("figure", figsize=(18, 12))

    kwargs = {
        "cmap": "magma",
        "annot": False,
        "cbar": True,
        "square": False,
        "linecolor": "gray",
        "linewidths": 0.5,
    }

    if metric == "accuracy":
        ax = sns.heatmap(df_metric, vmin=0, vmax=1, **kwargs)
    elif metric == "silh_score":
        ax = sns.heatmap(df_metric, vmin=-1, vmax=1, **kwargs)
    elif metric == "inertia":
        ax = sns.heatmap(df_metric, vmin=0, **kwargs)
    elif metric == "norm_kmeans_dist":
        ax = sns.heatmap(df_metric, vmin=0, **kwargs)
    else:
        ax = sns.heatmap(df_metric, **kwargs)

    plt.savefig(
        f"./plots/figure4_{metric}.pdf",
        transparent=True,
    )

    plt.cla()
    plt.clf()
