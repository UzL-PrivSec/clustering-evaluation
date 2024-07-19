import seaborn as sns
import pandas as pd
import pickle as pkl
from matplotlib import pyplot as plt
import matplotlib
from pandas.plotting import parallel_coordinates


with open(
    f"./data/exps/epsDist/ccs24_paper_reps10_eps1.0/dpm_mnist_embs.pkl", "rb"
) as _file:
    df = pkl.load(_file)

matplotlib.rcParams.update({"font.size": 18})
matplotlib.rc("figure", figsize=(12, 4))

cmap_name = "flare"

df["eps_dist"] = list(
    zip(df["eps_bandwidth"], df["eps_split"], df["eps_count"], df["eps_average"])
)
df = df.groupby(["eps_dist"]).mean(numeric_only=True).reset_index()

top = 0.2

leng = len(df) * top

for metric in ["accuracy", "silh_score", "inertia", "norm_kmeans_dist"]:

    ascending = metric in ["inertia", "norm_kmeans_dist"]
    df_metric = df.sort_values(by=metric, ascending=ascending)

    df_metric = pd.concat(
        [
            df_metric.head(int(len(df_metric) * top)),
            df_metric.tail(int(len(df_metric) * top)),
        ],
        ignore_index=True,
    )

    df_metric["class"] = df_metric.index < leng

    cl_palette = sns.color_palette("colorblind")

    styles = {
        False: {"color": cl_palette[2], "linestyle": "--", "linewidth": 2},
        True: {"color": cl_palette[1], "linestyle": "-", "linewidth": 2},
    }

    for class_name, style in styles.items():
        subset = df_metric[df_metric["class"] == class_name]
        lines = parallel_coordinates(
            subset,
            class_column="class",
            cols=["eps_bandwidth", "eps_split", "eps_count", "eps_average"],
            color=[style["color"]],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            use_columns=False,
        )

    handles = [
        plt.Line2D(
            [0],
            [0],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
        )
        for style in styles.values()
    ]
    labels = ["Bottom 20%", "Top 20%"]

    plt.ylabel("Epsilon Proportion")

    if metric == "norm_kmeans_dist":
        plt.xticks(
            [0, 1, 2, 3],
            ["Epsilon Bandwidth", "Epsilon Split", "Epsilon Count", "Epsilon Average"],
        )
    else:
        plt.xticks([0, 1, 2, 3], ["", "", "", ""])

    metric_names = {
        "inertia": "Inertia",
        "norm_kmeans_dist": "Norm. KMeans Dist.",
        "silh_score": "Silh. Score",
        "accuracy": "Accuracy",
    }

    plt.legend(
        loc="upper left",
        title=metric_names[metric],
        labels=labels[::-1],
        handles=handles[::-1],
    )

    plt.savefig(
        f"./plots/figure6_{metric}.pdf",
        transparent=True,
    )

    plt.cla()
    plt.clf()
