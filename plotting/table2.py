import pandas as pd
import os
import pickle as pkl

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
        "dpm": f"./data/exps/kopt/ccs24_paper_reps20_eps1.0/dpm_{dt_name}.pkl",
        "dplloyd": f"./data/exps/kopt/ccs24_paper_reps20_eps1.0/dplloyd_{dt_name}.pkl",
        "kmeans": f"./data/exps/kopt/ccs24_paper_reps20_eps1.0/kmeans_{dt_name}.pkl",
        "emmc": f"./data/exps/kopt/ccs24_paper_reps20_eps1.0/emmc_{dt_name}.pkl",
        "lshsplits": f"./data/exps/kopt/ccs24_paper_reps20_eps1.0/lshsplits_{dt_name}.pkl",
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

stats_df = (
    df.groupby(["algo", "dataset"])
    .agg(
        {
            "silh_score": ["mean", "std"],
            "inertia": ["mean", "std"],
            "accuracy": ["mean", "std"],
            "norm_kmeans_dist": ["mean", "std"],
        }
    )
    .reset_index()
)

stats_df.columns = [" ".join(col).strip() for col in stats_df.columns.values]

for metric in ["silh_score", "inertia", "accuracy", "norm_kmeans_dist"]:
    mean_col = f"{metric} mean"
    std_col = f"{metric} std"
    if metric == "inertia":
        stats_df[f"{metric}"] = stats_df.apply(
            lambda x: f"{x[mean_col]:.1e} (± {x[std_col]:.1e})", axis=1
        )
    else:
        stats_df[f"{metric}"] = stats_df.apply(
            lambda x: f"{x[mean_col]:.2f} (± {x[std_col]:.2f})", axis=1
        )
    stats_df.drop(columns=[mean_col, std_col], inplace=True)

stats_df = stats_df.transpose()

stats_df.reset_index(inplace=True)
stats_df.columns = stats_df.iloc[0]
stats_df = stats_df[1:]

stats_df.to_markdown(f"./plots/table2.md", index=False)
