# Artifact Evaluation CCS24 for "DPM: Clustering Sensitive Data through Separation"
This repository contains all necessary code and instructions to replicate the experiments and analyses described in our paper "DPM: Clustering Sensitive Data through Separation. Please follow the guidelines below for setup, running experiments, and generating plots.

## Setup
If you are coming from Zenodo then you can either clone this repository and proceed OR skip the next step and add the clustering algorithms manually from the assets of the corresponding release.

First, make sure to download the submodules as well:
```bash
git submodule update --init --recursive
```
Install **Python 3.10** and download the required packages:
```bash
pip install -r requirements.txt
pip install -r clustering_algorithms/dpm/requirements.txt
pip install -r clustering_algorithms/emmc/requirements.txt
```
Setup the clustering algorithm lshsplits:
```bash
cd /clustering_algorithms/lshsplits/learning/
python3 setup.py install
cd ../../../
```
Finally, prepare the missing data sets:
```bash
python3 init.py
```

## Experiments
This repository contains experiments to reproduce every plot including Table 2. Each experiment is split into two parts, generating the data and creating a plot or table. The available experiments are:
| Figure      | Experiment  Name    | Description      |
| ------------- |------------- |------------ |
| Figure 1 | KOpt | Comparison of the KMeans distance |
| Figure 4 | Centreness | Influence of t,q on the centreness |
| Figure 5 | Time | Running time of each algorithm |
| Figure 6 | EpsDist | Distribution of the privacy budget (epsilon) of DPM |
| Table 2 | KOpt | Algorithms evaluated on all metrics |

First you have to generate data that will be stored in *data/exps/\<Experiment Name\>/\**
```bash
. start.sh <Experiment Name>
```
and afterwards you can create a corresponding plot or table that will be stored in *plots/\<Figure\>*
```bash
python3 plotting/<Figure>
```
## Misc
- To create Figure 1 and Table 2, the experiment KOpt only needs to be run once.
- Currently, *start.sh* performs all runs sequentially. However, the runs can also be parallelised to speed up the experiments.
