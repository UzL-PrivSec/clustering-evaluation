# Artifact Evaluation CCS24 for "DPM: Clustering Sensitive Data through Separation"
This repository contains all necessary code and instructions to replicate the experiments and analyses described in our paper "DPM: Clustering Sensitive Data through Separation. Please follow the guidelines below for setup, running experiments, and generating plots.

## Setup
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
The code contains an experiment to reproduce every plot including table2. Each experiment is split into two partes, generating the data and creating a plot or table. The available experiments are:
| Figure      | Experiment  Name    | Description      |
| ------------- |------------- |------------ |
| Figure1 | KOpt | Comparison of the KMeans distance |
| Figure4 | Centreness | Influence of t,q onf the centreness |
| Figure5 | Timing | Running Time of each algorithm |
| Figure6 | EpsDist | Distribution of the privacy budget (epsilon) inside DPM |
| Table2 | KOpt | Algorithms evaluated on all metrics |

First you have to generate data that will be stored in *data/exps/\<Experiment Name\>/\**
```bash
. start.sh <Experiment Name>
```
and afterwards you can create a corresponding plot or table that will be stored in *plots/\<Figure\>*
```bash
python3 plotting/<Figure>
```
## Misc
- To create figure 1 and table2, the experiment KOpt only needs to be run once. 
