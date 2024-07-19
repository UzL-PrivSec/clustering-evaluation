#!/bin/bash

experiment=$1 

if [ ${experiment} = 'KOpt' ]; then
    datasets=("fashion_embs" "mnist_embs" "synth_big1" "synth_10d" "uci_letters" "uci_gas_emissions")
    algos=("kmeans" "dplloyd" "emmc" "lshsplits" "dpm")
elif [ ${experiment} = 'Timing' ]; then
    datasets=("fashion_embs" "mnist_embs" "synth_big1" "synth_10d" "uci_letters" "uci_gas_emissions")
    algos=("kmeans" "dplloyd" "emmc" "lshsplits" "dpm")
elif [ ${experiment} = 'EpsDist' ]; then
    datasets=("mnist_embs")
    algos=("dpm")
elif [ ${experiment} = 'Centreness' ]; then
    datasets=("synth_10d")
    algos=("dpm")
fi

for dataset in "${datasets[@]}"
do
    for algorithm in "${algos[@]}"
        do
            python experiments.py --experiment $experiment --dataset $dataset --setting paper --algorithm $algorithm
        done
done    