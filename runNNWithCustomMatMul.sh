#!/bin/bash
#PBS -o output/NeuralNetwork.out
#PBS -e error/NeuralNetwork.err
#PBS -l walltime=01:00:00
#PBS -q serial
#PBS -l nodes=silver1:ppn=40

cd $HOME/ParallelMatMulOnNeuralNetworkAssignment

python3 testNN.py
