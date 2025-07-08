#!/bin/bash
#PBS -o output/MultiprocessingMatMul.out
#PBS -e error/MultiprocessingMatMul.err
#PBS -l walltime=00:30:00
#PBS -q serial
#PBS -l nodes=silver1:ppn=40

cd $HOME/ParallelMatMulOnNeuralNetworkAssignment

python3 testMultiprocessingMatMul.py 
