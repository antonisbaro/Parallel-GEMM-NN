import sys
from os import getcwd
from os.path import dirname
from time import perf_counter
import numpy as np
from prettytable import PrettyTable

sys.path.append(getcwd())
from Multiprocessing.MultiprocessingMatMul import ReleaseSharedMemory
from nn import MLP, SequentialNN, ReLU, Optimizer, LogSoftmax, train
from utilities import load_data

#######################################################################
#  Versions are: 'numpy', 'cudaSimple', 'cudaTranspose', 'cudaShmem'  #
#                                                                     #
#!-!-!-!-!-!-!-!-!-!-!-!-Do not remove numpy-!-!-!-!-!-!-!-!-!-!-!-!-!#
#######################################################################
matMulVersions = ['numpy', 'cudaSimple', 'cudaTranspose', 'cudaShmem']

numberOfEpochs = 1

batchSize      = 8192
outputSizeMLP0 = 8192
outputSizeMLP1 = 4096


outTableHeader = [
                    'Version', 
                    'NN Test Accuracy', 
                    'Total Time (s)', 
                    'Transfers Time (s)', 
                    'Transfers to Total Time ratio',
                    'speedup to numpy'
                ]
resultsList = [outTableHeader]        
outTable = PrettyTable(outTableHeader)
serialTime = 1

# Load and process data
(trainX, trainy), (testX, testy) = load_data('mnistDataset.npz')
trainX = (trainX - 127.5) / 127.5
testX = (testX - 127.5) / 127.5
trainX = trainX.reshape(trainX.shape[0], 28 * 28)


transfersTimes = {}
parallelTimes = {}
accuracies = {}

bigTic = perf_counter()

for iMatMulMode in matMulVersions:

    mlp = SequentialNN([
                        MLP(28 * 28, outputSizeMLP0, matMullMode=iMatMulMode), ReLU(),
                        MLP(outputSizeMLP0, outputSizeMLP1,    matMullMode=iMatMulMode), ReLU(),
                        MLP(outputSizeMLP1, 10,      matMullMode=iMatMulMode), LogSoftmax()
                    ])

    optimizer = Optimizer(1e-3, mlp)

    tic = perf_counter()
    training_loss = train(mlp, optimizer, trainX, trainy, nb_epochs=numberOfEpochs, batch_size=batchSize)
    toc = perf_counter()

    parallelTime = toc - tic

    accuracy = 0
    for i in range(testX.shape[0]):
        prediction = mlp.forward(testX[i].reshape(1, 784)).argmax()
        if prediction == testy[i]:
            accuracy += 1

    transfersTime = mlp.getCUDATransferTimes()
    
    transfersTimes[iMatMulMode] = transfersTime
    parallelTimes[iMatMulMode]  = parallelTime
    accuracies[iMatMulMode]     = str(np.round((accuracy / testX.shape[0]) * 100, 1))+'%'


for iMatMulMode in matMulVersions:
    transfersTime = transfersTimes[iMatMulMode]
    parallelTime  = parallelTimes[iMatMulMode]
    accuracy      = accuracies[iMatMulMode]     

    iRow = [
            iMatMulMode, 
            accuracy,
            np.round(parallelTime, 3),
            np.round(transfersTime, 3),
            np.round(transfersTime/parallelTime, 3),
            np.round(parallelTimes['numpy']/parallelTime, 3)
        ]
        
    resultsList.append(iRow)
        
    outTable.add_row(iRow)

bigToc = perf_counter()

with open(f'results/NN_epochs-{numberOfEpochs}_batchSize-{batchSize}_outMLP0-{outputSizeMLP0}_outMLP1-{outputSizeMLP1}.out', 'w+') as f:
    print(outTable, file=f)
    print("Total Time:", np.round((bigToc-bigTic)/60, 2), "mins", file=f)

    f.close()

# CSV 
csvHeader = ['Version', 'NN_Test_Accuracy', 'TotalTime_s', 'TransfersTime_s', 'Transfers_to_Total_Time_ratio', 'speedup_to_numpy']
csvTable = PrettyTable(csvHeader)

for row in resultsList[1:]:
    csvTable.add_row(row)

csv_output_filename = f'plotFiles/NN_epochs-{numberOfEpochs}_batchSize-{batchSize}_outMLP0-{outputSizeMLP0}_outMLP1-{outputSizeMLP1}.csv'

with open(csv_output_filename, 'w+') as f:
    f.write(csvTable.get_csv_string())
    f.close()
