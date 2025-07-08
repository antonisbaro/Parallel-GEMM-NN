import sys
from os import getcwd

sys.path.append(getcwd())

from time import perf_counter
import numpy as np
from utilities import CustomMatmul
from prettytable import PrettyTable

#######################################################################
#  Versions are: 'numpy', 'cudaSimple', 'cudaTranspose', 'cudaShmem'  #
#                                                                     #
#         Fill in the list with the versions you've implemented       #
#                                                                     #
#!-!-!-!-!-!-!-!-!-!-!-!-Do not remove numpy-!-!-!-!-!-!-!-!-!-!-!-!-!#
#######################################################################

versions = ['numpy', 'cudaSimple', 'cudaTranspose', 'cudaShmem'] 

# Dimensions for the square matrices to be tested
squareDimensions = [1024, 2048, 4096, 8192]

outTableHeader = [
                'Version',
                'Dimensions',
                'isValid',
                'Total Time (ms)',
                'Transfers Time (ms)',
                'speedup to numpy'
                ]

resultsList = [outTableHeader]        
outTable = PrettyTable(outTableHeader)

totalTimes = {}
transferTimes = {}
testResults = {}
resultsValidity = {}

arraysA = []
arraysB = []

for iSquareMatrixDim in squareDimensions:
    arraysA.append(np.random.randn(iSquareMatrixDim, iSquareMatrixDim).astype('float64'))
    arraysB.append(np.random.randn(iSquareMatrixDim, iSquareMatrixDim).astype('float64'))

bigTic = perf_counter()
for iVersion in versions:
    
    totalTimes[iVersion]    = {}
    transferTimes[iVersion] = {}
    testResults[iVersion]   = {}

    for idx, iSquareMatrixDim in enumerate(squareDimensions):    
        # perform a warmup 
        for i in range(5):
            iVersionTest, transfersTime = CustomMatmul(arraysA[idx], arraysB[idx], mode=iVersion)

        tic = perf_counter()
        iVersionTest, transfersTime = CustomMatmul(arraysA[idx], arraysB[idx], mode=iVersion)
        toc = perf_counter()

        totalTimes[iVersion][iSquareMatrixDim]    = toc - tic
        transferTimes[iVersion][iSquareMatrixDim] = transfersTime
        testResults[iVersion][iSquareMatrixDim]   = iVersionTest



for iVersion in versions:
    
    resultsValidity[iVersion] = True

    for idx, iSquareMatrixDim in enumerate(squareDimensions):    
        resultsValidity[iVersion] = resultsValidity[iVersion] and np.allclose(testResults['numpy'][iSquareMatrixDim], testResults[iVersion][iSquareMatrixDim])

for idx, iSquareMatrixDim in enumerate(squareDimensions):
    for iVersion in versions:    
        iRow = [
                iVersion,
                f'({iSquareMatrixDim}x{iSquareMatrixDim}) • ({iSquareMatrixDim}x{iSquareMatrixDim})', 
                resultsValidity[iVersion],
                np.round(1000*totalTimes[iVersion][iSquareMatrixDim], 2),  
                np.round(1000*transferTimes[iVersion][iSquareMatrixDim], 2),
                np.round(totalTimes['numpy'][iSquareMatrixDim]/totalTimes[iVersion][iSquareMatrixDim], 2),  
            ]

        outTable.add_row(iRow)
    outTable.add_row(['', '', '', '', '', ''])

bigToc = perf_counter()

with open(f'results/CUDAMatMulTests.out', 'w+') as f:
    print(outTable, file=f)
    print("Total Time:", np.round((bigToc-bigTic)/60, 2), "mins", file=f)

    f.close()

with open(f'plotFiles/CUDAMatMulTests.csv', 'w+') as f:
    print("Version,SquareMatrixDimension,TotalTime,TransfersTime", file=f)
    for idx, iSquareMatrixDim in enumerate(squareDimensions):
        for iVersion in versions:    
            print(f"{iVersion},{iSquareMatrixDim},{np.round(1000*totalTimes[iVersion][iSquareMatrixDim], 2)},{np.round(1000*transferTimes[iVersion][iSquareMatrixDim], 2)}", file=f)

    f.close()
