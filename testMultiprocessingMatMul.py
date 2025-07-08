import sys
from os import getcwd

sys.path.append(getcwd())

from time import perf_counter
import numpy as np
from utilities import CustomMatmul
from prettytable import PrettyTable
from Multiprocessing.tools import ReleaseSharedMemory

try:
    ReleaseSharedMemory(['C'])
except:
    pass


numberOfProcesses = [1, 10, 20, 40]

# Dimensions for the square matrices to be tested
squareDimensions = [100, 200, 400, 600]


outTableHeader = ['Number of Processes', 'Dimensions', 'Multiprocessing Time', 'speedup to serial', 'Multiprocessing isValid']
resultsList = [outTableHeader]        
outTable = PrettyTable(outTableHeader)

csvHeader = ['NumberOfProcesses', 'SquareMatrixDimension', 'ExecutionTime', 'speedupToSerial', 'NumpyExecutionTime']
csvTable = PrettyTable(csvHeader)

serialTime = 1
bigTic = perf_counter()

for iSquareMatrixDim in squareDimensions:
    for iNumProcesses in numberOfProcesses:
        
        A = np.random.randn(iSquareMatrixDim, iSquareMatrixDim).astype('float64')
        B = np.random.randn(iSquareMatrixDim, iSquareMatrixDim).astype('float64')

        tic = perf_counter()
        numpyTest = np.matmul(A, B)
        toc = perf_counter()
        numpyTestTime = toc - tic

        tic = perf_counter()
        multiprocessingTest, _ = CustomMatmul(A, B, mode="multiprocessing", numberOfProcesses=iNumProcesses)
        toc = perf_counter()
        multiprocessingTestTime = toc - tic
        if iNumProcesses==1:
            serialTime = multiprocessingTestTime

        iRow = [
                iNumProcesses,
                f'({iSquareMatrixDim}x{iSquareMatrixDim}) • ({iSquareMatrixDim}x{iSquareMatrixDim})',
                np.round(multiprocessingTestTime, 3),
                np.round(serialTime/multiprocessingTestTime, 3),
                np.allclose(numpyTest, multiprocessingTest)
                ]
        
        outTable.add_row(iRow)
        csvTable.add_row([
                iNumProcesses,
                iSquareMatrixDim,
                np.round(multiprocessingTestTime, 3),
                np.round(serialTime/multiprocessingTestTime, 3),
                np.round(numpyTestTime, 3), #Αυτό το πρόσθεσα εγω γτ δεν υπήρχε
        ])
    outTable.add_row(['', '', '', '', ''])

bigToc = perf_counter()

with open(f'results/MultiprocMatMulTests.out', 'w+') as f:
    print(outTable, file=f)
    print("Total Time:", np.round((bigToc-bigTic)/60, 2), "mins", file=f)

    f.close()

with open(f'plotFiles/MultiprocMatMulTests.csv', 'w+') as f:
    f.write(csvTable.get_csv_string())
    f.close()

try:
    ReleaseSharedMemory(['C'])
except:
    pass
