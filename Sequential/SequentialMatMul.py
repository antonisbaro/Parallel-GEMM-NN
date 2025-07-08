import numpy as np
from tqdm import tqdm

def SequentialMatMul(A, B):

    numRows = A.shape[0]
    numCols = B.shape[1]
    innerDim = B.shape[0]
    outArrayShape = (numRows, numCols)
    
    C = np.zeros(outArrayShape)

    for row in range(numRows):
        for col in range(numCols):
            tempSum = 0.0
            for k in range(innerDim):
                tempSum += A[row][k] * B[k][col]

            C[row][col] = tempSum
    return C
