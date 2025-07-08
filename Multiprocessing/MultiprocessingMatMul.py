from copy import deepcopy
from multiprocessing import Process, shared_memory, cpu_count
from Multiprocessing.tools import ReleaseSharedMemory, CreateSharedNumpyArray
import numpy as np

def MultiprocessingMatMul(A, B, numberOfProcesses=1):

    def MultiprocessingTargetFunction(A, B, C, iStartIndex, iEndIndex):
        numCols = B.shape[1]
        innerDim = B.shape[0]

        # Perform GEMM for the rows assigned to each process
        # Iterate over the subset of rows assigned to this process
        for i in range(iStartIndex, iEndIndex):
            # Iterate over the columns of the output matrix
            for j in range(numCols):
                # Perform the dot product for the element C[i, j]
                sum_val = 0
                for k in range(innerDim):
                    sum_val += A[i, k] * B[k, j]
                C[i, j] = sum_val
        return
    
    outArrayShape = (A.shape[0], B.shape[1])

    # Create the shared memory block for the output array C
    CreateSharedNumpyArray(outArrayShape, arrayName='C')
    shm = shared_memory.SharedMemory(name='C')
    C = np.ndarray(outArrayShape, dtype=np.float64, buffer=shm.buf)
    
    # Initialize the output array to zeros
    C.fill(0)

    # Calculate the number of rows per process
    numRowsPerProcess = A.shape[0] // numberOfProcesses  

    # Calculate the remaining rows to be handled by the last process
    lastProcessExtraRows = A.shape[0] % numberOfProcesses # The last process might get extra rows, if there are any left

    processes = []
    for i in range(numberOfProcesses):
        # Calculate the start and end row index for each process
        iStartIndex = i * numRowsPerProcess
        iEndIndex = iStartIndex + numRowsPerProcess

        # The last process takes any remaining rows
        if i == numberOfProcesses - 1:
            iEndIndex += lastProcessExtraRows 

        # Create a process targeting our function
        iProcess = Process(target=MultiprocessingTargetFunction,
                              args=(A, B, C, iStartIndex, iEndIndex,))

        processes.append(iProcess)
        iProcess.start()

    # Wait for all processes to finish their execution
    for iProcess in processes:
        iProcess.join()

    # Deepcopy the result from shared memory to a new array
    outC = deepcopy(C)
    
    # Release the shared memory block
    ReleaseSharedMemory(['C'])
    
    return outC, 0
