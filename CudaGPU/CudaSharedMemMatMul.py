from numba import cuda, float64
import numpy as np
from time import perf_counter
from os import environ
import math # Added for math.ceil

environ["CUDA_VISIBLE_DEVICES"] = "1"

threadGridDim = 32
threadsPerBlock = (threadGridDim, threadGridDim)

@cuda.jit("void(float64[:,:], float64[:,:],  float64[:,:])")
def SharedMemMatMul2DKernel(A, B, C):
    # 2D thread grid
    # get thread coords from cuda.grid
    # (returns absolute position of the current thread in the entire grid of blocks)
    col, row = cuda.grid(2)

    # Create the shared memory arrays whose shape is dictated by the thread dimensions of each block
    sharedMemA = cuda.shared.array((threadGridDim, threadGridDim), dtype=float64)
    sharedMemB = cuda.shared.array((threadGridDim, threadGridDim), dtype=float64)

    if row >= C.shape[0] or col >= C.shape[1]:
        return
    
    # Indexes on the current thread block will be used for array indexing
    threadX = cuda.threadIdx.x
    threadY = cuda.threadIdx.y
    k_chunks = int(B.shape[0] / threadGridDim)
    if B.shape[0] % threadGridDim != 0:
        k_chunks+=1
    
    # each thread will compute one element of C
    # to do that it will use the shared memory for
    # computing and storing the sum of the subproducts (dot product)
    tempSum = 0.0
    for i in range(k_chunks):
        
        # Each thread fills an element of sharedMemA and one of sharedMemB. Note: beware of the boundaries of the K dimension
        # Calculate global indices for this tile
        global_A_row = row
        global_A_col = i * threadGridDim + threadX

        global_B_row = i * threadGridDim + threadY
        global_B_col = col

        # Load from global memory to shared memory with boundary checks
        if global_A_row < A.shape[0] and global_A_col < A.shape[1]:
            sharedMemA[threadY, threadX] = A[global_A_row, global_A_col]
        else:
            sharedMemA[threadY, threadX] = 0.0

        if global_B_row < B.shape[0] and global_B_col < B.shape[1]:
            sharedMemB[threadY, threadX] = B[global_B_row, global_B_col]
        else:
            sharedMemB[threadY, threadX] = 0.0

            
        cuda.syncthreads()

        # Each thread updates its tempSum using the elements of sharedMemA and sharedMemB. Note: beware of the boundaries of the K dimension
        # Multiply tiles from shared memory
        for j in range(threadGridDim):
            tempSum += sharedMemA[threadY, j] * sharedMemB[j, threadX]

        cuda.syncthreads()
    C[row, col] = tempSum


def CudaSharedMemMatMul(A, B):

    # This part remains identical to previous GPU implementations.

    numRows = A.shape[0]
    numCols = B.shape[1]
    
    outArrayShape = (numRows, numCols)

    tic = perf_counter()
    # Initialize & transfer A and B to device memory (A_global_mem, B_global_mem)
    A_global_mem = cuda.to_device(A)
    B_global_mem = cuda.to_device(B)
    toc = perf_counter()

    C_global_mem = cuda.device_array(outArrayShape, dtype=np.float64)

    transfersTime = toc-tic

    # Create a grid of thread blocks and fire the SimpleMatMul2DKernel kernel. You should use a 2D grid (of 2D blocks)
    # The global threadGridDim is already defined as 32 at the top of the file.
    threadsPerBlock = (threadGridDim, threadGridDim)
    
    # In the provided kernel, (col, row) = cuda.grid(2), so x-dim maps to columns and y-dim to rows.
    blocksPerGrid_x = int(np.ceil(numCols / threadsPerBlock[0])) 
    blocksPerGrid_y = int(np.ceil(numRows / threadsPerBlock[1])) 
    blocksPerGrid = (blocksPerGrid_x, blocksPerGrid_y)
    
    # Invoke kernel
    SharedMemMatMul2DKernel[blocksPerGrid, threadsPerBlock](A_global_mem, B_global_mem, C_global_mem)

    C = C_global_mem.copy_to_host()

    return C, transfersTime

