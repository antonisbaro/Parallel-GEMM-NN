from numba import cuda
import numpy as np
from time import perf_counter
from os import environ

environ["CUDA_VISIBLE_DEVICES"] = "1"

@cuda.jit("void(float64[:,:], float64[:,:],  float64[:,:])")
def SimpleMatMul2DKernel(A, B, C):
    # 2D thread grid
    # get thread coords from cuda.grid
    # (returns absolute position of the current thread in the entire grid of blocks)
    row, col = cuda.grid(2)
    
    # The code that thread (row,col) will execute (calculating C[row][col])
    # Boundary check: Ensure the thread is within the bounds of the output matrix C.
    # This is important when matrix dimensions are not a multiple of the block dimensions.
    if row < C.shape[0] and col < C.shape[1]:
        # Perform the dot product for C[row, col]
        tmp_sum = 0.0
        # Iterate over the inner dimension (columns of A or rows of B)
        for k in range(A.shape[1]):
            tmp_sum += A[row, k] * B[k, col]
        C[row, col] = tmp_sum
        
    return


def CudaSimpleMatMul(A, B):

    numRows = A.shape[0]
    numCols = B.shape[1]
    
    outArrayShape = (numRows, numCols)

    tic = perf_counter()
    # Initialize & transfer A and B to device memory (A_global_mem, B_global_mem) 
    A_global_mem = cuda.to_device(A)
    B_global_mem = cuda.to_device(B)
    toc = perf_counter()

    # Allocate device memory for the output matrix C
    C_global_mem = cuda.device_array(outArrayShape, dtype=np.float64)

    transfersTime = toc-tic

    # Create a grid of thread blocks and fire the SimpleMatMul2DKernel kernel. You should use a 2D grid (of 2D blocks)
    threadGridDim = 32
    threadsPerBlock = (threadGridDim, threadGridDim)

    # Calculate the number of blocks needed in each dimension.
    # We use ceiling division to ensure we have enough blocks to cover the entire matrix.

    blocksPerGrid_x = int(np.ceil(numRows / threadsPerBlock[0])) 
    blocksPerGrid_y = int(np.ceil(numCols / threadsPerBlock[1])) 
    blocksPerGrid = (blocksPerGrid_x, blocksPerGrid_y)
    
    # Invoke kernel
    # Launch the kernel with the specified grid and block configuration.
    SimpleMatMul2DKernel[blocksPerGrid, threadsPerBlock](A_global_mem, B_global_mem, C_global_mem)

    # Copy the result from device memory back to host memory
    C = C_global_mem.copy_to_host()

    return C, transfersTime
