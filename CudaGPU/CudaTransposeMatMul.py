from numba import cuda
import numpy as np
from time import perf_counter
from os import environ

environ["CUDA_VISIBLE_DEVICES"] = "1"

@cuda.jit("void(float64[:,:], float64[:,:],  float64[:,:])")
def TransposeMatMul2DKernel(A, B, C):
    # 2D thread grid
    # get thread coords from cuda.grid
    # (returns absolute position of the current thread in the entire grid of blocks)
    
    # Use an inverse/transposed grid for threads
    # The grid's x-dimension is mapped to columns, and y-dimension to rows.
    # This is the opposite of the simple version, and is the conventional mapping.
    # It improves memory coalescing.
    col, row = cuda.grid(2)

    # The code that thread (col,row) will execute --> THE SAME with CudaSimpleMatMul
    # Boundary check to ensure the thread is within the matrix dimensions.
    if row < C.shape[0] and col < C.shape[1]:
        # Perform the dot product for C[row, col]
        tmp_sum = 0.0
        # Iterate over the inner dimension
        for k in range(A.shape[1]):
            tmp_sum += A[row, k] * B[k, col]
        C[row, col] = tmp_sum
        
    return

def CudaTransposeMatMul(A, B):
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
    threadGridDim = 32
    threadsPerBlock = (threadGridDim, threadGridDim)

    blocksPerGrid_x = int(np.ceil(numCols / threadsPerBlock[0])) 
    blocksPerGrid_y = int(np.ceil(numRows / threadsPerBlock[1]))
    blocksPerGrid = (blocksPerGrid_x, blocksPerGrid_y)
    
    # Invoke kernel
    TransposeMatMul2DKernel[blocksPerGrid, threadsPerBlock](A_global_mem, B_global_mem, C_global_mem)

    C = C_global_mem.copy_to_host()

    return C, transfersTime


