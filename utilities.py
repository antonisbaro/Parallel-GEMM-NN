import numpy as np
from Sequential.SequentialMatMul import SequentialMatMul
from Multiprocessing.MultiprocessingMatMul import MultiprocessingMatMul
from Multiprocessing.tools import ReleaseSharedMemory
from CudaGPU.CudaSharedMemMatMul import CudaSharedMemMatMul
from CudaGPU.CudaSimpleMatMul import CudaSimpleMatMul
from CudaGPU.CudaTransposeMatMul import CudaTransposeMatMul


def CustomMatmul(A, B, mode='numpy', numberOfProcesses=1):
    
    if mode == 'sequential':
        out = SequentialMatMul(A, B)
        return out, 0
    
    if mode == 'numpy' or len(A.shape) == 3:
        # print(A.shape, 'x', B.shape)
        return np.matmul(A, B), 0

    if mode == 'multiprocessing':
        out = MultiprocessingMatMul(A, B, numberOfProcesses=numberOfProcesses)
        return out

    if mode == 'multiprocessingOneRowPerProc':
        out = MultiprocessingMatMulOneRowPerProc(A, B, numberOfProcesses=numberOfProcesses)
        return out

    if mode == 'cudaSimple':
        out = CudaSimpleMatMul(A, B)
        return out

    if mode == 'cudaTranspose':
        out = CudaTransposeMatMul(A, B)
        return out    

    if mode == 'cudaShmem':
        out = CudaSharedMemMatMul(A, B)
        # if not np.allclose(out[0], np.matmul(A,B)):
        #     print(40*'-')
        #     print(mode)
        #     print('out', out[0])
        #     print(np.matmul(A,B))
        #     print(40*'-')
        return out
    else:
        raise Exception(f"Unknown MatMul mode: {mode}")


def load_data(path,datatype='float64'):
    with np.load(path) as f:
        x_train, y_train = f['x_train'].astype(datatype), f['y_train'].astype(datatype)
        x_test, y_test = f['x_test'].astype(datatype), f['y_test'].astype(datatype)
        return (x_train, y_train), (x_test, y_test)
