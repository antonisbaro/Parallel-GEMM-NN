# Accelerating Neural Network Training with CPU & GPU Parallelism

This repository contains the project for the "Parallel Computer Architectures for Machine Learning" course (NTUA). It provides an in-depth exploration of optimizing the General Matrix Multiply (GEMM) kernel, a critical computational bottleneck, to accelerate the training of a Multi-Layer Perceptron (MLP) on the MNIST dataset.

The project implements and benchmarks parallel solutions from scratch, targeting both multi-core CPU architectures and massively parallel NVIDIA GPUs, demonstrating a deep dive into hardware-aware programming.

---

## 🚀 Core Objectives & Features

-   **CPU Parallelization**:
    -   Implemented a data-parallel GEMM algorithm using Python's **`multiprocessing`** library to leverage multi-core processors.
    -   Utilized **`shared_memory`** to avoid costly data serialization overhead between processes.
    -   Conducted scalability analysis to measure speedup and identify the limitations imposed by Amdahl's Law.

-   **GPU Acceleration with CUDA & Numba**:
    -   Developed three progressively optimized GEMM kernels for an **NVIDIA Tesla V100 GPU**:
        -   **Naive Kernel**: A baseline implementation mapping one thread per output element.
        -   **Coalesced Memory Access Kernel**: Optimized memory access patterns by transposing the thread-grid mapping to align with how GPUs read from global memory.
        -   **Tiling with Shared Memory**: The most advanced kernel, which leverages fast on-chip shared memory to minimize slow global memory access, transforming the operation from **memory-bound** to **compute-bound**.

-   **End-to-End Neural Network Integration**:
    -   Integrated the custom-built GEMM kernels into a complete MLP training workflow.
    -   Analyzed the final training speedup under two distinct workload scenarios: a memory-bound "validation" scenario and a compute-bound "high-throughput" scenario.

---

## 🛠️ Execution & Environment

All experiments were executed on a high-performance computing node (2x Intel Xeon Silver 4114 CPUs, 1x NVIDIA Tesla V100 GPU) via the **Torque (`qsub`)** job scheduling system.

### Reproducing the Experiments

1.  **Run CPU Benchmarks**:
    ```bash
    qsub runMultiprocessingMatMul.sh
    ```
2.  **Run GPU GEMM Benchmarks**:
    ```bash
    qsub runCUDAMatMul.sh
    ```
3.  **Run Full Neural Network Training**:
    ```bash
    qsub runNNWithCustomMatMul.sh
    ```
4.  **Generate Plots**:
    The Jupyter Notebook located in `/plotFiles/Plot Scripts` can be executed to regenerate all plots from the resulting `.csv` files.

---

## 📈 Key Findings

-   **CPU vs. GPU**: While CPU parallelization provided significant speedup, it was limited by process management overhead. GPU parallelism, when properly optimized, delivered orders-of-magnitude higher performance.
-   **GPU Optimization Impact**: The **tiling technique using shared memory** proved to be the most effective strategy for large matrices, achieving a **~2.5x speedup** over a highly optimized NumPy library on the CPU.
-   **Workload-Dependent Performance**: The final analysis revealed that the optimal implementation depends heavily on the workload. For memory-bound tasks (many small operations), CPU-GPU data transfer overhead is the bottleneck. For compute-bound tasks (few large operations), the superior computational power of the tiled GPU kernel provides the greatest advantage.

---

## 💻 Technology Stack

-   **Languages/Libraries**: Python, NumPy, Numba
-   **Parallelism Paradigms**: CPU Multiprocessing, CUDA
-   **Hardware**: Intel Xeon Multi-core CPU, NVIDIA Tesla V100 GPU
-   **Job Scheduler**: Torque (`qsub`)

---

## ✍️ Authors

*   Manousos Linardakis
*   Lydia Ioanna Kolitsi
*   Antonios Barotsakis
*   Georgia Chatzigianni
