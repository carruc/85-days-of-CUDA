# Data parallelism on heterogeneous systems

Data is the main cause of slowness in software execution. When partitions of data can be processed independently, we can employ **data parallelism** to speed up computation - this entails re-organizing tasks around partitions.

These partitions can be "atomic", such as a single channel of a single pixel, or seemingly global operations split in smaller computations.

Another approach to parallelism is **task parallelism**: splitting computation on the basis of task decomposition.

# CUDA C program structure

CUDA C extends C with some new syntax and libraries, easing massively parallel and heterogeneous computing.

The core paradigm shift is programming with CPU-GPU co-operation in mind. CUDA C source files can feature both serial *host* code and *device* (GPU) code:
- CUDA C functions are marked by special keywords and called *kernels*;
- execution starts with host code which in turn calls kernels;

When a kernel is called, a certain number of threads are launched - these are collectively called a *grid*. After the threads of a grid have finished, execution continues on the host.

## Threads

A thread is a simplistic model to describe execution inside CPU. It consists of:
- code of the program
- the point of the code being executed
- values of its variables and data structures
It also appears as a sequential list of operations, but [000] execution exists. 

`add 2.4, 2.5, 2.6, 2.8`

# Compiling

CUDA C can be compiled only by the NVIDIA C Compiler (**NVCC**).
1. NVCC takes as input CUDA extended C applications.
2. Splits CUDA and C code by checking for CUDA keywords.
3. Feeds host code to the host's C prepocessor/compiler/linker as usual;
4. At the same time, the device code is compiled by NVCC in virtual binaries called PTX files, which are then fed to the device's *just-in-time compiler*.

