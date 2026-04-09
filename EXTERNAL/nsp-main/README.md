## Disclaimer

This is a test version of the code developed in this project and should not be considered for production use.

# nsp

The *nsp* library provides an implementation of the sparse matrix product that is based on *Computing the product between sparse matrices on multiple GPUs*

The code multiplies two sparse matrices stored in *CSR* format by splitting them between several GPUs and calculating the local products concurrently. The communication between GPUs is arranged via MPI library, each GPU corresponds to an MPI rank. In this version, the matrices are loaded from disk to device (GPU) through the host (CPU) where the matrices are split between all the MPI ranks. Each GPU computes its local product and stores the output on the device side. Therefore, the matrix product is completed as soon as the local products are finalized on each GPU card involved in the execution of the matrix product.

## Getting started

1. Linux platform (f.e. Ubuntu 18.04/20.04 or higher), MPI version 4.0.3 and CUDA version 11.3

2. Compile with *make*

3. To launch - modify and execute the *run* file


