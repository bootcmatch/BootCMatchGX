# BootCMatchGX
BootCMatch for multi-GPU systems.

Sparse solvers are one of the building blocks of any technology for reliable and high-performance scientific and engineering computing. In BootCMatchGX we make available an Algebraic MultiGrid (alpha-AMG) method for preconditioning algebraic linear systems Ax = b, where A is a symmetric positive definite (s.p.d.), large and sparse matrix. All the computational kernels for setup and application of the adaptive AMG method, as preconditioner of an efficient version of the Conjugate Gradient Krylov solver, were designed and tuned for hybrid MPI-CUDA programming environments when multiple distributed nodes hosting Nvidia GPUs are available.

## Installation
### Dependencies and Requirements:

The software requires:
* GCC >= 5.0
* CUDA >= 9.0
* **[CUB](https://nvlabs.github.io/cub/)**: we tested the code for the version: 1.7.4., but any newer version should work as well.
  * Install the software and setup the variable **CUB_PATH** in the BootCMatchGX's Makefile: e.g., CUB_PATH = ./cub-1.7.4 or CUB_PATH = . if you are using CUDA >= 11.0
* **[NSPARSE](https://github.com/EBD-CREST/nsparse)**: We included in this repository a slightly modified version *NSPARSE* that supports CUDA >= 9.0. This is located in *EXTERNAL*

Set the following variables inside the Makefile located in *BCMGX*:
* CUDA_HOME
* MPI_DIR
* CUB_PATH (Not needed for CUDA >= 11.0)
* GPU_ARCH
* NSPARSE_GPU_ARCH

### Compilation

```sh
cd BCMGX 
make all
```

## Solving 

The solver supports three different running modes, that can be selected as follows:

```sh
Usage: bcmgx [--matrix <FILE_NAME> | --laplacian-3d <FILE_NAME> | --laplacian <SIZE>] --settings <FILE_NAME>

        You can specify only one out of the three available options: --matrix, --laplacian-3d and --laplacian

        -m, --matrix <FILE_NAME>         Read the matrix from file <FILE_NAME> in "Matrix Market" format.
        -l, --laplacian-3d <FILE_NAME>   Read generation parameters from file <FILE_NAME>.
        -a, --laplacian <SIZE>           Generate a matrix whose size is <SIZE>^3.
        -s, --settings <FILE_NAME>       Read settings from file <FILE_NAME>.
```

The directory *src/cfg_files* contains three examples of configuration files that can be used with the *--laplacian-3d* mode. Each file is named as *lap3d_{N}.cfg* where the N denotes the number of processes that you want to run. The directory *test_matrix* contains a matrix in the *Matrix Market* format that can be used with the *--matrix* mode. Finally the file *AMGsettings* contains an example of a configuration file that can be used for the *--settings* option. 

The following are three examples of how you can run the solver in the three different running modes using 2 MPI processes:

```sh
mpirun -np 2 bin/bcmgx -m ../test_matrix/poisson_100x100.mtx -s ../AMGsettings

mpirun -np 2 bin/bcmgx -a 126 -s ../AMGsettings

mpirun -np 2 bin/bcmgx -l cfg_files/lap3d_2.cfg -s ../AMGsettings

```

### Configuration file

The configuration file defines the preconditioning and solving procedure.

The configuration parameters are:

* bootstrap_type: type of final AMG composition; 0 multiplicative, 1 symmetrized multi., 2 additive; NB: Here put 0 for single AMG component
* max_hrc: max number of hierarchies in the final bootstrap AMG; NB: Here put 1 for single AMG component
* rho: desired convergence rate of the composite AMG; NB: This is not generally obtained if criterion on max_hrc is reached
* matchtype: 3 Suitor
* aggrsweeps: pairwise aggregation steps; 0 pairs; 1 double pairs, etc ...
* aggr_type; 0 unsmoothed, 1 smoothed (not yet supported)
* max_levels: max number of levels built for the single hierarchy
* cycle_type: 0-Vcycle, 1-Hcycle, 2-Wcycle
* coarse_solver: 0 Jacobi, 1 FGS/BGS, 3 symmetrized GS, 4 l1-Jacobi
* relax_type: 0 Jacobi, 1 FGS/BGS, 3 symmetrized GS, 4 l1-Jacobi
* relaxnumber_coarse: number of iterations for the coarsest solver
* prerelax_sweeps: number of pre-smoother iterations at the intermediate levels
* postrelax_sweeps: number of post-smoother iterations at the intermediate levels
* itnlim: maximum number of iterations for the solver
* rtol: relative accuracy on the solution
* mem_alloc_size: memory size of the data structures used to exchange data between processes
 
An example of configuration file is given in *AMGsettings*
