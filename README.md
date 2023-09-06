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

The solver supports different running modes, that can be selected as follows:

```sh
Usage: sample_main [--matrix <FILE_NAME> | --laplacian <SIZE>] [--preconditioner <BOOL>] --settings <FILE_NAME>
       
       You can specify only one out of the available options: --matrix and --laplacian

	      -m, --matrix <FILE_NAME>         Read the matrix from file <FILE_NAME>.
	      -a, --laplacian <SIZE>           Generate a matrix whose size is <SIZE>^3.
	      -s, --settings <FILE_NAME>       Read settings from file <FILE_NAME>.
	      -p, --preconditioner <BOOL>      If 0 the preconditioner will not be applied, otherwise it will be applied. If the parameter 
	                                       is not passed on the command line the preconditioner will be applied.
```

The directory *test_matrix* contains a matrix in the *Matrix Market* format that can be used with the *--matrix* mode. Finally the file *AMGsettings* contains an example of a configuration file that can be used for the *--settings* option. 

The following are two examples of how you can run the solver in the three different running modes using 2 MPI processes:

```sh
mpirun -np 2 bin/sample_main -m ../test_matrix/poisson_100x100.mtx -s ../AMGsettings

mpirun -np 2 bin/sample_main -a 126 -s ../AMGsettings

```

### Configuration file

The configuration file defines the preconditioning and solving procedure.

The configuration parameters are:

0                  % bootstrap_type: 0 multiplicative, 1 symmetrized multi., 2 additive; NB: This is the composition rule when bootstrap is applied and more than 1 AMG hierarchy is setup (bootstrap is not yet supported, so only 1 AMG component is built and applied)

1                  % max_hrc, in bootstrap AMG, max hierarchies; NB: Here put 1 for single AMG component

0.8                % desired convergence rate of the composite AMG; NB: This is not generally obtained if criterion on max_hrc is reached

3                  % matchtype: 3 Suitor

2                  % aggrsweeps; pairs aggregation steps. 0 pairs; 1 double pairs, etc ...

0                  % aggr_type; 0 unsmoothed, 1 smoothed (not yet supported)

39                 % max_levels; max number of levels built for the single hierarchy

0                  % cycle_type: 0-Vcycle, 1-Hcycle, 2-Wcycle

4                  % coarse_solver: 4 l1-Jacobi

4                  % relax_type: 4 l1-Jacobi

20                 % relaxnumber_coarse

0                  % coarsesolver_type: 0 Distributed, 1 Replicated

4                  % prerelax_sweeps

4                  % postrelax_sweeps

1000               % itnlim

1.e-6              % rtol
 
An example of configuration file is given in *AMGsettings*
