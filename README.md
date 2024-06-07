# BootCMatchGX
BootCMatch for multi-GPU systems.

Sparse solvers are one of the building blocks of any technology for reliable and high-performance scientific and engineering computing. In BootCMatchGX we make available an Algebraic MultiGrid (AMG) 
method for preconditioning algebraic linear systems Ax = b, where A is a symmetric positive definite (s.p.d.), large and sparse matrix. All the computational kernels for setup and application
of the adaptive AMG method, as preconditioner of an efficient version of the Conjugate Gradient Krylov solver, were designed and tuned for hybrid MPI-CUDA programming environments when multiple 
distributed nodes hosting Nvidia GPUs are available.

This software project has been partially supported by:

-TEXTAROSSA: Towards EXtreme scale Technologies and Accelerators for euROhpc hw/Sw Supercomputing Applications for exascale, a EuroHPC-JU Project, Horizon 2020 Program for Research and Innovation, 
funded by European Commission (EC), Project ID: 956831.

-ICSC: the Italian Research Center on High-Performance Computing, Big Data and Quantum Computing, funded by MUR - Next Generation EU (NGEU)

Main reference:
-M. Bernaschi, A. Celestini, F. Vella and P. D'Ambra, A Multi-GPU Aggregation-Based AMG Preconditioner for Iterative Linear Solvers, in IEEE Transactions on Parallel and Distributed Systems, 
vol. 34, no. 8, pp. 2365-2376, Aug. 2023, doi: 10.1109/TPDS.2023.3287238. 

## Installation
### Dependencies and Requirements:

The software requires:
* GCC >= 8.0
* CUDA >= 12.3
* **[NSPARSE](https://github.com/EBD-CREST/nsparse)**: We included in this repository a slightly modified version of *NSPARSE*. This is located in *EXTERNAL*
* **[LAPACK][https://www.netlib.org/lapack/]** >= 3.12.0

Set the following variables inside the config.mk located in *BCMGX*:
* CUDA_DIR
* MPI_DIR
* CUDA_GPU_ARCH
* NSPARSE_GPU_ARCH
* MPI_INCLUDE_DIR
* LAPACK_LIB

### Lapack

Please, download and compile LAPACK before going on.

### Compilation

```sh
cd BCMGX 
make
```

## Solving 

The solver supports different running modes, that can be selected as follows:

```sh
Usage: ./bin/main [--matrix <FILE_NAME> | --laplacian <SIZE>] --settings <FILE_NAME> --info <FILE_NAME>

        You can specify only one out of the three available options: --matrix, --laplacian-3d and --laplacian

        -m, --matrix <FILE_NAME>                    Read the matrix from file <FILE_NAME>.
        -l, --laplacian-3d <FILE_NAME>              Read generation parameters from file <FILE_NAME>.
        -g, --laplacian-3d-generator [ 7p | 27p ]   Choose laplacian 3d generator (7 points or 27 points).
        -a, --laplacian <SIZE>                      Generate a matrix whose size is <SIZE>^3.
        -s, --settings <FILE_NAME>                  Read settings from file <FILE_NAME>.
        -e, --errlog <FILE_NAME>                    Write process-specific log to <FILE_NAME><PROC_ID>.
        -o, --out <FILE_NAME>                       Write solution to <FILE_NAME>.
        -i, --info <FILE_NAME>                      Write info to <FILE_NAME>.
```

Please, refer to the following directories in order to find sample input matrixes
and configuration files:
* *src/test/data/mtx* contains various matrixes in the *Matrix Market* format that can be used with the *--matrix* mode;
* *src/test/data/settings* contains various setting files in order to select
different solvers and/or preconditioners;
* *src/test/data/cfg* contains various configuration files in order to supply
parameters to the laplacian 3d matrix generator (both 7 and 27 points).

The following are three examples of how you can run the solver in the three different running modes using 1 MPI process:

```sh
mpirun -np 1 bin/main -m src/test/data/mtx/poisson_100x100.mtx -s src/test/data/settings/FCG_BCMG -i /tmp/out_info -o /tmp/out_solution

mpirun -np 1 bin/main -a 50 -s src/test/data/settings/FCG_BCMG -i /tmp/out_info -o /tmp/out_solution

mpirun -np 1 bin/main -g 27p -l src/test/data/cfg/lap3d_10x10x10_1x1x1.cfg -s src/test/data/settings/FCG_BCMG -i /tmp/out_info -o /tmp/out_solution
```

The following are two examples of how you can run the solver in the two different running modes using 2 MPI processes:

```sh
mpirun -np 2 bin/main -a 50 -s src/test/data/settings/FCG_BCMG -i /tmp/out_info -o /tmp/out_solution

mpirun -np 2 bin/main -g 27p -l src/test/data/cfg/lap3d_10x10x10_2x1x1.cfg -s src/test/data/settings/FCG_BCMG -i /tmp/out_info -o /tmp/out_solution
```

### Configuration file

The configuration file defines the preconditioning and solving procedure.

The configuration parameters are:

NONE            % rhs file NONE if not present

NONE            % sol file NONE if not present

FCG             % solver_type: CGHS | FCG

MULTIPLICATIVE  % bootstrap_type: MULTIPLICATIVE; NB: This is the composition rule when bootstrap is applied and more than 1 AMG hierarchy is setup

1               % max_hrc, in bootstrap AMG, max hierarchies; NB: Here put 1 for single AMG component

0.8             % desired convergence rate of the composite AMG; NB: This is not generally obtained if criterion on max_hrc is reached

SUITOR          % matchtype: SUITOR

2               % aggrsweeps; pairs aggregation steps. 0 pairs; 1 double pairs, etc ...

0               % aggr_type; 0 unsmoothed, 1 smoothed (not yet supported)

39              % max_levels; max number of levels built for the single hierarchy

V_CYCLE         % cycle_type: V_CYCLE

L1_JACOBI       % coarse_solver: L1_JACOBI

L1_JACOBI       % relax_type: L1_JACOBI

20              % relaxnumber_coarse

4               % prerelax_sweeps

4               % postrelax_sweeps

2000            % itnlim

1.e-6           % rtol

1               % stop criterion (0 absolute, 1 relative)

BCMG            % preconditioner: NONE | L1_JACOBI | BCMG

4               % l1-jacobi preconditioner iterations

1               % If 1 display norm

### More examples

In order to launch the complete suite of regression tests, please use the command (inside directory BCMGX):

    make regressionTests

The following combinations of solver+preconditioner+input will be tested:

    SOLVER         = CGHS FCG
    PRECONDITIONER = noPreconditioner l1Jacobi BCMG
    TEST_CONFIG    = lap3d_7p_a50 \
        lap3d_7p_50x50x50_1x1x1 lap3d_7p_50x50x50_2x1x1 \
        lap3d_7p_50x50x50_2x2x1 lap3d_7p_50x50x50_2x2x2 \
        lap3d_27p_50x50x50_1x1x1 lap3d_27p_50x50x50_2x1x1 \
        lap3d_27p_50x50x50_2x2x1 lap3d_27p_50x50x50_2x2x2

Command line options can be used in order to select only the desired combinations:

    make regressionTests SOLVER="FCG" PRECONDITIONER="BCMG" TEST_CONFIG="lap3d_27p_50x50x50_1x1x1 lap3d_27p_50x50x50_2x1x1"

This make target does not stop by default in case of error. If you want to enable this behavior, you use STOP_ON_ERROR=1:

    make regressionTests TEST_CONFIG=lap3d_27p_50x50x50_2x2x2 STOP_ON_ERROR=1

Process-specific log files can be enabled by setting ENABLE_LOG=1.

CUDA-MemCheck can be enabled by setting USE_CUDA_MEMCHECK=1:

    make regressionTests USE_CUDA_MEMCHECK=1

Please, find an updated version of the supported options by invoking:

    make helpRegressionTests
	
### Important operations

The following three unit tests can be taken as examples for important operations:
* BCMGX/src/test/testMatrixVectorProduct.cu;
* BCMGX/src/test/testMatrixMatrixProduct.cu;
* BCMGX/src/test/testMatrixMatching.cu.
