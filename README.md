# BootCMatchGX
BootCMatch for multi-GPU systems.

Sparse solvers are fundamental components in reliable, high-performance scientific and engineering computing. In BootCMatchGX we make available an Algebraic MultiGrid (AMG) 
method for preconditioning algebraic linear systems Ax = b, where A is a symmetric positive definite (s.p.d.), large and sparse matrix. All the computational kernels for setup and application
of the AMG method, as preconditioner of an efficient version of the Conjugate Gradient Krylov solver, were designed and tuned for hybrid MPI-CUDA programming environments when multiple 
distributed nodes hosting Nvidia GPUs are available. The current release of the library includes also some communication-reduced variants of Conjugate Gradient solver and a sparse approximate inverse (AFSAI) as possible one-level preconditioner for the various variants of CG method. 

This software project has been partially supported by:

-TEXTAROSSA: Towards EXtreme scale Technologies and Accelerators for euROhpc hw/Sw Supercomputing Applications for exascale, a EuroHPC-JU Project, Horizon 2020 Program for Research and Innovation, 
funded by European Commission (EC), Project ID: 956831.

-EoCoE III: Energy Oriented Center of Excellence, Fostering the European Energy Transition with Exascale, a EuroHPC Project, Horizon Europe Program for Research and Innovation, funded by European Commission (EC), Project ID: 101144014.

-ICSC: the Italian Research Center on High-Performance Computing, Big Data and Quantum Computing, funded by MUR - Next Generation EU (NGEU).

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

### Lapack

Please, download and compile LAPACK before going on.
`BCMGX/Makefile` expects to find directory `../../../lapack-master`, but it can be changed by editing `BCMGX/config.mk`.

### Compilation

Compilation can be accomplished using GNU Make. 

```sh
cd BCMGX 
make
```

The process can be customized by editing `BCMGX/config.mk`.

## Solving 

The solver supports different running modes, that can be selected as follows:

```sh
Usage: mpirun -np <NPROCS> ./example/driverSolve [--matrix <FILE_NAME> | --laplacian <SIZE> | --laplacian-3d <FILE_NAME>] --settings <FILE_NAME>

        You can specify only one out of the three available options: --matrix, --laplacian-3d and --laplacian

        -a, --laplacian <SIZE>                      Generate a matrix whose size is <SIZE>^3.
        -B, --out-prefix <STRING>                   Use <PREFIX> when writing additional files to output dir.
        -d, --dump-matrix <FILE_NAME>               Write process-specific local input matrix to <FILE_NAME><PROC_ID>.
        -e, --errlog <FILE_NAME>                    Write process-specific log to <FILE_NAME><PROC_ID>.
        -g, --laplacian-3d-generator [ 7p | 27p ]   Choose laplacian 3d generator (7 points or 27 points).
        -h, --help                                  Print this message.
        -i, --info <FILE_NAME>                      Write info to <FILE_NAME>.
        -l, --laplacian-3d <FILE_NAME>              Read generation parameters from file <FILE_NAME>.
        -m, --matrix <FILE_NAME>                    Read the matrix from file <FILE_NAME>.
        -M, --detailed-metrics <FILE_NAME>          Write process-specific detailed profile log to <FILE_NAME><PROC_ID>.
        -o, --out <FILE_NAME>                       Write solution to <FILE_NAME>.
        -O, --out-dir <DIR>                         Write additional files to <DIR>.
        -p, --summary-prof <FILE_NAME>              Write process-specific summary profile log to <FILE_NAME><PROC_ID>.
        -P, --detailed-prof <FILE_NAME>             Write process-specific detailed profile log to <FILE_NAME><PROC_ID>.
        -s, --settings <FILE_NAME>                  Read settings from file <FILE_NAME>.
        -S, --out-suffix <STRING>                   Use <SUFFIX> when writing additional files to output dir.
        -x, --extended-prof                         Write extended profile info inside the info-file.
```

Please, refer to the following directories in order to find sample input matrices
and configuration files:
* *src/test/data/mtx* contains various matrices in the *Matrix Market* format that can be used with the *--matrix* mode;
* *src/test/data/settings* contains various setting files in order to select
different solvers and/or preconditioners;
* *src/test/data/cfg* contains various configuration files in order to supply
parameters to the laplacian 3d matrix generator (both 7 and 27 points).

### Configuration file

The configuration file defines the preconditioning and solving procedure.
A sample (`sample.properties`) containing all the available options is located under `BCMGX/src/test/data/settings/`.

### More examples

In order to launch the complete suite of regression tests, please use the command (inside directory BCMGX):

    make regressionTests

Some default combinations of solver+preconditioner+input will be tested.
Please, find all supported options by invoking:

    make helpRegressionTests

Command line options can be used in order to select only the desired combinations:

    make regressionTests SOLVER="FCG" PRECONDITIONER="BCMG" TEST_CONFIG="lap3d_27p_50x50x50_1x1x1 lap3d_27p_50x50x50_2x1x1"

This make target does not stop by default in case of error. If you want to enable this behavior, you use STOP_ON_ERROR=1:

    make regressionTests TEST_CONFIG=lap3d_27p_50x50x50_2x2x2 STOP_ON_ERROR=1

Process-specific log files can be enabled by setting ENABLE_LOG=1.

CUDA-MemCheck can be enabled by setting USE_CUDA_MEMCHECK=1:

    make regressionTests USE_CUDA_MEMCHECK=1

### Important operations

The following three unit tests can be taken as examples for some basic sparse-matrix operations:
* BCMGX/src/example/driverSpMV.cu;
* BCMGX/src/example/driverSpMM.cu;
* BCMGX/src/example/driverMWM.cu.
