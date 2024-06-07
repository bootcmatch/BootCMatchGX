CUDA_DIR        = /usr/local/cuda
MPI_DIR         = /usr/lib64/openmpi
CUDA_GPU_ARCH   = sm_80
MPI_INCLUDE_DIR = /usr/include/openmpi-x86_64/
CPP_STD         = -std=c++14

CC = gcc
NVCC = $(CUDA_DIR)/bin/nvcc
NVCC_FLAG = -DOMPI_SKIP_MPICXX $(CPP_STD)
GPU_ARCH = -arch=$(CUDA_GPU_ARCH) -m64


LIBS = -lcuda -lcudart -lcublas -lcusolver -lcurand -L$(MPI_DIR)/lib -lmpi -lnvToolsExt -lnvidia-ml -lpthread -L${LAPACK_LIB} -llapack -lrefblas -llapacke -lcblas -lgfortran #-lcusparse
INCLUDE = -Isrc  -I$(MPI_INCLUDE_DIR) -I$(NSPARSE_PATH)/inc -Iinclude -I${LAPACK_INC} -I${CBLAS_INC}

export LD_LIBRARY_PATH=$(MPI_DIR)/lib

NSPARSE_PATH = ../EXTERNAL/nsparse-master/cuda-c
NSPARSE_GPU_ARCH = $(GPU_ARCH)
MKDIR := @mkdir -p
FORMATTER := clang-format -i

LINTER           = clang-tidy --quiet -header-filter=.*
LINTER_FLAGS     = $(INCLUDE) --cuda-gpu-arch=$(CUDA_GPU_ARCH) --cuda-path=$(CUDA_DIR) $(CPP_STD)
LINTER_CHECKS    = -*,cert-*
LINTER_ISYSTEM   = -isystem /usr/local/cuda/include

LAPACK_LIB = ../../../lapack-master/
LAPACK_INC = $(LAPACK_LIB)/LAPACKE/include/
CBLAS_INC = $(LAPACK_LIB)/CBLAS/include/
CBLAS_LIB = $(LAPACK_LIB)

NVCC_OPT = -O3 -std=c++14 -DUSE_NVTX -Xcompiler -rdynamic -lineinfo
CC_OPT = -O3 -rdynamic -lineinfo

ifeq ($(USE_CUDA_MEMCHECK),1)
CUDA_MEMCHECK = /usr/local/cuda/bin/compute-sanitizer --tool memcheck
endif

NPROCS = 4

SOURCEDIR   := src
BUILDDIR    := obj
TARGETDIR   := bin
TESTOUTDIR  := testout

MPIRUN=$(MPI_DIR)/bin/mpirun

H_SRCS := $(shell find ./$(SOURCEDIR) -name "*.h")
CU_SRCS := $(shell find ./$(SOURCEDIR) -name "*.cu" ! -name "*.unused.cu")

