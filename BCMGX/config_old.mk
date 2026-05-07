HOSTNAME := $(shell hostname)

ifndef MAKEFLAGS
NUM_CORES := $(shell nproc 2>/dev/null || sysctl -n hw.ncpu)
MAKEFLAGS += -j$(NUM_CORES)
endif

# ===========================================================================
# CUDA_DIR
# ===========================================================================

ifeq ($(shell test -d /usr/local/cuda && echo -n yes),yes)
CUDA_DIR := /usr/local/cuda
endif
ifeq ($(shell test -d /usr/lib/nvidia-cuda-toolkit && echo -n yes),yes)
CUDA_DIR := /usr/lib/nvidia-cuda-toolkit
endif
ifeq ($(shell test -d /opt/share/libs/intel/nvidia/cuda-12.3.2 && echo -n yes),yes)
CUDA_DIR := /opt/share/libs/intel/nvidia/cuda-12.3.2
endif
ifeq ($(shell test -d /apps/SPACK/0.19.1/opt/linux-almalinux8-zen/gcc-8.5.0/cuda-12.4.1-n7tdo6v5wbgz7vuixtvyshytmgkm7cfa && echo -n yes),yes)
CUDA_DIR := /apps/SPACK/0.19.1/opt/linux-almalinux8-zen/gcc-8.5.0/cuda-12.4.1-n7tdo6v5wbgz7vuixtvyshytmgkm7cfa
endif
ifeq ($(shell test -d /leonardo/prod/opt/compilers/cuda/12.3/none && echo -n yes),yes)
CUDA_DIR := /leonardo/prod/opt/compilers/cuda/12.3/none
endif
ifeq ($(CUDA_DIR),)
$(error Could not find CUDA_DIR: please edit config.mk)
endif

# ===========================================================================
# MPI_DIR
# ===========================================================================

ifeq ($(shell test -d /usr/lib64/openmpi && echo -n yes),yes)
MPI_DIR         := /usr/lib64/openmpi
MPI_INCLUDE_DIR := /usr/include/openmpi-x86_64/
MPIRUN          := $(MPI_DIR)/bin/mpirun
endif
ifeq ($(shell test -d /usr/lib/x86_64-linux-gnu/openmpi && echo -n yes),yes)
MPI_DIR         := /usr/lib/x86_64-linux-gnu/openmpi
MPI_INCLUDE_DIR := $(MPI_DIR)/include
MPIRUN          := /usr/bin/mpirun
endif
ifeq ($(shell test -d /opt/share/comps/intel/gcc-12.2.1/openmpi-4.1.6 && echo -n yes),yes)
MPI_DIR         := /opt/share/comps/intel/gcc-12.2.1/openmpi-4.1.6
MPI_INCLUDE_DIR := $(MPI_DIR)/include
MPIRUN          := $(MPI_DIR)/bin/mpirun
endif
ifeq ($(shell test -d /apps/hpcx/2.11-gcc-inbox-gdrcopy2/ompi && echo -n yes),yes)
MPI_DIR         := /apps/hpcx/2.11-gcc-inbox-gdrcopy2/ompi
MPI_INCLUDE_DIR := $(MPI_DIR)/include
MPIRUN          := $(MPI_DIR)/bin/mpirun
endif
ifeq ($(shell test -d /leonardo/prod/opt/libraries/openmpi/4.1.6/gcc--12.2.0 && echo -n yes),yes)
MPI_DIR         := /leonardo/prod/opt/libraries/openmpi/4.1.6/gcc--12.2.0
MPI_INCLUDE_DIR := $(MPI_DIR)/include
MPIRUN          := $(MPI_DIR)/bin/mpirun
endif
ifeq ($(shell test -d /usr/local/lib/openmpi && echo -n yes),yes)
MPI_DIR         := /usr/local/lib/openmpi
MPI_INCLUDE_DIR := /usr/local/include/openmpi
MPIRUN          := $(MPI_DIR)/bin/mpirun
endif

ifeq ($(MPI_DIR),)
$(error Could not find MPI_DIR: please edit config.mk)
endif

# ===========================================================================
# GFORTRAN
# ===========================================================================

ifeq ($(shell test -d /usr/lib/x86_64-linux-gnu && echo -n yes),yes)
LIBS += -L/usr/lib/gcc/x86_64-linux-gnu/13
endif

# ===========================================================================
# LAPACK
# ===========================================================================

ifeq ($(shell test -d ../../../lapack-master && echo -n yes),yes)
LAPACK_LIB := ../../../lapack-master
endif
ifeq ($(shell test -d ../../lapack-master && echo -n yes),yes)
LAPACK_LIB := ../../lapack-master
endif
ifeq ($(shell test -d ../../../../InstalledSW/LAPACK/lapack-master && echo -n yes),yes)
LAPACK_LIB := ../../../../InstalledSW/LAPACK/lapack-master
endif
ifeq ($(shell test -d ../../../../InstalledSW/lapack-master && echo -n yes),yes)
LAPACK_LIB := ../../../../InstalledSW/lapack-master
endif

ifeq ($(LAPACK_LIB),)
$(error Could not find LAPACK_LIB: please edit config.mk)
endif

# ===========================================================================
# Order of preference for usage inside scalar_work:
# - LAPACK
# - Custom
# ===========================================================================

ifeq ($(LAPACK_LIB),)
# Could not find LAPACK
SW_USE_LIB := 0
else
# LAPACK found
SW_USE_LIB := 1
endif

USE_MKL := 1

# ===========================================================================
USE_CUDA_MEMCHECK = 0
USE_CUDA_PROFILER = 0
# ===========================================================================

CUDA_GPU_ARCH   = sm_80
CPP_STD         = -std=c++17

CC = gcc
NVCC = $(CUDA_DIR)/bin/nvcc
NVCC_FLAG = -DOMPI_SKIP_MPICXX $(CPP_STD) --extended-lambda # -Xptxas=-v
GPU_ARCH = -arch=$(CUDA_GPU_ARCH) -m64

LIBS += -lcuda -lcudart -lcublas -lcusolver -lcurand -L$(MPI_DIR)/lib -lmpi -lnvToolsExt -lnvidia-ml -lcusparse #-lpthread
INCLUDE = -Isrc -Isrc/nsp -I$(MPI_INCLUDE_DIR) -I$(NSP_PATH) -Iinclude #-I$(NSPARSE_PATH)/inc
DEFINE = #-DGENERAL_TRANSPOSE #-DDETAILED_TIMING

export LD_LIBRARY_PATH+=$(MPI_DIR)/lib

# NSPARSE_PATH = ../EXTERNAL/nsparse-master/cuda-c
# NSPARSE_GPU_ARCH = $(GPU_ARCH)

NSP_PATH = ../EXTERNAL/nsp-new
NSP_GPU_ARCH = $(GPU_ARCH)

# NUMDIFF = /home/giacomop/numdiff-5.9.0/numdiff
MKDIR := @mkdir -p
FORMATTER := clang-format -i
COPY := cp

LINTER           = clang-tidy --quiet -header-filter=.*
LINTER_FLAGS     = $(INCLUDE) --cuda-gpu-arch=$(CUDA_GPU_ARCH) --cuda-path=$(CUDA_DIR) $(CPP_STD)
LINTER_CHECKS    = -*,cert-*
LINTER_ISYSTEM   = -isystem $(CUDA_DIR)/include

#LINTER_CHECKS    = *,\
-altera-unroll-loops,\
-fuchsia-default-arguments-calls,\
-fuchsia-overloaded-operator,\
-llvmlibc-callee-namespace,\
-llvmlibc-implementation-in-namespace,\
-llvm-namespace-comment,\
-llvmlibc-restrict-system-libc-headers,\
-misc-include-cleaner,\
-modernize-pass-by-value,\
-modernize-use-trailing-return-type

ifeq ($(SW_USE_LIB),1)
DEFINE += -DSW_USE_LIB

# Use LAPACK+CBLAS: please, adjust the following variables according to your system
ifeq ($(LAPACK_LIB),)
$(error Could not find LAPACK_LIB: please edit config.mk)
endif

LAPACK_INC = $(LAPACK_LIB)/LAPACKE/include/
CBLAS_INC = $(LAPACK_LIB)/CBLAS/include/
CBLAS_LIB = $(LAPACK_LIB)

INCLUDE += -I${LAPACK_INC} -I${CBLAS_INC}
LIBS    += -L${LAPACK_LIB} -llapack -lrefblas -llapacke -lcblas -lgfortran
endif #ifeq ($(SW_USE_LIB),1)

#CUDA_MEMCHECK =
NVCC_OPT = -O3 $(CPP_STD) -DUSE_NVTX -Xcompiler -rdynamic -lineinfo
CC_OPT = -O3 -rdynamic -lineinfo

ifeq ($(USE_CUDA_MEMCHECK),1)
CUDA_MEMCHECK = $(CUDA_DIR)/bin/compute-sanitizer --tool memcheck
endif

ifeq ($(USE_CUDA_PROFILER),1)
CUDA_PROFILER = $(CUDA_DIR)/bin/nsys profile
endif

$(info    NVCC_OPT is $(NVCC_OPT))
$(info    NVCC_FLAG is $(NVCC_FLAG))
$(info    LIBS is $(LIBS))

NPROCS = 4

SOURCEDIR   := src
BUILDDIR    := obj
TARGETDIR   := bin
LIBDIR      := lib
LIBNAME     := gpu
TESTOUTDIR  := testout
PERCENT     := %

H_SRCS := $(shell find ./$(SOURCEDIR) -name "*.h")
CU_SRCS := $(shell find ./$(SOURCEDIR) -name "*.cu" ! -name "*.unused.cu")

