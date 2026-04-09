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
CC = $(CUDA_DIR)/bin/nvcc
MKDIR := @mkdir -p
# ===========================================================================

GPU_ARCH = -arch=sm_80 #sm_75 doesn't work on m100, only sm_70. Both work on our other machine (fry)
NSPARSE_PATH = ./nsparse
NSP2_PATH = ./nsp
#NVCC_FLAG = --compiler-options -Wall --compiler-options -fPIC -std=c++14
NVCC_FLAG = -DOMPI_SKIP_MPICXX -std=c++14
ifndef NSPARSE_PATH
$(error NSPARSE_PATH not set)
endif
ifndef GPU_ARCH
$(error GPU_ARCH not set)
endif
# MPIDIR=/usr/mpi/gcc/openmpi-4.0.3rc4/include/openmpi
MPIDIR=/usr/include/openmpi-x86_64/
MPILIB=/usr/lib64/openmpi/
NSPARSE_GPU_ARCH = ${GPU_ARCH}
LIBS = -lcudart -lcusparse -lcublas -lcusolver -lcurand -L$(MPI_DIR)/lib -lmpi 
INCLUDE = -Isrc -I${MPI_INCLUDE_DIR}

OPT = -O3 -std=c++14

BUILDDIR    := obj
TARGETDIR   := bin

#all: $(TARGETDIR)/main $(TARGETDIR)/main_nsp $(TARGETDIR)/main_cusparse # $(TARGETDIR)/main_cusparse_mem
all: $(TARGETDIR)/main $(TARGETDIR)/main_nsp $(TARGETDIR)/main_cusparse
#all: $(TARGETDIR)/main_nsp

OBJECTS = $(BUILDDIR)/utils.o $(BUILDDIR)/CSR.o $(BUILDDIR)/matrixIO.o $(BUILDDIR)/myMPI.o $(BUILDDIR)/nsparse.o $(BUILDDIR)/spspmpi.o $(BUILDDIR)/handles.o ${BUILDDIR}/getmct.o ${BUILDDIR}/vector.o

NSP2_OBJECTS = $(BUILDDIR)/utils.o $(BUILDDIR)/CSR.o $(BUILDDIR)/matrixIO.o $(BUILDDIR)/myMPI.o $(BUILDDIR)/nsp.o $(BUILDDIR)/nsp_calc_val_sort_rows.o $(BUILDDIR)/nsp_spspmpi.o  $(BUILDDIR)/handles.o ${BUILDDIR}/getmct.o ${BUILDDIR}/vector.o

CUSPARSE_OBJ = $(BUILDDIR)/utils.o $(BUILDDIR)/CSR.o $(BUILDDIR)/matrixIO.o $(BUILDDIR)/myMPI.o $(BUILDDIR)/spgemmcusparse.o $(BUILDDIR)/spspmpicusparse.o $(BUILDDIR)/handles.o  ${BUILDDIR}/vector.o $(BUILDDIR)/compactcol.o

CUSPARSE_OBJ_MEM = $(BUILDDIR)/utils.o $(BUILDDIR)/CSR.o $(BUILDDIR)/matrixIO.o $(BUILDDIR)/myMPI.o $(BUILDDIR)/spgemmcusparse_mem.o $(BUILDDIR)/spspmpicusparse.o $(BUILDDIR)/handles.o $(BUILDDIR)/compactcol.o

# Vecchia versione di nsparse
$(TARGETDIR)/main: main.cu $(OBJECTS)
	$(MKDIR) $(@D)
	$(CC) -g $^ -UCSRSEG -UNSP2_NSPARSE -o $@ $(INCLUDE) -I$(NSPARSE_PATH)/cuda-c/inc $(GPU_ARCH) $(NVCC_FLAG) $(LIBS) $(OPT)

# Nuova versione di nsparse v2
$(TARGETDIR)/main_nsp: main.cu $(NSP2_OBJECTS)
	$(MKDIR) $(@D)
	$(CC) -g $^ -UCSRSEG -DNSP2_NSPARSE -o $@ $(INCLUDE) -I$(NSP2_PATH)/inc $(GPU_ARCH) $(NVCC_FLAG) $(LIBS) $(OPT)

# cusparse
$(TARGETDIR)/main_cusparse: main.cu $(CUSPARSE_OBJ)
	$(MKDIR) $(@D)
	$(CC) -g $^ -o $@ $(INCLUDE) -I$(NSPARSE_PATH)/cuda-c/inc $(GPU_ARCH)  $(LIBS) $(OPT)

# cusparse MEM
$(TARGETDIR)/main_cusparse_mem: main.cu $(CUSPARSE_OBJ_MEM)
	$(MKDIR) $(@D)
	$(CC) -g $^ -o $@ $(INCLUDE) -I$(NSPARSE_PATH)/cuda-c/inc $(GPU_ARCH)  $(LIBS) $(OPT)

$(BUILDDIR)/CSR.o: src/CSR.cu
	$(MKDIR) $(@D)
	$(CC) -c -o $@ $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $^ $(OPT)

$(BUILDDIR)/utils.o: src/utils.cu
	$(MKDIR) $(@D)
	$(CC) -c -o $@ $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $^ $(OPT)

$(BUILDDIR)/handles.o: src/handles.cu
	$(MKDIR) $(@D)
	$(CC) -c -o $@ $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $^ $(OPT)

$(BUILDDIR)/matrixIO.o: src/matrixIO.cu
	$(MKDIR) $(@D)
	$(CC) -c -o $@ $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $^ $(OPT)

$(BUILDDIR)/vector.o: src/vector.cu
	$(MKDIR) $(@D)
	$(CC) -c -o $@ $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $^ $(OPT)

$(BUILDDIR)/getmct.o: src/getmct.cu
	$(MKDIR) $(@D)
	$(CC) -c -o $@ $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $^ $(OPT)

$(BUILDDIR)/myMPI.o: src/myMPI.cu
	$(MKDIR) $(@D)
	$(CC) -c -o $@ $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $^ $(OPT)

$(BUILDDIR)/spspmpicusparse.o: src/spspmpicusparse.cu
	$(MKDIR) $(@D)
	$(CC) -c -o $@ $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $^ $(OPT) -I$(NSPARSE_PATH)/cuda-c/inc

$(BUILDDIR)/compactcol.o: src/compactcol.cu
	$(MKDIR) $(@D)
	$(CC) -c -o $@ $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $^ $(OPT)

$(BUILDDIR)/spgemmcusparse.o: src/spgemmcusparse.cu
	$(MKDIR) $(@D)
	$(CC) -c -o $@ $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $^ $(OPT)

$(BUILDDIR)/spgemmcusparse_mem.o: src/spgemmcusparse_mem.cu
	$(MKDIR) $(@D)
	$(CC) -c -o $@ $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $^ $(OPT)

####################################
# Vecchia versione di nsparse
$(BUILDDIR)/spspmpi.o: src/spspmpi.cu
	$(MKDIR) $(@D)
	$(CC) -c -UCSRSEG -UNSP2_NSPARSE -o $@ $(INCLUDE) -I$(NSPARSE_PATH)/cuda-c/inc $(GPU_ARCH) $(NVCC_FLAG) $^ $(OPT)

$(BUILDDIR)/nsparse.o: $(NSPARSE_PATH)/cuda-c/src/kernel/kernel_spgemm_hash_d.cu
	$(MKDIR) $(@D)
	$(CC) -c -DDOUBLE -o $@ $(LIBS) $(INCLUDE) -I$(NSPARSE_PATH)/cuda-c/inc ${GPU_ARCH} $(NVCC_FLAG) $^ $(OPT)

####################################
# Nuova versione di nsparse v2
$(BUILDDIR)/nsp_spspmpi.o: src/spspmpi.cu
	$(MKDIR) $(@D)
	$(CC) -c -UCSRSEG -DNSP2_NSPARSE -o $@ $(INCLUDE) -I$(NSP2_PATH)/inc $(GPU_ARCH) $(NVCC_FLAG) $^ $(OPT)

$(BUILDDIR)/nsp.o: $(NSP2_PATH)/src/nsp.cu
	$(MKDIR) $(@D)
	$(CC) -c -DDOUBLE -o $@ $(LIBS) $(INCLUDE) -I$(NSP2_PATH)/inc ${GPU_ARCH} $(NVCC_FLAG) $^ $(OPT)

$(BUILDDIR)/nsp_calc_val_sort_rows.o: $(NSP2_PATH)/src/nsp_calc_val_sort_rows.old.cu
	$(MKDIR) $(@D)
	$(CC) -c -DDOUBLE -o $@ $(LIBS) $(INCLUDE)  -I$(NSP2_PATH)/inc ${GPU_ARCH} $(NVCC_FLAG) $^ $(OPT)



clean:
	rm $(BUILDDIR)/*.o $(TARGETDIR)/* -f
