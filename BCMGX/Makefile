CUDA_HOME = /usr/local/cuda-11.1
CC = $(CUDA_HOME)/bin/nvcc
NVCC_FLAG = -std=c++14 -DOMPI_SKIP_MPICXX
GPU_ARCH = -arch=sm_70
MPI_DIR = /usr/local
CUB_PATH = ../EXTERNAL/nsparse-master/cub-1.7.4/cub
NSPARSE_PATH = ../EXTERNAL/nsparse-master/cuda-c
NSPARSE_GPU_ARCH = -arch=sm_70
LIBS = -L$(CUDA_HOME)/lib64 -lcudart -lcusparse -lcublas -lcusolver -lcurand -L$(MPI_DIR)/lib -lmpi 
INCLUDE = -Isrc  -I$(MPI_DIR)/include -I$(NSPARSE_PATH)/inc -I$(CUB_PATH)
OPT = -O3

MAIN = bcmgx.cu

BUILDDIR    := obj
TARGETDIR   := bin

all: $(TARGETDIR)/bcmgx

OBJECTS = $(BUILDDIR)/matchingAggregation.o $(BUILDDIR)/solver.o $(BUILDDIR)/utils.o $(BUILDDIR)/CSR.o $(BUILDDIR)/AMG.o $(BUILDDIR)/matrixIO.o  $(BUILDDIR)/myMPI.o $(BUILDDIR)/nsparse.o  $(BUILDDIR)/spspmpi.o $(BUILDDIR)/handles.o $(BUILDDIR)/scalar.o $(BUILDDIR)/vector.o $(BUILDDIR)/solutionAggregator.o $(BUILDDIR)/getmct.o

$(TARGETDIR)/bcmgx: $(MAIN) $(OBJECTS)
	$(CC) -g $^ -o $@ $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $(LIBS) $(OPT)

$(BUILDDIR)/matchingAggregation.o: src/AMG/matchingAggregation.cu src/AMG/matchingPairAggregation.cu src/AMG/matching.cu src/AMG/suitor.cu src/utility/memoryPools.cu
	mkdir -p $(BUILDDIR) $(TARGETDIR)
	$(CC) -c -o $@ $(LIBS) $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) src/AMG/matchingAggregation.cu  $(OPT)

$(BUILDDIR)/solver.o: src/solver/solver.cu src/matrix/distribuite.cu src/solver/relaxation.cu src/solver/relaxation_sm.cu src/solver/GAMG_cycle.cu src/solver/FCG.cu src/solver/bootstrap.cu
	mkdir -p $(BUILDDIR) $(TARGETDIR)
	$(CC) -c -o $@ $(LIBS) $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) src/solver/solver.cu $(OPT)

$(BUILDDIR)/CSR.o: src/matrix/CSR.cu
	$(CC) -c -o $@ $(LIBS) $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $^ $(OPT)

$(BUILDDIR)/scalar.o: src/matrix/scalar.cu
	$(CC) -c -o $@ $(LIBS) $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $^ $(OPT)

$(BUILDDIR)/vector.o: src/matrix/vector.cu
	$(CC) -c -o $@ $(LIBS) $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $^ $(OPT)

$(BUILDDIR)/utils.o: src/utility/utils.cu
	$(CC) -c -o $@ $(LIBS) $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $^ $(OPT)

$(BUILDDIR)/handles.o: src/utility/handles.cu
	$(CC) -c -o $@ $(LIBS) $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $^ $(OPT)

$(BUILDDIR)/matrixIO.o: src/matrix/matrixIO.cu
	$(CC) -c -o $@ $(LIBS) $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $^ $(OPT)

$(BUILDDIR)/myMPI.o: src/utility/myMPI.cu
	$(CC) -c -o $@ $(LIBS) $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $^ $(OPT)

$(BUILDDIR)/AMG.o: src/AMG/AMG.cu
	$(CC) -c -o $@ $(LIBS) $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $^ $(OPT)

$(BUILDDIR)/getmct.o: src/spmspmMGPU/getmct.cu
	$(CC) -c -o $@ $(LIBS) $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $^ $(OPT)

$(BUILDDIR)/spspmpi.o: src/spmspmMGPU/spspmpi.cu
	$(CC) -c -o $@ $(LIBS) $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $^ $(OPT)

$(BUILDDIR)/solutionAggregator.o: src/solver/solutionAggregator.cu
	$(CC) -c -o $@ $(LIBS) $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $^ $(OPT)

$(BUILDDIR)/nsparse.o: $(NSPARSE_PATH)/src/kernel/kernel_spgemm_hash_d.cu
	$(CC) -c -DDOUBLE -o $@ $(LIBS) $(INCLUDE) $(NSPARSE_GPU_ARCH) $(NVCC_FLAG) $^ $(OPT)

clean:
	rm $(BUILDDIR)/*.o $(TARGETDIR)/*
