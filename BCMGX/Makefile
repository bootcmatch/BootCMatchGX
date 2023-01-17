CC = nvcc
NVCC_FLAG = -DOMPI_SKIP_MPICXX -std=c++14 
GPU_ARCH = -arch=sm_70 -m64
MPI_DIR=/usr/local
CUB_PATH = ../EXTERNAL/nsparse-master/cub-1.7.4/cub
NSPARSE_PATH = ../EXTERNAL/nsparse-master/cuda-c
NSPARSE_GPU_ARCH = -arch=sm_70 -m64
LIBS = -lcudart -lcusparse -lcublas -lcusolver -lcurand -L$(MPI_DIR)/lib -lmpi -lnvToolsExt
INCLUDE = -Isrc  -I$(MPI_DIR)/include -I$(NSPARSE_PATH)/inc -I$(CUB_PATH)
OPT = -O3 -std=c++14 -DUSE_NVTX

MAIN = sample_main.cu

BUILDDIR    := obj
TARGETDIR   := bin

all: $(TARGETDIR)/sample_main

OBJECTS = $(BUILDDIR)/matchingAggregation.o $(BUILDDIR)/relax.o $(BUILDDIR)/distribuite.o $(BUILDDIR)/GAMG_cycle.o $(BUILDDIR)/FCG.o $(BUILDDIR)/bcmgx.o $(BUILDDIR)/utils.o $(BUILDDIR)/CSR.o $(BUILDDIR)/AMG.o $(BUILDDIR)/matrixIO.o  $(BUILDDIR)/myMPI.o $(BUILDDIR)/nsparse.o  $(BUILDDIR)/spspmpi.o $(BUILDDIR)/handles.o $(BUILDDIR)/scalar.o $(BUILDDIR)/vector.o $(BUILDDIR)/solutionAggregator.o $(BUILDDIR)/getmct.o $(BUILDDIR)/local_permutation.o $(BUILDDIR)/custom_cudamalloc.o $(BUILDDIR)/extern.o

$(TARGETDIR)/sample_main: $(MAIN) $(OBJECTS)
	$(CC) -g $^ -o $@ $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $(LIBS) $(OPT)

$(BUILDDIR)/matchingAggregation.o: src/prec_setup/matchingAggregation.cu src/prec_setup/matchingPairAggregation.cu src/prec_setup/matching.cu src/prec_setup/suitor.cu src/utility/memoryPools.cu src/prec_setup/bootstrap.cu
	mkdir -p ${BUILDDIR} ${TARGETDIR}
	$(CC) -c -o $@ $(LIBS) $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) src/prec_setup/matchingAggregation.cu  $(OPT)

$(BUILDDIR)/relax.o: src/basic_kernel/smoother/relaxation.cu src/basic_kernel/smoother/relaxation_sm.cu
	mkdir -p ${BUILDDIR} ${TARGETDIR}
	$(CC) -c -o $@ $(LIBS) $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) src/basic_kernel/smoother/relaxation.cu $(OPT)
	
$(BUILDDIR)/distribuite.o: src/utility/distribuite.cu
	$(CC) -c -o $@ $(LIBS) $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $^ $(OPT)
	
$(BUILDDIR)/GAMG_cycle.o: src/prec_apply/GAMG_cycle.cu
	$(CC) -c -o $@ $(LIBS) $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $^ $(OPT)
	
$(BUILDDIR)/FCG.o: src/solver/FCG.cu
	mkdir -p ${BUILDDIR} ${TARGETDIR}
	$(CC) -c -o $@ $(LIBS) $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $^ $(OPT)

$(BUILDDIR)/bcmgx.o: src/solver/bcmgx.cu
	mkdir -p ${BUILDDIR} ${TARGETDIR}
	$(CC) -c -o $@ $(LIBS) $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $^ $(OPT)

$(BUILDDIR)/CSR.o: src/basic_kernel/matrix/CSR.cu
	$(CC) -c -o $@ $(LIBS) $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $^ $(OPT)

$(BUILDDIR)/scalar.o: src/basic_kernel/matrix/scalar.cu
	$(CC) -c -o $@ $(LIBS) $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $^ $(OPT)

$(BUILDDIR)/vector.o: src/basic_kernel/matrix/vector.cu
	$(CC) -c -o $@ $(LIBS) $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $^ $(OPT)

$(BUILDDIR)/utils.o: src/utility/utils.cu
	$(CC) -c -o $@ $(LIBS) $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $^ $(OPT)

$(BUILDDIR)/handles.o: src/utility/handles.cu
	$(CC) -c -o $@ $(LIBS) $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $^ $(OPT)

$(BUILDDIR)/matrixIO.o: src/basic_kernel/matrix/matrixIO.cu
	$(CC) -c -o $@ $(LIBS) $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $^ $(OPT)

$(BUILDDIR)/myMPI.o: src/utility/myMPI.cu
	$(CC) -c -o $@ $(LIBS) $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $^ $(OPT)

$(BUILDDIR)/AMG.o: src/prec_setup/AMG.cu
	$(CC) -c -o $@ $(LIBS) $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $^ $(OPT)

$(BUILDDIR)/getmct.o: src/prec_setup/spmspmMGPU/getmct.cu
	$(CC) -c -o $@ $(LIBS) $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $^ $(OPT)

$(BUILDDIR)/spspmpi.o: src/prec_setup/spmspmMGPU/spspmpi.cu
	$(CC) -c -o $@ $(LIBS) $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $^ $(OPT)

$(BUILDDIR)/solutionAggregator.o: src/basic_kernel/halo_communication/halo_communication.cu
	$(CC) -c -o $@ $(LIBS) $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $^ $(OPT)

$(BUILDDIR)/extern.o: src/basic_kernel/halo_communication/extern.cu
	$(CC) -c -o $@ $(LIBS) $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $^ $(OPT)

$(BUILDDIR)/local_permutation.o: src/basic_kernel/halo_communication/local_permutation.cu
	$(CC) -c -o $@ $(LIBS) $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $^ $(OPT)

$(BUILDDIR)/custom_cudamalloc.o: src/basic_kernel/custom_cudamalloc/custom_cudamalloc.cu
	$(CC) -c -o $@ $(LIBS) $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $^ $(OPT)
	
$(BUILDDIR)/nsparse.o: $(NSPARSE_PATH)/src/kernel/kernel_spgemm_hash_d.cu
	$(CC) -c -DDOUBLE -o $@ $(LIBS) $(INCLUDE) $(NSPARSE_GPU_ARCH) $(NVCC_FLAG) $^ $(OPT)
	
clean:
	rm $(BUILDDIR)/*.o $(TARGETDIR)/*
	