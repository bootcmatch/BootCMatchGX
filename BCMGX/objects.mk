# ==============================================================================
# $(TARGETDIR)/main
# ==============================================================================

OBJECTS = \
	$(BUILDDIR)/EXTERNAL/nsparse.o \
	$(BUILDDIR)/basic_kernel/halo_communication/extern.o \
	$(BUILDDIR)/basic_kernel/halo_communication/halo_communication.o \
	$(BUILDDIR)/basic_kernel/halo_communication/local_permutation.o \
	$(BUILDDIR)/basic_kernel/halo_communication/newoverlap.o \
	$(BUILDDIR)/custom_cudamalloc/custom_cudamalloc.o \
	$(BUILDDIR)/config/Params.o \
	$(BUILDDIR)/datastruct/CSR.o \
	$(BUILDDIR)/datastruct/matrixItem.o \
	$(BUILDDIR)/datastruct/scalar.o \
	$(BUILDDIR)/datastruct/vector.o \
	$(BUILDDIR)/generator/laplacian.o \
	$(BUILDDIR)/op/addAbsoluteRowSumNoDiag.o \
	$(BUILDDIR)/op/basic.o \
	$(BUILDDIR)/op/double_merged_axpy.o \
	$(BUILDDIR)/op/mydiag.o \
	$(BUILDDIR)/op/getmct.o \
	$(BUILDDIR)/op/spspmpi.o \
	$(BUILDDIR)/op/triple_inner_product.o \
	$(BUILDDIR)/preconditioner/bcmg/AMG.o \
	$(BUILDDIR)/preconditioner/bcmg/bcmg.o \
	$(BUILDDIR)/preconditioner/bcmg/BcmgPreconditionContext.o \
	$(BUILDDIR)/preconditioner/bcmg/bootstrap.o \
	$(BUILDDIR)/preconditioner/bcmg/GAMG_cycle.o \
	$(BUILDDIR)/preconditioner/bcmg/matching.o \
	$(BUILDDIR)/preconditioner/bcmg/matchingAggregation.o \
	$(BUILDDIR)/preconditioner/bcmg/matchingPairAggregation.o \
	$(BUILDDIR)/preconditioner/bcmg/suitor.o \
	$(BUILDDIR)/preconditioner/l1jacobi/l1jacobi.o \
	$(BUILDDIR)/preconditioner/prec_apply.o \
	$(BUILDDIR)/preconditioner/prec_finalize.o \
	$(BUILDDIR)/preconditioner/prec_setup.o \
	$(BUILDDIR)/solver/cghs/CG_HS.o \
	$(BUILDDIR)/solver/fcg/FCG.o \
	$(BUILDDIR)/solver/solve.o \
	$(BUILDDIR)/solver/SolverOut.o \
	$(BUILDDIR)/utility/assignDeviceToProcess.o \
	$(BUILDDIR)/utility/ColumnIndexSender.o \
	$(BUILDDIR)/utility/distribuite.o \
	$(BUILDDIR)/utility/globals.o \
	$(BUILDDIR)/utility/handles.o \
	$(BUILDDIR)/utility/MatrixItemSender.o \
	$(BUILDDIR)/utility/memoryPools.o \
	$(BUILDDIR)/utility/metrics.o \
	$(BUILDDIR)/utility/mpi.o \
	$(BUILDDIR)/utility/ProcessSelector.o \
	$(BUILDDIR)/utility/string.o \
	$(BUILDDIR)/utility/timing.o \
	$(BUILDDIR)/utility/utils.o

#	$(BUILDDIR)/op/LBfunctions.o \

# ==============================================================================
# src/EXTERNAL
# ==============================================================================

$(BUILDDIR)/EXTERNAL/nsparse.o: $(NSPARSE_PATH)/src/kernel/kernel_spgemm_hash_d.cu
	$(MKDIR) $(@D)
	$(NVCC) -c -DDOUBLE -o $@ $(LIBS) $(INCLUDE) $(NSPARSE_GPU_ARCH) $(NVCC_FLAG) $^ $(NVCC_OPT)


# ==============================================================================
# src/basic_kernel/halo_communication
# ==============================================================================

$(BUILDDIR)/basic_kernel/halo_communication/%.o: src/basic_kernel/halo_communication/%.cu
	$(MKDIR) $(@D)
	$(NVCC) -c -o $@ $(LIBS) $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $^ $(NVCC_OPT)

# ==============================================================================
# src/config
# ==============================================================================

$(BUILDDIR)/config/%.o: src/config/%.cu
	$(MKDIR) $(@D)
	$(NVCC) -c -o $@ $(LIBS) $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $^ $(NVCC_OPT)

# ==============================================================================
# src/custom_cudamalloc
# ==============================================================================

$(BUILDDIR)/custom_cudamalloc/%.o: src/custom_cudamalloc/%.cu
	$(MKDIR) $(@D)
	$(NVCC) -c -o $@ $(LIBS) $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $^ $(NVCC_OPT)

# ==============================================================================
# src/datastruct
# ==============================================================================

$(BUILDDIR)/datastruct/%.o: src/datastruct/%.cu
	$(MKDIR) $(@D)
	$(NVCC) -c -o $@ $(LIBS) $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $^ $(NVCC_OPT)

# ==============================================================================
# src/generator
# ==============================================================================

$(BUILDDIR)/generator/%.o: $(SOURCEDIR)/generator/%.cu
	$(MKDIR) $(@D)
	$(NVCC) -c -o $@ $(LIBS) $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $^ $(NVCC_OPT)

# ==============================================================================
# src/op
# ==============================================================================

$(BUILDDIR)/op/LBfunctions.o: src/op/LBfunctions.c
	$(MKDIR) $(@D)
	$(CC) -c -o $@ -I$(LAPACK_INC) -I$(CBLAS_INC) $^ $(CC_OPT)

$(BUILDDIR)/op/%.o: $(SOURCEDIR)/op/%.cu
	$(MKDIR) $(@D)
	$(NVCC) -c -o $@ $(LIBS) $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $^ $(NVCC_OPT)

# ==============================================================================
# src/preconditioner
# ==============================================================================

$(BUILDDIR)/preconditioner/%.o: src/preconditioner/%.cu 
	$(MKDIR) $(@D)
	$(NVCC) -c -o $@ $(LIBS) $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $^ $(NVCC_OPT)

# ==============================================================================
# src/preconditioner/bcmg
# ==============================================================================

$(BUILDDIR)/preconditioner/bcmg/%.o: src/preconditioner/bcmg/%.cu 
	$(MKDIR) $(@D)
	$(NVCC) -c -o $@ $(LIBS) $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $^ $(NVCC_OPT)

# ==============================================================================
# src/preconditioner/l1jacobi
# ==============================================================================

$(BUILDDIR)/preconditioner/l1jacobi/%.o: src/preconditioner/l1jacobi/%.cu 
	$(MKDIR) $(@D)
	$(NVCC) -c -o $@ $(LIBS) $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $^ $(NVCC_OPT)

# ==============================================================================
# src/solver
# ==============================================================================

$(BUILDDIR)/solver/%.o: src/solver/%.cu
	$(MKDIR) $(@D)
	$(NVCC) -c -o $@ $(LIBS) $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $^ $(NVCC_OPT)

# ==============================================================================
# src/utility
# ==============================================================================

$(BUILDDIR)/utility/%.o: $(SOURCEDIR)/utility/%.cu
	$(MKDIR) $(@D)
	$(NVCC) -c -o $@ $(LIBS) $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $^ $(NVCC_OPT)
	