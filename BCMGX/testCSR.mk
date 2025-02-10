testCSR: \
	testTransposeDenseMatrix \
	testTransposeSparseMatrix \
	testTransposePrecondMatrix \
	testTransposePoissonMatrix
	#testTransposeDenseMatrixWithColShift \
	#testTransposePrecondMatrixWithColShift \
	#testMatrixVectorProduct \
	#testMatrixMatrixProduct \
	#testMatrixMatching

# testTransposeDenseMatrixNoGpu: $(TARGETDIR)/example/collectMtx $(TARGETDIR)/test/testTransposeNoGpu
# 	$(MKDIR) $(TESTOUTDIR)

# 	@echo "Transposing matrix with 1 process"
# 	$(MPIRUN) -np 1 $(CUDA_MEMCHECK) $(TARGETDIR)/test/testTransposeNoGpu \
# 		--matrix $(SOURCEDIR)/test/data/mtx/dense_matrix_10x10.mtx \
# 		--log $(TESTOUTDIR)/testTransposeDenseMatrix_1 \
# 		--out $(TESTOUTDIR)/T_dense_matrix_10x10

# 	@echo "Processing transposed matrix (reordering rows and formatting output)"
# 	rm -f $(TESTOUTDIR)/T_dense_matrix_10x10_mono.mtx
# 	test -f $(TESTOUTDIR)/T_dense_matrix_10x10_0_1 && \
# 	$(TARGETDIR)/example/collectMtx \
# 		$(TESTOUTDIR)/T_dense_matrix_10x10_mono.mtx \
# 		$(TESTOUTDIR)/T_dense_matrix_10x10_0_1

# 	@echo "Transposing matrix with $(NPROCS) process, eventually forcing multiproc version"
# 	$(MPIRUN) -np $(NPROCS) $(CUDA_MEMCHECK) $(TARGETDIR)/test/testTransposeNoGpu \
# 		--force-multiproc \
# 		--matrix $(SOURCEDIR)/test/data/mtx/dense_matrix_10x10.mtx \
# 		--log $(TESTOUTDIR)/testTransposeDenseMatrix_$(NPROCS) \
# 		--out $(TESTOUTDIR)/T_dense_matrix_10x10

# 	@echo "Processing transposed matrix (collecting output from $(NPROCS) processes, reordering rows and formatting output)"
# 	rm -f $(TESTOUTDIR)/T_dense_matrix_10x10_multi.mtx
# 	test -f $(TESTOUTDIR)/T_dense_matrix_10x10_0_$(NPROCS) && \
# 	$(TARGETDIR)/example/collectMtx \
# 		$(TESTOUTDIR)/T_dense_matrix_10x10_multi.mtx \
# 		$(TESTOUTDIR)/T_dense_matrix_10x10_*_$(NPROCS)

# 	@echo "Comparing transposed matrix computed with 1 process to transposed matrix computed with $(NPROCS) processes"
# 	diff -y --suppress-common-lines \
# 		$(TESTOUTDIR)/T_dense_matrix_10x10_mono.mtx \
# 		$(TESTOUTDIR)/T_dense_matrix_10x10_multi.mtx

testTransposeDenseMatrix: $(TARGETDIR)/example/collectMtx $(TARGETDIR)/test/testTranspose
	$(MKDIR) $(TESTOUTDIR)

	@echo "Transposing matrix with 1 process"
	$(MPIRUN) -np 1 $(CUDA_MEMCHECK) $(TARGETDIR)/test/testTranspose \
		--matrix $(SOURCEDIR)/test/data/mtx/dense_matrix_10x10.mtx \
		--log $(TESTOUTDIR)/testTransposeDenseMatrix_1 \
		--out $(TESTOUTDIR)/T_dense_matrix_10x10

	@echo "Processing transposed matrix (reordering rows and formatting output)"
	rm -f $(TESTOUTDIR)/T_dense_matrix_10x10_mono.mtx
	test -f $(TESTOUTDIR)/T_dense_matrix_10x10_0_1 && \
	$(TARGETDIR)/example/collectMtx \
		$(TESTOUTDIR)/T_dense_matrix_10x10_mono.mtx \
		$(TESTOUTDIR)/T_dense_matrix_10x10_0_1

	@echo "Transposing matrix with $(NPROCS) process, eventually forcing multiproc version"
	$(MPIRUN) -np $(NPROCS) $(CUDA_MEMCHECK) $(TARGETDIR)/test/testTranspose \
		--force-multiproc \
		--matrix $(SOURCEDIR)/test/data/mtx/dense_matrix_10x10.mtx \
		--log $(TESTOUTDIR)/testTransposeDenseMatrix_$(NPROCS) \
		--out $(TESTOUTDIR)/T_dense_matrix_10x10

	@echo "Processing transposed matrix (collecting output from $(NPROCS) processes, reordering rows and formatting output)"
	rm -f $(TESTOUTDIR)/T_dense_matrix_10x10_multi.mtx
	test -f $(TESTOUTDIR)/T_dense_matrix_10x10_0_$(NPROCS) && \
	$(TARGETDIR)/example/collectMtx \
		$(TESTOUTDIR)/T_dense_matrix_10x10_multi.mtx \
		$(TESTOUTDIR)/T_dense_matrix_10x10_*_$(NPROCS)

	@echo "Comparing transposed matrix computed with 1 process to transposed matrix computed with $(NPROCS) processes"
	diff -y --suppress-common-lines \
		$(TESTOUTDIR)/T_dense_matrix_10x10_mono.mtx \
		$(TESTOUTDIR)/T_dense_matrix_10x10_multi.mtx

# testTransposeDenseMatrixWithColShift: $(TARGETDIR)/example/collectMtx $(TARGETDIR)/test/testTranspose
# 	$(MKDIR) $(TESTOUTDIR)

# 	@echo "Transposing matrix with 1 process"
# 	$(MPIRUN) -np 1 $(CUDA_MEMCHECK) $(TARGETDIR)/test/testTranspose \
# 		--matrix $(SOURCEDIR)/test/data/mtx/dense_matrix_10x10.mtx \
# 		--log $(TESTOUTDIR)/testTransposeDenseMatrixWithColShift_1 \
# 		--out $(TESTOUTDIR)/T_dense_matrix_with_col_shift_10x10 \
# 		--col-shift

# 	@echo "Processing transposed matrix (reordering rows and formatting output)"
# 	rm -f $(TESTOUTDIR)/T_dense_matrix_with_col_shift_10x10_mono.mtx
# 	test -f $(TESTOUTDIR)/T_dense_matrix_with_col_shift_10x10_0_1 && \
# 	$(TARGETDIR)/example/collectMtx \
# 		$(TESTOUTDIR)/T_dense_matrix_with_col_shift_10x10_mono.mtx \
# 		$(TESTOUTDIR)/T_dense_matrix_with_col_shift_10x10_0_1

# 	@echo "Transposing matrix with $(NPROCS) process, eventually forcing multiproc version"
# 	$(MPIRUN) -np $(NPROCS) $(CUDA_MEMCHECK) $(TARGETDIR)/test/testTranspose \
# 		--force-multiproc \
# 		--matrix $(SOURCEDIR)/test/data/mtx/dense_matrix_10x10.mtx \
# 		--log $(TESTOUTDIR)/testTransposeDenseMatrixWithColShift_$(NPROCS) \
# 		--out $(TESTOUTDIR)/T_dense_matrix_with_col_shift_10x10 \
# 		--col-shift

# 	@echo "Processing transposed matrix (collecting output from $(NPROCS) processes, reordering rows and formatting output)"
# 	rm -f $(TESTOUTDIR)/T_dense_matrix_with_col_shift_10x10_multi.mtx
# 	test -f $(TESTOUTDIR)/T_dense_matrix_with_col_shift_10x10_0_$(NPROCS) && \
# 	$(TARGETDIR)/example/collectMtx \
# 		$(TESTOUTDIR)/T_dense_matrix_with_col_shift_10x10_multi.mtx \
# 		$(TESTOUTDIR)/T_dense_matrix_with_col_shift_10x10_*_$(NPROCS)

# 	@echo "Comparing transposed matrix computed with 1 process to transposed matrix computed with $(NPROCS) processes"
# 	diff -y --suppress-common-lines \
# 		$(TESTOUTDIR)/T_dense_matrix_with_col_shift_10x10_mono.mtx \
# 		$(TESTOUTDIR)/T_dense_matrix_with_col_shift_10x10_multi.mtx

testTransposeSparseMatrix: $(TARGETDIR)/example/collectMtx $(TARGETDIR)/test/testTranspose
	$(MKDIR) $(TESTOUTDIR)

	@echo "Transposing matrix with 1 process"
	$(MPIRUN) -np 1 $(CUDA_MEMCHECK) $(TARGETDIR)/test/testTranspose \
		--matrix $(SOURCEDIR)/test/data/mtx/sparse_matrix_10x10.mtx \
		--log $(TESTOUTDIR)/testTransposeSparseMatrix_1 \
		--out $(TESTOUTDIR)/T_sparse_matrix_10x10

	@echo "Processing transposed matrix (reordering rows and formatting output)"
	rm -f $(TESTOUTDIR)/T_sparse_matrix_10x10_mono.mtx
	test -f $(TESTOUTDIR)/T_sparse_matrix_10x10_0_1 && \
	$(TARGETDIR)/example/collectMtx \
		$(TESTOUTDIR)/T_sparse_matrix_10x10_mono.mtx \
		$(TESTOUTDIR)/T_sparse_matrix_10x10_0_1

	@echo "Transposing matrix with $(NPROCS) process, eventually forcing multiproc version"
	$(MPIRUN) -np $(NPROCS) $(CUDA_MEMCHECK) $(TARGETDIR)/test/testTranspose \
		--force-multiproc \
		--matrix $(SOURCEDIR)/test/data/mtx/sparse_matrix_10x10.mtx \
		--log $(TESTOUTDIR)/testTransposeSparseMatrix_$(NPROCS) \
		--out $(TESTOUTDIR)/T_sparse_matrix_10x10

	@echo "Processing transposed matrix (collecting output from $(NPROCS) processes, reordering rows and formatting output)"
	rm -f $(TESTOUTDIR)/T_sparse_matrix_10x10_multi.mtx
	test -f $(TESTOUTDIR)/T_sparse_matrix_10x10_0_$(NPROCS) && \
	$(TARGETDIR)/example/collectMtx \
		$(TESTOUTDIR)/T_sparse_matrix_10x10_multi.mtx \
		$(TESTOUTDIR)/T_sparse_matrix_10x10_*_$(NPROCS)

	@echo "Comparing transposed matrix computed with 1 process to transposed matrix computed with $(NPROCS) processes"
	diff -y --suppress-common-lines \
		$(TESTOUTDIR)/T_sparse_matrix_10x10_mono.mtx \
		$(TESTOUTDIR)/T_sparse_matrix_10x10_multi.mtx

testTransposePoissonMatrix: $(TARGETDIR)/example/collectMtx $(TARGETDIR)/test/testTranspose
	$(MKDIR) $(TESTOUTDIR)

	@echo "Transposing matrix with 1 process"
	$(MPIRUN) -np 1 $(CUDA_MEMCHECK) $(TARGETDIR)/test/testTranspose \
		--matrix $(SOURCEDIR)/test/data/mtx/poisson_100x100.mtx \
		--log $(TESTOUTDIR)/testTransposePoissonMatrix_1 \
		--out $(TESTOUTDIR)/T_poisson_100x100

	@echo "Processing transposed matrix (reordering rows and formatting output)"
	rm -f $(TESTOUTDIR)/T_poisson_100x100_mono.mtx
	test -f $(TESTOUTDIR)/T_poisson_100x100_0_1 && \
	$(TARGETDIR)/example/collectMtx \
		$(TESTOUTDIR)/T_poisson_100x100_mono.mtx \
		$(TESTOUTDIR)/T_poisson_100x100_0_1

	@echo "Transposing matrix with $(NPROCS) process, eventually forcing multiproc version"
	$(MPIRUN) -np $(NPROCS) $(CUDA_MEMCHECK) $(TARGETDIR)/test/testTranspose \
		--force-multiproc \
		--matrix $(SOURCEDIR)/test/data/mtx/poisson_100x100.mtx \
		--log $(TESTOUTDIR)/testTransposePoissonMatrix_$(NPROCS) \
		--out $(TESTOUTDIR)/T_poisson_100x100

	@echo "Processing transposed matrix (collecting output from $(NPROCS) processes, reordering rows and formatting output)"
	rm -f $(TESTOUTDIR)/T_poisson_100x100_multi.mtx
	test -f $(TESTOUTDIR)/T_poisson_100x100_0_$(NPROCS) && \
	$(TARGETDIR)/example/collectMtx \
		$(TESTOUTDIR)/T_poisson_100x100_multi.mtx \
		$(TESTOUTDIR)/T_poisson_100x100_*_$(NPROCS)

	@echo "Comparing transposed matrix computed with 1 process to transposed matrix computed with $(NPROCS) processes"
	diff -y --suppress-common-lines \
		$(TESTOUTDIR)/T_poisson_100x100_mono.mtx \
		$(TESTOUTDIR)/T_poisson_100x100_multi.mtx

testTransposePrecondMatrix: $(TARGETDIR)/example/collectMtx $(TARGETDIR)/test/testTranspose
	$(MKDIR) $(TESTOUTDIR)

	@echo "Transposing matrix with 1 process"
	$(MPIRUN) -np 1 $(CUDA_MEMCHECK) $(TARGETDIR)/test/testTranspose \
		--matrix $(SOURCEDIR)/test/data/mtx/precond.mtx \
		--log $(TESTOUTDIR)/testTransposePrecondMatrix_1 \
		--out $(TESTOUTDIR)/T_precond

	@echo "Processing transposed matrix (reordering rows and formatting output)"
	rm -f $(TESTOUTDIR)/T_precond_mono.mtx
	test -f $(TESTOUTDIR)/T_precond_0_1 && \
	$(TARGETDIR)/example/collectMtx \
		$(TESTOUTDIR)/T_precond_mono.mtx \
		$(TESTOUTDIR)/T_precond_0_1

	@echo "Transposing matrix with $(NPROCS) process, eventually forcing multiproc version"
	$(MPIRUN) -np $(NPROCS) $(CUDA_MEMCHECK) $(TARGETDIR)/test/testTranspose \
		--force-multiproc \
		--matrix $(SOURCEDIR)/test/data/mtx/precond.mtx \
		--log $(TESTOUTDIR)/testTransposePrecondMatrix_$(NPROCS) \
		--out $(TESTOUTDIR)/T_precond

	@echo "Processing transposed matrix (collecting output from $(NPROCS) processes, reordering rows and formatting output)"
	rm -f $(TESTOUTDIR)/T_precond_multi.mtx
	test -f $(TESTOUTDIR)/T_precond_0_$(NPROCS) && \
	$(TARGETDIR)/example/collectMtx \
		$(TESTOUTDIR)/T_precond_multi.mtx \
		$(TESTOUTDIR)/T_precond_*_$(NPROCS)

	@echo "Comparing transposed matrix computed with 1 process to transposed matrix computed with $(NPROCS) processes"
	diff -y --suppress-common-lines \
		$(TESTOUTDIR)/T_precond_mono.mtx \
		$(TESTOUTDIR)/T_precond_multi.mtx

# testTransposePrecondMatrixWithColShift: $(TARGETDIR)/example/collectMtx $(TARGETDIR)/test/testTranspose
# 	$(MKDIR) $(TESTOUTDIR)

# 	@echo "Transposing matrix with 1 process"
# 	@$(MPIRUN) -np 1 $(CUDA_MEMCHECK) $(TARGETDIR)/test/testTranspose \
# 		--matrix $(SOURCEDIR)/test/data/mtx/precond.mtx \
# 		--log $(TESTOUTDIR)/testTransposePrecondMatrixWithColShift_1 \
# 		--out $(TESTOUTDIR)/T_precondWithColShift \
# 		--col-shift

# 	@echo "Processing transposed matrix (reordering rows and formatting output)"
# 	rm -f $(TESTOUTDIR)/T_precondWithColShift_mono.mtx
# 	test -f $(TESTOUTDIR)/T_precondWithColShift_0_1 && \
# 	$(TARGETDIR)/example/collectMtx \
# 		$(TESTOUTDIR)/T_precondWithColShift_mono.mtx \
# 		$(TESTOUTDIR)/T_precondWithColShift_0_1

# 	@echo "Transposing matrix with $(NPROCS) process, eventually forcing multiproc version"
# 	$(MPIRUN) -np $(NPROCS) $(CUDA_MEMCHECK) $(TARGETDIR)/test/testTranspose \
# 		--force-multiproc \
# 		--matrix $(SOURCEDIR)/test/data/mtx/precond.mtx \
# 		--log $(TESTOUTDIR)/testTransposePrecondMatrixWithColShift_$(NPROCS) \
# 		--out $(TESTOUTDIR)/T_precondWithColShift \
# 		--col-shift

# 	@echo "Processing transposed matrix (collecting output from $(NPROCS) processes, reordering rows and formatting output)"
# 	rm -f $(TESTOUTDIR)/T_precondWithColShift_multi.mtx
# 	test -f $(TESTOUTDIR)/T_precondWithColShift_0_$(NPROCS) && \
# 	$(TARGETDIR)/example/collectMtx \
# 		$(TESTOUTDIR)/T_precondWithColShift_multi.mtx \
# 		$(TESTOUTDIR)/T_precondWithColShift_*_$(NPROCS)

# 	@echo "Comparing transposed matrix computed with 1 process to transposed matrix computed with $(NPROCS) processes"
# 	diff -y --suppress-common-lines \
# 		$(TESTOUTDIR)/T_precondWithColShift_mono.mtx \
# 		$(TESTOUTDIR)/T_precondWithColShift_multi.mtx

#testMatrixVectorProduct: $(TARGETDIR)/test/testMatrixVectorProduct
#	$(MKDIR) $(TESTOUTDIR)
#	$(MPIRUN) -np $(NPROCS) $(CUDA_MEMCHECK) $(TARGETDIR)/test/testMatrixVectorProduct \
#		--matrix $(SOURCEDIR)/test/data/mtx/poisson_100x100.mtx \
#		--time \
#		--energy

#testMatrixMatrixProduct: $(TARGETDIR)/test/testMatrixMatrixProduct
#	$(MKDIR) $(TESTOUTDIR)
#	$(MPIRUN) -np $(NPROCS) $(CUDA_MEMCHECK) $(TARGETDIR)/test/testMatrixMatrixProduct \
#		--matrix $(SOURCEDIR)/test/data/mtx/poisson_100x100.mtx \
#		--time \
#		--energy

#testMatrixMatching: $(TARGETDIR)/test/testMatrixMatching
#	$(MKDIR) $(TESTOUTDIR)
#	$(MPIRUN) -np $(NPROCS) $(CUDA_MEMCHECK) $(TARGETDIR)/test/testMatrixMatching \
#		--matrix $(SOURCEDIR)/test/data/mtx/poisson_100x100.mtx \
#		--time \
#		--energy
		
