testCSR: \
	testTransposeDenseMatrix \
	testTransposeDenseMatrixWithColShift \
	testTransposeSparseMatrix \
	testTransposePrecondMatrix \
	testTransposePrecondMatrixWithColShift \
	testTransposePoissonMatrix \
	testMatrixVectorProduct \
	testMatrixMatrixProduct \
	testMatrixMatching

testTransposeDenseMatrix: prepare $(TARGETDIR)/collectMtx $(TARGETDIR)/test/testTranspose
	@echo "Transposing matrix with 1 process"
	$(MPIRUN) -np 1 $(CUDA_MEMCHECK) $(TARGETDIR)/test/testTranspose \
		-m $(SOURCEDIR)/test/data/mtx/dense_matrix_10x10.mtx \
		-l $(TESTOUTDIR)/testTransposeDenseMatrix_1 \
		-o $(TESTOUTDIR)/T_dense_matrix_10x10

	@echo "Processing transposed matrix (reordering rows and formatting output)"
	rm -f $(TESTOUTDIR)/T_dense_matrix_10x10_mono.mtx
	test -f $(TESTOUTDIR)/T_dense_matrix_10x10_0_1 && \
	$(TARGETDIR)/collectMtx \
		$(TESTOUTDIR)/T_dense_matrix_10x10_mono.mtx \
		$(TESTOUTDIR)/T_dense_matrix_10x10_0_1

	@echo "Transposing matrix with $(NPROCS) process, eventually forcing multiproc version"
	$(MPIRUN) -np $(NPROCS) $(CUDA_MEMCHECK) $(TARGETDIR)/test/testTranspose \
		--force-multiproc \
		-m $(SOURCEDIR)/test/data/mtx/dense_matrix_10x10.mtx \
		-l $(TESTOUTDIR)/testTransposeDenseMatrix_$(NPROCS) \
		-o $(TESTOUTDIR)/T_dense_matrix_10x10

	@echo "Processing transposed matrix (collecting output from $(NPROCS) processes, reordering rows and formatting output)"
	rm -f $(TESTOUTDIR)/T_dense_matrix_10x10_multi.mtx
	test -f $(TESTOUTDIR)/T_dense_matrix_10x10_0_$(NPROCS) && \
	$(TARGETDIR)/collectMtx \
		$(TESTOUTDIR)/T_dense_matrix_10x10_multi.mtx \
		$(TESTOUTDIR)/T_dense_matrix_10x10_*_$(NPROCS)

	@echo "Comparing transposed matrix computed with 1 process to transposed matrix computed with $(NPROCS) processes"
	diff -y --suppress-common-lines \
		$(TESTOUTDIR)/T_dense_matrix_10x10_mono.mtx \
		$(TESTOUTDIR)/T_dense_matrix_10x10_multi.mtx

testTransposeDenseMatrixWithColShift: prepare $(TARGETDIR)/collectMtx $(TARGETDIR)/test/testTranspose
	@echo "Transposing matrix with 1 process"
	$(MPIRUN) -np 1 $(CUDA_MEMCHECK) $(TARGETDIR)/test/testTranspose \
		-m $(SOURCEDIR)/test/data/mtx/dense_matrix_10x10.mtx \
		-l $(TESTOUTDIR)/testTransposeDenseMatrixWithColShift_1 \
		-o $(TESTOUTDIR)/T_dense_matrix_with_col_shift_10x10 \
		--col-shift

	@echo "Processing transposed matrix (reordering rows and formatting output)"
	rm -f $(TESTOUTDIR)/T_dense_matrix_with_col_shift_10x10_mono.mtx
	test -f $(TESTOUTDIR)/T_dense_matrix_with_col_shift_10x10_0_1 && \
	$(TARGETDIR)/collectMtx \
		$(TESTOUTDIR)/T_dense_matrix_with_col_shift_10x10_mono.mtx \
		$(TESTOUTDIR)/T_dense_matrix_with_col_shift_10x10_0_1

	@echo "Transposing matrix with $(NPROCS) process, eventually forcing multiproc version"
	$(MPIRUN) -np $(NPROCS) $(CUDA_MEMCHECK) $(TARGETDIR)/test/testTranspose \
		--force-multiproc \
		-m $(SOURCEDIR)/test/data/mtx/dense_matrix_10x10.mtx \
		-l $(TESTOUTDIR)/testTransposeDenseMatrixWithColShift_$(NPROCS) \
		-o $(TESTOUTDIR)/T_dense_matrix_with_col_shift_10x10 \
		--col-shift

	@echo "Processing transposed matrix (collecting output from $(NPROCS) processes, reordering rows and formatting output)"
	rm -f $(TESTOUTDIR)/T_dense_matrix_with_col_shift_10x10_multi.mtx
	test -f $(TESTOUTDIR)/T_dense_matrix_with_col_shift_10x10_0_$(NPROCS) && \
	$(TARGETDIR)/collectMtx \
		$(TESTOUTDIR)/T_dense_matrix_with_col_shift_10x10_multi.mtx \
		$(TESTOUTDIR)/T_dense_matrix_with_col_shift_10x10_*_$(NPROCS)

	@echo "Comparing transposed matrix computed with 1 process to transposed matrix computed with $(NPROCS) processes"
	diff -y --suppress-common-lines \
		$(TESTOUTDIR)/T_dense_matrix_with_col_shift_10x10_mono.mtx \
		$(TESTOUTDIR)/T_dense_matrix_with_col_shift_10x10_multi.mtx

testTransposeSparseMatrix: prepare $(TARGETDIR)/collectMtx $(TARGETDIR)/test/testTranspose
	@echo "Transposing matrix with 1 process"
	$(MPIRUN) -np 1 $(CUDA_MEMCHECK) $(TARGETDIR)/test/testTranspose \
		-m $(SOURCEDIR)/test/data/mtx/sparse_matrix_10x10.mtx \
		-l $(TESTOUTDIR)/testTransposeSparseMatrix_1 \
		-o $(TESTOUTDIR)/T_sparse_matrix_10x10

	@echo "Processing transposed matrix (reordering rows and formatting output)"
	rm -f $(TESTOUTDIR)/T_sparse_matrix_10x10_mono.mtx
	test -f $(TESTOUTDIR)/T_sparse_matrix_10x10_0_1 && \
	$(TARGETDIR)/collectMtx \
		$(TESTOUTDIR)/T_sparse_matrix_10x10_mono.mtx \
		$(TESTOUTDIR)/T_sparse_matrix_10x10_0_1

	@echo "Transposing matrix with $(NPROCS) process, eventually forcing multiproc version"
	$(MPIRUN) -np $(NPROCS) $(CUDA_MEMCHECK) $(TARGETDIR)/test/testTranspose \
		--force-multiproc \
		-m $(SOURCEDIR)/test/data/mtx/sparse_matrix_10x10.mtx \
		-l $(TESTOUTDIR)/testTransposeSparseMatrix_$(NPROCS) \
		-o $(TESTOUTDIR)/T_sparse_matrix_10x10

	@echo "Processing transposed matrix (collecting output from $(NPROCS) processes, reordering rows and formatting output)"
	rm -f $(TESTOUTDIR)/T_sparse_matrix_10x10_multi.mtx
	test -f $(TESTOUTDIR)/T_sparse_matrix_10x10_0_$(NPROCS) && \
	$(TARGETDIR)/collectMtx \
		$(TESTOUTDIR)/T_sparse_matrix_10x10_multi.mtx \
		$(TESTOUTDIR)/T_sparse_matrix_10x10_*_$(NPROCS)

	@echo "Comparing transposed matrix computed with 1 process to transposed matrix computed with $(NPROCS) processes"
	diff -y --suppress-common-lines \
		$(TESTOUTDIR)/T_sparse_matrix_10x10_mono.mtx \
		$(TESTOUTDIR)/T_sparse_matrix_10x10_multi.mtx

testTransposePoissonMatrix: prepare $(TARGETDIR)/collectMtx $(TARGETDIR)/test/testTranspose
	@echo "Transposing matrix with 1 process"
	$(MPIRUN) -np 1 $(CUDA_MEMCHECK) $(TARGETDIR)/test/testTranspose \
		-m $(SOURCEDIR)/test/data/mtx/poisson_100x100.mtx \
		-l $(TESTOUTDIR)/testTransposePoissonMatrix_1 \
		-o $(TESTOUTDIR)/T_poisson_100x100

	@echo "Processing transposed matrix (reordering rows and formatting output)"
	rm -f $(TESTOUTDIR)/T_poisson_100x100_mono.mtx
	test -f $(TESTOUTDIR)/T_poisson_100x100_0_1 && \
	$(TARGETDIR)/collectMtx \
		$(TESTOUTDIR)/T_poisson_100x100_mono.mtx \
		$(TESTOUTDIR)/T_poisson_100x100_0_1

	@echo "Transposing matrix with $(NPROCS) process, eventually forcing multiproc version"
	$(MPIRUN) -np $(NPROCS) $(CUDA_MEMCHECK) $(TARGETDIR)/test/testTranspose \
		--force-multiproc \
		-m $(SOURCEDIR)/test/data/mtx/poisson_100x100.mtx \
		-l $(TESTOUTDIR)/testTransposePoissonMatrix_$(NPROCS) \
		-o $(TESTOUTDIR)/T_poisson_100x100

	@echo "Processing transposed matrix (collecting output from $(NPROCS) processes, reordering rows and formatting output)"
	rm -f $(TESTOUTDIR)/T_poisson_100x100_multi.mtx
	test -f $(TESTOUTDIR)/T_poisson_100x100_0_$(NPROCS) && \
	$(TARGETDIR)/collectMtx \
		$(TESTOUTDIR)/T_poisson_100x100_multi.mtx \
		$(TESTOUTDIR)/T_poisson_100x100_*_$(NPROCS)

	@echo "Comparing transposed matrix computed with 1 process to transposed matrix computed with $(NPROCS) processes"
	diff -y --suppress-common-lines \
		$(TESTOUTDIR)/T_poisson_100x100_mono.mtx \
		$(TESTOUTDIR)/T_poisson_100x100_multi.mtx

testTransposePrecondMatrix: prepare $(TARGETDIR)/collectMtx $(TARGETDIR)/test/testTranspose
	@echo "Transposing matrix with 1 process"
	$(MPIRUN) -np 1 $(CUDA_MEMCHECK) $(TARGETDIR)/test/testTranspose \
		-m $(SOURCEDIR)/test/data/mtx/precond.mtx \
		-l $(TESTOUTDIR)/testTransposePrecondMatrix_1 \
		-o $(TESTOUTDIR)/T_precond

	@echo "Processing transposed matrix (reordering rows and formatting output)"
	rm -f $(TESTOUTDIR)/T_precond_mono.mtx
	test -f $(TESTOUTDIR)/T_precond_0_1 && \
	$(TARGETDIR)/collectMtx \
		$(TESTOUTDIR)/T_precond_mono.mtx \
		$(TESTOUTDIR)/T_precond_0_1

	@echo "Transposing matrix with $(NPROCS) process, eventually forcing multiproc version"
	$(MPIRUN) -np $(NPROCS) $(CUDA_MEMCHECK) $(TARGETDIR)/test/testTranspose \
		--force-multiproc \
		-m $(SOURCEDIR)/test/data/mtx/precond.mtx \
		-l $(TESTOUTDIR)/testTransposePrecondMatrix_$(NPROCS) \
		-o $(TESTOUTDIR)/T_precond

	@echo "Processing transposed matrix (collecting output from $(NPROCS) processes, reordering rows and formatting output)"
	rm -f $(TESTOUTDIR)/T_precond_multi.mtx
	test -f $(TESTOUTDIR)/T_precond_0_$(NPROCS) && \
	$(TARGETDIR)/collectMtx \
		$(TESTOUTDIR)/T_precond_multi.mtx \
		$(TESTOUTDIR)/T_precond_*_$(NPROCS)

	@echo "Comparing transposed matrix computed with 1 process to transposed matrix computed with $(NPROCS) processes"
	diff -y --suppress-common-lines \
		$(TESTOUTDIR)/T_precond_mono.mtx \
		$(TESTOUTDIR)/T_precond_multi.mtx

testTransposePrecondMatrixWithColShift: prepare $(TARGETDIR)/collectMtx $(TARGETDIR)/test/testTranspose
	@echo "Transposing matrix with 1 process"
	@$(MPIRUN) -np 1 $(CUDA_MEMCHECK) $(TARGETDIR)/test/testTranspose \
		-m $(SOURCEDIR)/test/data/mtx/precond.mtx \
		-l $(TESTOUTDIR)/testTransposePrecondMatrixWithColShift_1 \
		-o $(TESTOUTDIR)/T_precondWithColShift \
		--col-shift

	@echo "Processing transposed matrix (reordering rows and formatting output)"
	rm -f $(TESTOUTDIR)/T_precondWithColShift_mono.mtx
	test -f $(TESTOUTDIR)/T_precondWithColShift_0_1 && \
	$(TARGETDIR)/collectMtx \
		$(TESTOUTDIR)/T_precondWithColShift_mono.mtx \
		$(TESTOUTDIR)/T_precondWithColShift_0_1

	@echo "Transposing matrix with $(NPROCS) process, eventually forcing multiproc version"
	$(MPIRUN) -np $(NPROCS) $(CUDA_MEMCHECK) $(TARGETDIR)/test/testTranspose \
		--force-multiproc \
		-m $(SOURCEDIR)/test/data/mtx/precond.mtx \
		-l $(TESTOUTDIR)/testTransposePrecondMatrixWithColShift_$(NPROCS) \
		-o $(TESTOUTDIR)/T_precondWithColShift \
		--col-shift

	@echo "Processing transposed matrix (collecting output from $(NPROCS) processes, reordering rows and formatting output)"
	rm -f $(TESTOUTDIR)/T_precondWithColShift_multi.mtx
	test -f $(TESTOUTDIR)/T_precondWithColShift_0_$(NPROCS) && \
	$(TARGETDIR)/collectMtx \
		$(TESTOUTDIR)/T_precondWithColShift_multi.mtx \
		$(TESTOUTDIR)/T_precondWithColShift_*_$(NPROCS)

	@echo "Comparing transposed matrix computed with 1 process to transposed matrix computed with $(NPROCS) processes"
	diff -y --suppress-common-lines \
		$(TESTOUTDIR)/T_precondWithColShift_mono.mtx \
		$(TESTOUTDIR)/T_precondWithColShift_multi.mtx

testMatrixVectorProduct: prepare $(TARGETDIR)/test/testMatrixVectorProduct
	$(MPIRUN) -np $(NPROCS) $(CUDA_MEMCHECK) $(TARGETDIR)/test/testMatrixVectorProduct \
		-m $(SOURCEDIR)/test/data/mtx/poisson_100x100.mtx \
		--time \
		--energy

testMatrixMatrixProduct: prepare $(TARGETDIR)/test/testMatrixMatrixProduct
	$(MPIRUN) -np $(NPROCS) $(CUDA_MEMCHECK) $(TARGETDIR)/test/testMatrixMatrixProduct \
		-m $(SOURCEDIR)/test/data/mtx/poisson_100x100.mtx \
		--time \
		--energy

testMatrixMatching: prepare $(TARGETDIR)/test/testMatrixMatching
	$(MPIRUN) -np $(NPROCS) $(CUDA_MEMCHECK) $(TARGETDIR)/test/testMatrixMatching \
		-m $(SOURCEDIR)/test/data/mtx/poisson_100x100.mtx \
		--time \
		--energy
		