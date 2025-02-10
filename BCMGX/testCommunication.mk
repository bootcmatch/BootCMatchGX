testCommunication: \
	testRequestLowerMissingRows \
	testRequestUpperMissingRows \
	testRequestBothMissingRows \
	testRequestBothMissingRowsWithColShift \
	testRequestAndCompatDenseMatrixForAFSAI_v1 \
	testRequestAndCompatSparseMatrixForAFSAI_v1 \
	testRequestAndCompatDenseMatrixForAFSAI_v2 \
	testRequestAndCompatSparseMatrixForAFSAI_v2

testRequestLowerMissingRows: $(TARGETDIR)/test/testRequestMissingRows
	$(MKDIR) $(TESTOUTDIR)
	$(MPIRUN) -np $(NPROCS) $(CUDA_MEMCHECK) $(TARGETDIR)/test/testRequestMissingRows \
		-m $(SOURCEDIR)/test/data/mtx/dense_matrix_10x10.mtx \
		-l $(TESTOUTDIR)/testRequestLowerMissingRows \
		-s lower \
		--verbose
	ls -l $(TESTOUTDIR)/testRequestLowerMissingRows*

testRequestUpperMissingRows: $(TARGETDIR)/test/testRequestMissingRows
	$(MKDIR) $(TESTOUTDIR)
	$(MPIRUN) -np $(NPROCS) $(CUDA_MEMCHECK) $(TARGETDIR)/test/testRequestMissingRows \
		-m $(SOURCEDIR)/test/data/mtx/dense_matrix_10x10.mtx \
		-l $(TESTOUTDIR)/testRequestUpperMissingRows \
		-s upper \
		--verbose
	ls -l $(TESTOUTDIR)/testRequestUpperMissingRows*

testRequestBothMissingRows: $(TARGETDIR)/test/testRequestMissingRows
	$(MKDIR) $(TESTOUTDIR)
	$(MPIRUN) -np $(NPROCS) $(CUDA_MEMCHECK) $(TARGETDIR)/test/testRequestMissingRows \
		-m $(SOURCEDIR)/test/data/mtx/dense_matrix_10x10.mtx \
		-l $(TESTOUTDIR)/testRequestBothMissingRows \
		-s both \
		--verbose
	ls -l $(TESTOUTDIR)/testRequestBothMissingRows_*

testRequestBothMissingRowsWithColShift: $(TARGETDIR)/test/testRequestMissingRows
	$(MKDIR) $(TESTOUTDIR)
	$(MPIRUN) -np $(NPROCS) $(CUDA_MEMCHECK) $(TARGETDIR)/test/testRequestMissingRows \
		-m $(SOURCEDIR)/test/data/mtx/dense_matrix_10x10.mtx \
		-l $(TESTOUTDIR)/testRequestBothMissingRowsWithColShift \
		-s both \
		--col-shift \
		--verbose
	ls -l $(TESTOUTDIR)/testRequestBothMissingRowsWithColShift*

testRequestAndCompatDenseMatrixForAFSAI_v1: $(TARGETDIR)/test/testRequestAndCompatForAFSAI_v1
	$(MKDIR) $(TESTOUTDIR)
	time -o $(TESTOUTDIR)/testRequestAndCompatDenseMatrixForAFSAI_v1_time \
		$(MPIRUN) -np $(NPROCS) $(CUDA_MEMCHECK) $(TARGETDIR)/test/testRequestAndCompatForAFSAI_v1 \
		-m $(SOURCEDIR)/test/data/mtx/dense_matrix_10x10.mtx \
		-l $(TESTOUTDIR)/testRequestAndCompatDenseMatrixForAFSAI_v1 \
		--verbose
	ls -l $(TESTOUTDIR)/testRequestAndCompatDenseMatrixForAFSAI_v1*

testRequestAndCompatSparseMatrixForAFSAI_v1: $(TARGETDIR)/test/testRequestAndCompatForAFSAI_v1
	$(MKDIR) $(TESTOUTDIR)
	time -o $(TESTOUTDIR)/testRequestAndCompatSparseMatrixForAFSAI_v1_time \
		$(MPIRUN) -np $(NPROCS) $(CUDA_MEMCHECK) $(TARGETDIR)/test/testRequestAndCompatForAFSAI_v1 \
		-m $(SOURCEDIR)/test/data/mtx/sparse_matrix_10x10.mtx \
		-l $(TESTOUTDIR)/testRequestAndCompatSparseMatrixForAFSAI_v1 \
		--verbose
	ls -l $(TESTOUTDIR)/testRequestAndCompatSparseMatrixForAFSAI_v1*

testRequestAndCompatDenseMatrixForAFSAI_v2: $(TARGETDIR)/test/testRequestAndCompatForAFSAI_v2
	$(MKDIR) $(TESTOUTDIR)
	time -o $(TESTOUTDIR)/testRequestAndCompatDenseMatrixForAFSAI_v2_time \
		$(MPIRUN) -np $(NPROCS) $(CUDA_MEMCHECK) $(TARGETDIR)/test/testRequestAndCompatForAFSAI_v2 \
		-m $(SOURCEDIR)/test/data/mtx/dense_matrix_10x10.mtx \
		-l $(TESTOUTDIR)/testRequestAndCompatDenseMatrixForAFSAI_v2 \
		--verbose
	ls -l $(TESTOUTDIR)/testRequestAndCompatDenseMatrixForAFSAI_v2*

testRequestAndCompatSparseMatrixForAFSAI_v2: $(TARGETDIR)/test/testRequestAndCompatForAFSAI_v2
	$(MKDIR) $(TESTOUTDIR)
	time -o $(TESTOUTDIR)/testRequestAndCompatSparseMatrixForAFSAI_v2_time \
		$(MPIRUN) -np $(NPROCS) $(CUDA_MEMCHECK) $(TARGETDIR)/test/testRequestAndCompatForAFSAI_v2 \
		-m $(SOURCEDIR)/test/data/mtx/sparse_matrix_10x10.mtx \
		-l $(TESTOUTDIR)/testRequestAndCompatSparseMatrixForAFSAI_v2 \
		--verbose
	ls -l $(TESTOUTDIR)/testRequestAndCompatSparseMatrixForAFSAI_v2*

