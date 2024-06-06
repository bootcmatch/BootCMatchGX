testCommunication: \
	testRequestLowerMissingRows \
	testRequestUpperMissingRows \
	testRequestBothMissingRows \
	testRequestBothMissingRowsWithColShift

testRequestLowerMissingRows: prepare $(TARGETDIR)/test/testRequestMissingRows
	$(MPIRUN) -np $(NPROCS) $(CUDA_MEMCHECK) $(TARGETDIR)/test/testRequestMissingRows \
		-m $(SOURCEDIR)/test/data/mtx/dense_matrix_10x10.mtx \
		-l $(TESTOUTDIR)/testRequestLowerMissingRows \
		-s lower \
		--verbose
	ls -l $(TESTOUTDIR)/testRequestLowerMissingRows*

testRequestUpperMissingRows: prepare $(TARGETDIR)/test/testRequestMissingRows
	$(MPIRUN) -np $(NPROCS) $(CUDA_MEMCHECK) $(TARGETDIR)/test/testRequestMissingRows \
		-m $(SOURCEDIR)/test/data/mtx/dense_matrix_10x10.mtx \
		-l $(TESTOUTDIR)/testRequestUpperMissingRows \
		-s upper \
		--verbose
	ls -l $(TESTOUTDIR)/testRequestUpperMissingRows*

testRequestBothMissingRows: prepare $(TARGETDIR)/test/testRequestMissingRows
	$(MPIRUN) -np $(NPROCS) $(CUDA_MEMCHECK) $(TARGETDIR)/test/testRequestMissingRows \
		-m $(SOURCEDIR)/test/data/mtx/dense_matrix_10x10.mtx \
		-l $(TESTOUTDIR)/testRequestBothMissingRows \
		-s both \
		--verbose
	ls -l $(TESTOUTDIR)/testRequestBothMissingRows_*

testRequestBothMissingRowsWithColShift: prepare $(TARGETDIR)/test/testRequestMissingRows
	$(MPIRUN) -np $(NPROCS) $(CUDA_MEMCHECK) $(TARGETDIR)/test/testRequestMissingRows \
		-m $(SOURCEDIR)/test/data/mtx/dense_matrix_10x10.mtx \
		-l $(TESTOUTDIR)/testRequestBothMissingRowsWithColShift \
		-s both \
		--col-shift \
		--verbose
	ls -l $(TESTOUTDIR)/testRequestBothMissingRowsWithColShift*
