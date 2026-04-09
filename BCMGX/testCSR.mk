testCSR: \
	testTransposeDenseMatrix \
	testTransposeSparseMatrix \
	testTransposePrecondMatrix \
	testTransposePoissonMatrix

testTransposeLap3d27pMatrix: $(TARGETDIR)/example/collectMtx $(TARGETDIR)/test/testTranspose
	$(MKDIR) $(TESTOUTDIR)

	# @echo "Transposing matrix with 1 process"
	# $(MPIRUN) -np 1 $(CUDA_MEMCHECK) $(TARGETDIR)/test/testTranspose \
	# 	--matrix $(SOURCEDIR)/test/data/mtx/dense_matrix_10x10.mtx \
	# 	--log $(TESTOUTDIR)/testTransposeDenseMatrix_1 \
	# 	--out $(TESTOUTDIR)/T_dense_matrix_10x10

	# @echo "Processing transposed matrix (reordering rows and formatting output)"
	# rm -f $(TESTOUTDIR)/T_dense_matrix_10x10_mono.mtx
	# test -f $(TESTOUTDIR)/T_dense_matrix_10x10_0_1 && \
	# $(TARGETDIR)/example/collectMtx \
	# 	$(TESTOUTDIR)/T_dense_matrix_10x10_mono.mtx \
	# 	$(TESTOUTDIR)/T_dense_matrix_10x10_0_1

	@echo "Transposing matrix with 8 process, eventually forcing multiproc version"
	$(MPIRUN) -np 8 --oversubscribe $(CUDA_MEMCHECK) $(TARGETDIR)/test/testTranspose \
		--force-multiproc \
		-g 27p -l $(SOURCEDIR)/test/data/cfg/lap3d_10x10x10_2x2x2.cfg \
		--log $(TESTOUTDIR)/lap3d_10x10x10_2x2x2 \
		--out $(TESTOUTDIR)/T_lap3d_10x10x10_2x2x2

	@echo "Processing transposed matrix (collecting output from 8 processes, reordering rows and formatting output)"
	rm -f $(TESTOUTDIR)/T_lap3d_10x10x10_2x2x2_multi.mtx
	test -f $(TESTOUTDIR)/T_lap3d_10x10x10_2x2x2_0_8 && \
	$(TARGETDIR)/example/collectMtx \
		$(TESTOUTDIR)/T_lap3d_10x10x10_2x2x2_multi.mtx \
		$(TESTOUTDIR)/T_lap3d_10x10x10_2x2x2_*_8

	# @echo "Comparing transposed matrix computed with 1 process to transposed matrix computed with $(NPROCS) processes"
	# diff -y --suppress-common-lines \
	# 	$(TESTOUTDIR)/T_dense_matrix_10x10_mono.mtx \
	# 	$(TESTOUTDIR)/T_dense_matrix_10x10_multi.mtx
	
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

testAnorm: $(TARGETDIR)/test/testAnorm
	$(MPIRUN) -np 2 $(CUDA_MEMCHECK) $(TARGETDIR)/test/testAnorm

testCSRForEach: $(TARGETDIR)/test/testCSRForEach
	stdbuf -oL -eL \
	$(MPIRUN) -np 3 $(CUDA_MEMCHECK) $(TARGETDIR)/test/testCSRForEach

testDiagonalScaling: $(TARGETDIR)/test/testDiagonalScaling
	stdbuf -oL -eL \
	$(MPIRUN) -np 3 $(CUDA_MEMCHECK) $(TARGETDIR)/test/testDiagonalScaling

testMatrixMatrixProductSimple: $(TARGETDIR)/test/testMatrixMatrixProductSimple
	$(MKDIR) $(TESTOUTDIR)
	for np in 2 3 4; do \
		stdbuf -oL -eL \
		$(MPIRUN) -np $$np $(CUDA_MEMCHECK) $(TARGETDIR)/test/testMatrixMatrixProductSimple \
			-e $(TESTOUTDIR)/testMatrixMatrixProductSimple \
	; done
	
testMatrixMatrixProductLap3d: $(TARGETDIR)/test/testMatrixMatrixProductLap3d
	$(MKDIR) $(TESTOUTDIR)
	stdbuf -oL -eL \
	$(MPIRUN) -np 8 $(CUDA_MEMCHECK) $(TARGETDIR)/test/testMatrixMatrixProductLap3d \
		-e $(TESTOUTDIR)/testMatrixMatrixProductLap3d \
		-O $(TESTOUTDIR) \
		--nx1 10 --ny1 10 --nz1 10 --P1 2 --Q1 2 --R1 2 \
		--nx2 10 --ny2 10 --nz2 10 --P2 2 --Q2 2 --R2 2

testShrink: $(TARGETDIR)/test/testShrink
	$(MKDIR) $(TESTOUTDIR)
	stdbuf -oL -eL \
	$(MPIRUN) -np 2 $(TARGETDIR)/test/testShrink
	
# testMatrixMatrixProduct1pNsparse: $(TARGETDIR)/test/testMatrixMatrixProduct1pNsparse
# 	SPSP_LIB=CUSPARSE CUSPARSE_CHUNK_FRACTION=0.010 CUDA_LAUNCH_BLOCKING=1 stdbuf -oL -eL \
# 	$(CUDA_MEMCHECK) $(TARGETDIR)/test/testMatrixMatrixProduct1pNsparse \
# 		$(SOURCEDIR)/test/data/mtx/lap3d_7p_100x100x100_2x1x1_CGHS_BCMG_V_CYCLE_convratio0.6_hrc1_MULTIPLICATIVE_prerelaxsw4_postrelaxsw4_aggrsw1_maxlev39_relnuncoarse20_smoothed1_r1_Alocal32_id0_nprocs2.mtx \
# 		$(SOURCEDIR)/test/data/mtx/lap3d_7p_100x100x100_2x1x1_CGHS_BCMG_V_CYCLE_convratio0.6_hrc1_MULTIPLICATIVE_prerelaxsw4_postrelaxsw4_aggrsw1_maxlev39_relnuncoarse20_smoothed1_r1_Pfull32_id0_nprocs2.mtx

# testMatrixMatrixProduct1pNsparse: $(TARGETDIR)/test/testMatrixMatrixProduct1pNsparse
# 	SPSP_LIB=NSP CUDA_LAUNCH_BLOCKING=1 stdbuf -oL -eL \
# 	$(CUDA_MEMCHECK) $(TARGETDIR)/test/testMatrixMatrixProduct1pNsparse \
# 		$(SOURCEDIR)/test/data/mtx/lap3d_7p_100x100x100_2x1x1_CGHS_BCMG_V_CYCLE_convratio0.6_hrc1_MULTIPLICATIVE_prerelaxsw4_postrelaxsw4_aggrsw1_maxlev39_relnuncoarse20_smoothed1_r1_Alocal32_id0_nprocs2.mtx \
# 		$(SOURCEDIR)/test/data/mtx/lap3d_7p_100x100x100_2x1x1_CGHS_BCMG_V_CYCLE_convratio0.6_hrc1_MULTIPLICATIVE_prerelaxsw4_postrelaxsw4_aggrsw1_maxlev39_relnuncoarse20_smoothed1_r1_Pfull32_id0_nprocs2.mtx