testGenerators: testGenerateLaplacian3D

lap3d_nx=10
lap3d_ny=10
lap3d_nz=10
lap3d_P=2
lap3d_Q=2
lap3d_R=2
lap3d_nproc=8
lap3d_gen=27p

testGenerateLaplacian3D: prepare $(TARGETDIR)/collectMtx $(TARGETDIR)/test/testGenerateLaplacian3D
	@echo "Fixing order (input)"
	rm -f $(TESTOUTDIR)/lap3d_$(lap3d_gen)_$(lap3d_nx)x$(lap3d_ny)x$(lap3d_nz)_$(lap3d_P)x$(lap3d_Q)x$(lap3d_R)_benchmark.mtx
	$(TARGETDIR)/collectMtx \
		$(TESTOUTDIR)/lap3d_$(lap3d_gen)_$(lap3d_nx)x$(lap3d_ny)x$(lap3d_nz)_$(lap3d_P)x$(lap3d_Q)x$(lap3d_R)_benchmark.mtx \
		$(SOURCEDIR)/test/data/mtx/lap3d_$(lap3d_gen)_$(lap3d_nx)x$(lap3d_ny)x$(lap3d_nz)_$(lap3d_P)x$(lap3d_Q)x$(lap3d_R).mtx

	@echo "Generating laplacian"
	rm -f $(TESTOUTDIR)/lap3d_$(lap3d_gen)_$(lap3d_nx)x$(lap3d_ny)x$(lap3d_nz)_$(lap3d_P)x$(lap3d_Q)x$(lap3d_R)_log_*
	rm -f $(TESTOUTDIR)/lap3d_$(lap3d_gen)_$(lap3d_nx)x$(lap3d_ny)x$(lap3d_nz)_$(lap3d_P)x$(lap3d_Q)x$(lap3d_R)_*_$(lap3d_nproc)
	$(MPIRUN) --oversubscribe -np $(lap3d_nproc) $(CUDA_MEMCHECK) $(TARGETDIR)/test/testGenerateLaplacian3D \
		-o $(TESTOUTDIR)/lap3d_$(lap3d_gen)_$(lap3d_nx)x$(lap3d_ny)x$(lap3d_nz)_$(lap3d_P)x$(lap3d_Q)x$(lap3d_R) \
		-g $(lap3d_gen) -x $(lap3d_nx) -y $(lap3d_ny) -z $(lap3d_nz) -P $(lap3d_P) -Q $(lap3d_Q) -R $(lap3d_R) \

	@echo "Collecting results"
	test -f $(TESTOUTDIR)/lap3d_$(lap3d_gen)_$(lap3d_nx)x$(lap3d_ny)x$(lap3d_nz)_$(lap3d_P)x$(lap3d_Q)x$(lap3d_R)_0_$(lap3d_nproc) && \
	$(TARGETDIR)/collectMtx \
		$(TESTOUTDIR)/lap3d_$(lap3d_gen)_$(lap3d_nx)x$(lap3d_ny)x$(lap3d_nz)_$(lap3d_P)x$(lap3d_Q)x$(lap3d_R)_multi.mtx \
		$(TESTOUTDIR)/lap3d_$(lap3d_gen)_$(lap3d_nx)x$(lap3d_ny)x$(lap3d_nz)_$(lap3d_P)x$(lap3d_Q)x$(lap3d_R)_*_$(lap3d_nproc)

	@echo "Comparing results"
	diff -y --suppress-common-lines \
		$(TESTOUTDIR)/lap3d_$(lap3d_gen)_$(lap3d_nx)x$(lap3d_ny)x$(lap3d_nz)_$(lap3d_P)x$(lap3d_Q)x$(lap3d_R)_benchmark.mtx \
		$(TESTOUTDIR)/lap3d_$(lap3d_gen)_$(lap3d_nx)x$(lap3d_ny)x$(lap3d_nz)_$(lap3d_P)x$(lap3d_Q)x$(lap3d_R)_multi.mtx
