testGenerators: testGenerateLaplacian3D

lap3d_nx=10
lap3d_ny=10
lap3d_nz=10
lap3d_P=2
lap3d_Q=2
lap3d_R=2
lap3d_nproc=8
lap3d_gen=27p

# 1) Riordina la matrice di input generata in Matlab
# 2) Lancia il generatore laplaciano 3d
#    NB: il numero di processi deve essere pari a PxQxR
testGenerateLaplacian3D: $(TARGETDIR)/example/collectMtx $(TARGETDIR)/test/testGenerateLaplacian3D
	$(MKDIR) $(TESTOUTDIR)
	
	@echo "Fixing order (input)"
	rm -f $(TESTOUTDIR)/lap3d_$(lap3d_gen)_$(lap3d_nx)x$(lap3d_ny)x$(lap3d_nz)_$(lap3d_P)x$(lap3d_Q)x$(lap3d_R)_benchmark.mtx
	$(TARGETDIR)/example/collectMtx \
		$(TESTOUTDIR)/lap3d_$(lap3d_gen)_$(lap3d_nx)x$(lap3d_ny)x$(lap3d_nz)_$(lap3d_P)x$(lap3d_Q)x$(lap3d_R)_benchmark.mtx \
		$(SOURCEDIR)/test/data/mtx/lap3d_$(lap3d_gen)_$(lap3d_nx)x$(lap3d_ny)x$(lap3d_nz)_$(lap3d_P)x$(lap3d_Q)x$(lap3d_R).mtx

	@echo "Generating laplacian"
	rm -f $(TESTOUTDIR)/lap3d_$(lap3d_gen)_$(lap3d_nx)x$(lap3d_ny)x$(lap3d_nz)_$(lap3d_P)x$(lap3d_Q)x$(lap3d_R)_log_*
	rm -f $(TESTOUTDIR)/lap3d_$(lap3d_gen)_$(lap3d_nx)x$(lap3d_ny)x$(lap3d_nz)_$(lap3d_P)x$(lap3d_Q)x$(lap3d_R)_*_$(lap3d_nproc)
	$(MPIRUN) --oversubscribe -np $(lap3d_nproc) $(CUDA_MEMCHECK) $(TARGETDIR)/test/testGenerateLaplacian3D \
		-o $(TESTOUTDIR)/lap3d_$(lap3d_gen)_$(lap3d_nx)x$(lap3d_ny)x$(lap3d_nz)_$(lap3d_P)x$(lap3d_Q)x$(lap3d_R) \
		-g $(lap3d_gen) -x $(lap3d_nx) -y $(lap3d_ny) -z $(lap3d_nz) -P $(lap3d_P) -Q $(lap3d_Q) -R $(lap3d_R) \

#		-l $(TESTOUTDIR)/lap3d_$(lap3d_gen)_$(lap3d_nx)x$(lap3d_ny)x$(lap3d_nz)_$(lap3d_P)x$(lap3d_Q)x$(lap3d_R)_log

	@echo "Collecting results"
	test -f $(TESTOUTDIR)/lap3d_$(lap3d_gen)_$(lap3d_nx)x$(lap3d_ny)x$(lap3d_nz)_$(lap3d_P)x$(lap3d_Q)x$(lap3d_R)_0_$(lap3d_nproc) && \
	$(TARGETDIR)/example/collectMtx \
		$(TESTOUTDIR)/lap3d_$(lap3d_gen)_$(lap3d_nx)x$(lap3d_ny)x$(lap3d_nz)_$(lap3d_P)x$(lap3d_Q)x$(lap3d_R)_multi.mtx \
		$(TESTOUTDIR)/lap3d_$(lap3d_gen)_$(lap3d_nx)x$(lap3d_ny)x$(lap3d_nz)_$(lap3d_P)x$(lap3d_Q)x$(lap3d_R)_*_$(lap3d_nproc)

	@echo "Comparing results"
	diff -y --suppress-common-lines \
		$(TESTOUTDIR)/lap3d_$(lap3d_gen)_$(lap3d_nx)x$(lap3d_ny)x$(lap3d_nz)_$(lap3d_P)x$(lap3d_Q)x$(lap3d_R)_benchmark.mtx \
		$(TESTOUTDIR)/lap3d_$(lap3d_gen)_$(lap3d_nx)x$(lap3d_ny)x$(lap3d_nz)_$(lap3d_P)x$(lap3d_Q)x$(lap3d_R)_multi.mtx

fem_nx=30
fem_ny=30
fem_P=2
fem_Q=1
fem_nproc=2

testGenerateFem: $(TARGETDIR)/example/collectMtx $(TARGETDIR)/example/generateFemSerial $(TARGETDIR)/example/generateFemMpi
	$(MKDIR) $(TESTOUTDIR)

	$(TARGETDIR)/example/generateFemSerial \
		-O $(TESTOUTDIR) \
		-e $(TESTOUTDIR)/log_$(fem_nx)x$(fem_ny)_$(fem_P)x$(fem_Q)_serial \
		--nx $(fem_nx) \
		--ny $(fem_ny) \
		-P $(fem_P) \
		-Q $(fem_Q)

	$(MPIRUN) --oversubscribe -np $(fem_nproc) $(CUDA_MEMCHECK) $(TARGETDIR)/example/generateFemMpi \
		-O $(TESTOUTDIR) \
		-e $(TESTOUTDIR)/log_$(fem_nx)x$(fem_ny)_$(fem_P)x$(fem_Q)_mpi \
		--nx $(fem_nx) \
		--ny $(fem_ny) \
		-P $(fem_P) \
		-Q $(fem_Q)

	diff $(TESTOUTDIR)/A_$(fem_nx)x$(fem_ny)_$(fem_P)x$(fem_Q)_serial.mtx $(TESTOUTDIR)/A_$(fem_nx)x$(fem_ny)_$(fem_P)x$(fem_Q)_mpi.mtx > $(TESTOUTDIR)/A_$(fem_nx)x$(fem_ny)_$(fem_P)x$(fem_Q).diff
	diff $(TESTOUTDIR)/rhs_$(fem_nx)x$(fem_ny)_$(fem_P)x$(fem_Q)_serial.mtx $(TESTOUTDIR)/rhs_$(fem_nx)x$(fem_ny)_$(fem_P)x$(fem_Q)_mpi.mtx > $(TESTOUTDIR)/rhs_$(fem_nx)x$(fem_ny)_$(fem_P)x$(fem_Q).diff

# valgrind --leak-check=full
