include config.mk

.PHONY: clean

all: lib example

TEST_SRCS        := $(shell find ./$(SOURCEDIR)/test -name "*.cu")
TEST_OBJS        := $(patsubst ./$(SOURCEDIR)/test/%.cu,$(BUILDDIR)/test/%.o,$(TEST_SRCS))
TEST_DEPS        := $(TEST_OBJS:.o=.d)
TEST_BINS        := $(patsubst ./$(SOURCEDIR)/test/%.cu,$(TARGETDIR)/test/%,$(TEST_SRCS))

EXAMPLE_SRCS     := $(shell find ./$(SOURCEDIR)/example -name "*.cu")
EXAMPLE_OBJS     := $(patsubst ./$(SOURCEDIR)/example/%.cu,$(BUILDDIR)/example/%.o,$(EXAMPLE_SRCS))
EXAMPLE_DEPS     := $(EXAMPLE_OBJS:.o=.d)
EXAMPLE_BINS     := $(patsubst ./$(SOURCEDIR)/example/%.cu,$(TARGETDIR)/example/%,$(EXAMPLE_SRCS))

# ==============================================================================
# `example` and `lib`
# ==============================================================================

example: \
	$(EXAMPLE_BINS)

lib: \
	$(LIBDIR)/lib$(LIBNAME).a

# ==============================================================================
# Objects
# ==============================================================================

include objects.mk

# ==============================================================================
# $(LIBDIR)/lib$(LIBNAME).a
# ==============================================================================

$(LIBDIR)/lib$(LIBNAME).a: $(OBJECTS)
	$(MKDIR) $(@D)
	$(NVCC) --lib -o $@ $^

# ==============================================================================
# clean
# ==============================================================================

clean:
	rm -Rf $(BUILDDIR) $(TARGETDIR) $(TESTOUTDIR) $(LIBDIR)

# ==============================================================================
# format
# ==============================================================================

format:
	$(FORMATTER) $(CU_SRCS) $(H_SRCS)

# ==============================================================================
# lint
# ==============================================================================

lintcpp:
ifdef CPP_SRCS
	$(LINTER) -checks="$(LINTER_CHECKS)" $(CPP_SRCS) $(CPP_TEST_SRCS) -- $(LINTER_FLAGS)
endif

lintcu:
ifdef CU_SRCS
	$(LINTER) -checks="$(LINTER_CHECKS)" $(CU_SRCS) $(CU_TEST_SRCS) -- $(LINTER_FLAGS)
endif

# linth:
# ifdef H_SRCS
# 	$(LINTER) -checks="$(LINTER_CHECKS)" $(H_SRCS) -- $(LINTER_FLAGS)
# endif

lint: lintcpp lintcu #linth

listchecks:
	clang-tidy -checks=* --list-checks

# ==============================================================================
# generate
# ==============================================================================

generate: $(TARGETDIR)/example/generateSampleMatrix $(TARGETDIR)/example/generateLaplacian3D $(TARGETDIR)/example/collectMtx
	$(TARGETDIR)/example/generateSampleMatrix \
		--verbose \
		--size 10 \
		--out $(SOURCEDIR)/test/data/mtx/dense_matrix_10x10.mtx

	$(TARGETDIR)/example/generateSampleMatrix \
		--verbose \
		--sparse \
		--size 10 \
		--out $(SOURCEDIR)/test/data/mtx/sparse_matrix_10x10.mtx

	for GEN in 7p 27p; do\
		for DIM in 10; do\
			for P in 1 2; do\
				for Q in 1 2; do\
					for R in 1 2; do\
						if [ ! -f $(SOURCEDIR)/test/data/mtx/lap3d_$${GEN}_$${DIM}x$${DIM}x$${DIM}_$${P}x$${Q}x$${R}.mtx ]; then\
							rm -f /tmp/lap3d_$${GEN}_$${DIM}x$${DIM}x$${DIM}_$${P}x$${Q}x$${R}_*_$$(($$P * $$Q * $$R));\
							$(MPIRUN) --oversubscribe -np $$(($$P * $$Q * $$R)) $(CUDA_MEMCHECK) \
								$(TARGETDIR)/example/generateLaplacian3D \
									-g $${GEN} \
									-x $${DIM} -y $${DIM} -z $${DIM} \
									-P $${P} -Q $${Q} -R $${R} \
									--out /tmp/lap3d_$${GEN}_$${DIM}x$${DIM}x$${DIM}_$${P}x$${Q}x$${R};\
							$(TARGETDIR)/example/collectMtx \
								$(SOURCEDIR)/test/data/mtx/lap3d_$${GEN}_$${DIM}x$${DIM}x$${DIM}_$${P}x$${Q}x$${R}.mtx \
								/tmp/lap3d_$${GEN}_$${DIM}x$${DIM}x$${DIM}_$${P}x$${Q}x$${R}_*_$$(($$P * $$Q * $$R));\
						fi;\
					done;\
				done;\
			done;\
		done;\
	done

# ==============================================================================
# testCompareSolutions
# ==============================================================================

testCompareSolutions: $(TARGETDIR)/example/compareSolutions
	$(TARGETDIR)/example/compareSolutions \
		$(SOURCEDIR)/test/data/cmp/vec1.txt \
		$(SOURCEDIR)/test/data/cmp/vec2.txt

# ==============================================================================
# $(TARGETDIR)/example/
# ==============================================================================

-include $(EXAMPLE_DEPS)

$(TARGETDIR)/example/%: src/example/%.cu $(LIBDIR)/lib$(LIBNAME).a
	$(MKDIR) $(@D)
	$(NVCC) -MMD -MP $< -o $@ $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $(LIBS) -L$(LIBDIR) -l$(LIBNAME) $(NVCC_OPT)

# ==============================================================================
# $(TARGETDIR)/test
# ==============================================================================

-include $(TEST_DEPS)

$(TARGETDIR)/test/%: $(SOURCEDIR)/test/%.cu $(OBJECTS)
	$(MKDIR) $(@D)
	$(NVCC) -MMD -MP $^ -o $@ $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $(LIBS) $(NVCC_OPT)

# ==============================================================================
# Unit tests
# ==============================================================================

test: \
	testCore \
	testCSR \
	testGenerators \
	testCommunication

include testCore.mk
include testCSR.mk
include testGenerators.mk
include testCommunication.mk

testAggregateVector: $(TARGETDIR)/test/testAggregateVector
	$(MPIRUN) -np 32 $(CUDA_MEMCHECK) $(TARGETDIR)/test/testAggregateVector

testReadSettings: $(TARGETDIR)/test/testReadSettings
	$(TARGETDIR)/test/testReadSettings --settings $(DATA_DIR)/settings/CGHS_AFSAI
	$(TARGETDIR)/test/testReadSettings --settings $(DATA_DIR)/settings/CGHS_AFSAI.properties

# ==============================================================================
# Regression tests
# ==============================================================================

STOP_ON_ERROR       := 0
OVERSUBSCRIBE       := 1
DUMP_SOLUTION       := 0
ENABLE_LOG          := 0
DETAILED_METRICS    := 0
DETAILED_PROF       := 0
SUMMARY_PROF        := 0
REPETITIONS         := 5
EXTENDED_PROF       := 1

ifeq ($(STOP_ON_ERROR),1)
EXIT_TEST = exit 1
else
EXIT_TEST = :
endif

SOLVER = CGHS FCG CGS1 CGS2 CGS3 \
	#PipelinedCGS1 PipelinedCGS2 PipelinedCGS3 \
	#CGS_CUBLAS2

PRECONDITIONER = noPreconditioner BCMG \
				 l1Jacobi AFSAI

#TEST_CONFIG = bcsstk09 bcsstk13 bcsstk24 \
	cfd2 msc23052 lap3d_7p_a50 \
	lap3d_7p_10x10x10_1x1x1 lap3d_7p_10x10x10_2x1x1 \
	lap3d_7p_10x10x10_2x2x1 lap3d_7p_10x10x10_2x2x2 \
	lap3d_7p_50x50x50_1x1x1 lap3d_7p_50x50x50_2x1x1 \
	lap3d_7p_50x50x50_2x2x1 lap3d_7p_50x50x50_2x2x2 \
	lap3d_27p_10x10x10_1x1x1 lap3d_27p_10x10x10_2x1x1 \
	lap3d_27p_10x10x10_2x2x1 lap3d_27p_10x10x10_2x2x2 \
	lap3d_27p_50x50x50_1x1x1 lap3d_27p_50x50x50_2x1x1 \
	lap3d_27p_50x50x50_2x2x1 lap3d_27p_50x50x50_2x2x2

#TEST_CONFIG = lap3d_7p_a50 \
	lap3d_7p_50x50x50_1x1x1 lap3d_7p_50x50x50_2x1x1 \
	lap3d_7p_50x50x50_2x2x1 lap3d_7p_50x50x50_2x2x2 \
	lap3d_27p_50x50x50_1x1x1 lap3d_27p_50x50x50_2x1x1 \
	lap3d_27p_50x50x50_2x2x1 lap3d_27p_50x50x50_2x2x2

#TEST_CONFIG = lap3d_7p_a10 \
	lap3d_7p_10x10x10_1x1x1 lap3d_7p_10x10x10_2x1x1 \
	lap3d_7p_10x10x10_2x2x1 lap3d_7p_10x10x10_2x2x2 \
	lap3d_27p_10x10x10_1x1x1 lap3d_27p_10x10x10_2x1x1 \
	lap3d_27p_10x10x10_2x2x1 lap3d_27p_10x10x10_2x2x2

TEST_CONFIG = \
	lap3d_27p_50x50x50_1x1x1 \
	lap3d_27p_50x50x50_2x1x1 lap3d_27p_50x50x50_2x2x1 lap3d_27p_50x50x50_2x2x2 \
	#lap3d_27p_50x50x50_4x2x2 lap3d_27p_50x50x50_4x4x2 lap3d_27p_50x50x50_4x4x4

#TEST_CONFIG = \
	lap3d_27p_50x50x50_4x2x2 lap3d_27p_50x50x50_4x4x2 lap3d_27p_50x50x50_4x4x4

#TEST_CONFIG = \
	lap3d_7p_100x100x100_1x1x1 lap3d_7p_100x100x100_2x1x1 \
	lap3d_7p_100x100x100_2x2x1 lap3d_7p_100x100x100_2x2x2 \
	lap3d_27p_100x100x100_1x1x1 lap3d_27p_100x100x100_2x1x1 \
	lap3d_27p_100x100x100_2x2x1 lap3d_27p_100x100x100_2x2x2

#TEST_CONFIG    = \
	lap3d_27p_160x160x160_1x1x1 lap3d_27p_160x160x160_2x1x1 \
	lap3d_27p_160x160x160_2x2x1 lap3d_27p_160x160x160_2x2x2

regressionTests: $(TARGETDIR)/example/driverSolve $(TARGETDIR)/example/compareSolutions generate
	$(MKDIR) $(TESTOUTDIR)
	for r in `seq 1 $(REPETITIONS)`; do\
		for s in $(SOLVER); do\
			for p in $(PRECONDITIONER); do\
				for c in $(TEST_CONFIG); do\
					make -f launchAndCompare.mk\
						nproc=4\
						solver=$$s\
						preconditioner=$$p\
						config=$$c \
						repetition=$$r \
						OVERSUBSCRIBE=$(OVERSUBSCRIBE) \
						DUMP_SOLUTION=$(DUMP_SOLUTION) \
						ENABLE_LOG=$(ENABLE_LOG) \
						DETAILED_METRICS=$(DETAILED_METRICS) \
						DETAILED_PROF=$(DETAILED_PROF) \
						SUMMARY_PROF=$(SUMMARY_PROF) \
					| tee $(TESTOUTDIR)/$${c}_$${s}_$${p}_r$${r}.makeout;\
					if [ `echo "$${PIPESTATUS[@]}" | tr -s ' ' + | bc` -ne 0 ]; then\
						echo "Regression test with SOLVER=$$s PRECONDITIONER=$$p TEST_CONFIG=$$c failed. Try launching:";\
						echo "make regressionTests SOLVER=$$s PRECONDITIONER=$$p TEST_CONFIG=$$c STOP_ON_ERROR=1 ENABLE_LOG=1" USE_CUDA_MEMCHECK=1;\
						$(EXIT_TEST);\
					fi;\
				done;\
			done;\
		done;\
	done

helpRegressionTests:
	@echo make regressionTests - Available parameters:
	@echo "STOP_ON_ERROR    : 0 | 1, default: $(STOP_ON_ERROR)"
	@echo "OVERSUBSCRIBE    : 0 | 1, default: $(OVERSUBSCRIBE)"
	@echo "USE_CUDA_MEMCHECK: 0 | 1, default: $(USE_CUDA_MEMCHECK)"
	@echo "USE_CUDA_PROFILER: 0 | 1, default: $(USE_CUDA_PROFILER)"
	@echo "DUMP_SOLUTION    : 0 | 1, default: $(DUMP_SOLUTION)"
	@echo "ENABLE_LOG       : 0 | 1, default: $(ENABLE_LOG)"
	@echo "DETAILED_METRICS : 0 | 1, default: $(DETAILED_METRICS)"
	@echo "DETAILED_PROF    : 0 | 1, default: $(DETAILED_PROF)"
	@echo "SUMMARY_PROF     : 0 | 1, default: $(SUMMARY_PROF)"
	@echo "REPETITIONS      : 0 | 1, default: $(REPETITIONS)"
	@echo "SOLVER           : $(SOLVER)"
	@echo "PRECONDITIONER   : $(PRECONDITIONER)"
	@echo "TEST_CONFIG      : $(TEST_CONFIG)"
