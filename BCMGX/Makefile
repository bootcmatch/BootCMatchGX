include config.mk

.PHONY: clean prepare

all: prepare \
	$(TARGETDIR)/main \
	$(TARGETDIR)/checkAndFixOrder \
	$(TARGETDIR)/generate \
	$(TARGETDIR)/collectMtx \
	$(TARGETDIR)/compareSolutions

include objects.mk

# ==============================================================================
# $(TARGETDIR)/main
# ==============================================================================

$(TARGETDIR)/main: src/main.cu $(OBJECTS)
	$(NVCC) $^ -o $@ $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $(LIBS) $(NVCC_OPT)

# ==============================================================================
# $(TARGETDIR)/checkAndFixOrder
# ==============================================================================

$(TARGETDIR)/checkAndFixOrder: src/checkAndFixOrder.cu $(OBJECTS)
	$(NVCC) $^ -o $@ $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $(LIBS) $(NVCC_OPT)

# ==============================================================================
# $(TARGETDIR)/generate
# ==============================================================================

$(TARGETDIR)/generate: src/generate.cu $(OBJECTS)
	$(NVCC) $^ -o $@ $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $(LIBS) $(NVCC_OPT)

# ==============================================================================
# $(TARGETDIR)/collectMtx
# ==============================================================================

$(TARGETDIR)/collectMtx: src/collectMtx.cu $(OBJECTS)
	$(NVCC) $^ -o $@ $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $(LIBS) $(NVCC_OPT)

# ==============================================================================
# $(TARGETDIR)/compareSolutions
# ==============================================================================

$(TARGETDIR)/compareSolutions: src/compareSolutions.cu $(OBJECTS)
	$(NVCC) $^ -o $@ $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $(LIBS) $(NVCC_OPT)

# ==============================================================================
# clean
# ==============================================================================

clean:
	rm -Rf $(BUILDDIR) $(TARGETDIR) $(TESTOUTDIR)

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

lint: lintcpp lintcu

listchecks:
	clang-tidy -checks=* --list-checks

# ==============================================================================
# prepare
# ==============================================================================

prepare:
	mkdir -p ${TARGETDIR}
	mkdir -p ${TARGETDIR}/test
	mkdir -p ${TESTOUTDIR}

# ==============================================================================
# generate
# ==============================================================================

generate: prepare $(TARGETDIR)/generate
	$(TARGETDIR)/generate \
		--verbose \
		--size 10 \
		--out $(SOURCEDIR)/test/data/mtx/dense_matrix_10x10.mtx

	$(TARGETDIR)/generate \
		--verbose \
		--sparse \
		--size 10 \
		--out $(SOURCEDIR)/test/data/mtx/sparse_matrix_10x10.mtx

# ==============================================================================
# testCompareSolutions
# ==============================================================================

testCompareSolutions: prepare $(TARGETDIR)/compareSolutions
	$(TARGETDIR)/compareSolutions \
		$(SOURCEDIR)/test/data/cmp/vec1.txt \
		$(SOURCEDIR)/test/data/cmp/vec2.txt

# ==============================================================================
# $(TARGETDIR)/test
# ==============================================================================

$(TARGETDIR)/test/%: $(SOURCEDIR)/test/%.cu $(OBJECTS)
	$(NVCC) $^ -o $@ $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $(LIBS) $(NVCC_OPT)

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

# ==============================================================================
# Regression tests
# ==============================================================================

STOP_ON_ERROR = 0
ENABLE_LOG = 0

ifeq ($(STOP_ON_ERROR),1)
EXIT_TEST = exit 1
else
EXIT_TEST = :
endif

SOLVER         = CGHS FCG
PRECONDITIONER = noPreconditioner l1Jacobi BCMG
TEST_CONFIG    = lap3d_7p_a50 \
	lap3d_7p_50x50x50_1x1x1 lap3d_7p_50x50x50_2x1x1 \
	lap3d_7p_50x50x50_2x2x1 lap3d_7p_50x50x50_2x2x2 \
	lap3d_27p_50x50x50_1x1x1 lap3d_27p_50x50x50_2x1x1 \
	lap3d_27p_50x50x50_2x2x1 lap3d_27p_50x50x50_2x2x2

regressionTests: $(TARGETDIR)/main
	for s in $(SOLVER); do\
		for p in $(PRECONDITIONER); do\
			for c in $(TEST_CONFIG); do\
				make -f launchAndCompare.mk\
					nproc=4\
					solver=$$s\
					preconditioner=$$p\
					config=$$c \
					ENABLE_LOG=$(ENABLE_LOG) \
				| tee $(TESTOUTDIR)/test$${s}_$${p}_$${c}.makeout;\
				if [ `echo "$${PIPESTATUS[@]}" | tr -s ' ' + | bc` -ne 0 ]; then\
					echo "Regression test with SOLVER=$$s PRECONDITIONER=$$p TEST_CONFIG=$$c failed. Try launching:";\
					echo "make regressionTests SOLVER=$$s PRECONDITIONER=$$p TEST_CONFIG=$$c STOP_ON_ERROR=1 ENABLE_LOG=1" USE_CUDA_MEMCHECK=1;\
					$(EXIT_TEST);\
				fi;\
			done;\
		done;\
	done;

helpRegressionTests:
	@echo make regressionTests - Available parameters:
	@echo "USE_CUDA_MEMCHECK: 0 | 1, default: 0"
	@echo "STOP_ON_ERROR    : 0 | 1, default: 0"
	@echo "ENABLE_LOG       : 0 | 1, default: 0"
	@echo "SOLVER           : $(SOLVER)"
	@echo "PRECONDITIONER   : $(PRECONDITIONER)"
	@echo "TEST_CONFIG      : $(TEST_CONFIG)"

