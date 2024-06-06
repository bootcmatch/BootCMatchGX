# parameters:
# - nproc: 1, 2, ...
# - solver: CGHS | FCG
# - preconditioner: noPreconditioner | l1Jacobi | BCMG
# - config: bcsstk09 | bcsstk13 | bcsstk24 | cfd2 | msc23052 | ...

include config.mk

ENABLE_LOG          := 1
SOLVER_OPTS         := CGHS FCG
PRECONDITIONER_OPTS := noPreconditioner l1Jacobi BCMG
CONFIG_OPTS         := bcsstk09 bcsstk13 bcsstk24 cfd2 msc23052 \
						lap3d_7p_a50 \
						lap3d_7p_3x3x3_1x1x1 \
						lap3d_7p_3x3x3_2x1x1 \
						lap3d_7p_3x3x3_2x2x1 \
						lap3d_7p_3x3x3_2x2x2 \
						lap3d_7p_10x10x10_1x1x1 \
						lap3d_7p_10x10x10_2x1x1 \
						lap3d_7p_10x10x10_2x2x1 \
						lap3d_7p_10x10x10_2x2x2 \
						lap3d_7p_50x50x50_1x1x1 \
						lap3d_7p_50x50x50_2x1x1 \
						lap3d_7p_50x50x50_2x2x1 \
						lap3d_7p_50x50x50_2x2x2 \
						lap3d_27p_3x3x3_1x1x1 \
						lap3d_27p_3x3x3_2x1x1 \
						lap3d_27p_3x3x3_2x2x1 \
						lap3d_27p_3x3x3_2x2x2 \
						lap3d_27p_10x10x10_1x1x1 \
						lap3d_27p_10x10x10_2x1x1 \
						lap3d_27p_10x10x10_2x2x1 \
						lap3d_27p_10x10x10_2x2x2 \
						lap3d_27p_50x50x50_1x1x1 \
						lap3d_27p_50x50x50_2x1x1 \
						lap3d_27p_50x50x50_2x2x1 \
						lap3d_27p_50x50x50_2x2x2

NPROC__lap3d_7p_a50             := 1

NPROC__lap3d_7p_3x3x3_1x1x1  := 1
NPROC__lap3d_7p_3x3x3_2x1x1  := 2
NPROC__lap3d_7p_3x3x3_2x2x1  := 4
NPROC__lap3d_7p_3x3x3_2x2x2  := 8

NPROC__lap3d_7p_10x10x10_1x1x1  := 1
NPROC__lap3d_7p_10x10x10_2x1x1  := 2
NPROC__lap3d_7p_10x10x10_2x2x1  := 4
NPROC__lap3d_7p_10x10x10_2x2x2  := 8

NPROC__lap3d_7p_50x50x50_1x1x1  := 1
NPROC__lap3d_7p_50x50x50_2x1x1  := 2
NPROC__lap3d_7p_50x50x50_2x2x1  := 4
NPROC__lap3d_7p_50x50x50_2x2x2  := 8

NPROC__lap3d_27p_3x3x3_1x1x1 := 1
NPROC__lap3d_27p_3x3x3_2x1x1 := 2
NPROC__lap3d_27p_3x3x3_2x2x1 := 4
NPROC__lap3d_27p_3x3x3_2x2x2 := 8

NPROC__lap3d_27p_10x10x10_1x1x1 := 1
NPROC__lap3d_27p_10x10x10_2x1x1 := 2
NPROC__lap3d_27p_10x10x10_2x2x1 := 4
NPROC__lap3d_27p_10x10x10_2x2x2 := 8

NPROC__lap3d_27p_50x50x50_1x1x1 := 1
NPROC__lap3d_27p_50x50x50_2x1x1 := 2
NPROC__lap3d_27p_50x50x50_2x2x1 := 4
NPROC__lap3d_27p_50x50x50_2x2x2 := 8

LAUNCH_PARAMS__bcsstk09                 := -m $(SOURCEDIR)/test/data/mtx/bcsstk09.mtx
LAUNCH_PARAMS__bcsstk13                 := -m $(SOURCEDIR)/test/data/mtx/bcsstk13.mtx
LAUNCH_PARAMS__bcsstk24                 := -m $(SOURCEDIR)/test/data/mtx/bcsstk24.mtx
LAUNCH_PARAMS__cfd2                     := -m $(SOURCEDIR)/test/data/mtx/cfd2.mtx
LAUNCH_PARAMS__msc23052                 := -m $(SOURCEDIR)/test/data/mtx/msc23052.mtx

LAUNCH_PARAMS__lap3d_7p_a50             := -a 50

LAUNCH_PARAMS__lap3d_7p_3x3x3_1x1x1     := -g 7p -l $(SOURCEDIR)/test/data/cfg/lap3d_3x3x3_1x1x1.cfg
LAUNCH_PARAMS__lap3d_7p_3x3x3_2x1x1     := -g 7p -l $(SOURCEDIR)/test/data/cfg/lap3d_3x3x3_2x1x1.cfg
LAUNCH_PARAMS__lap3d_7p_3x3x3_2x2x1     := -g 7p -l $(SOURCEDIR)/test/data/cfg/lap3d_3x3x3_2x2x1.cfg
LAUNCH_PARAMS__lap3d_7p_3x3x3_2x2x2     := -g 7p -l $(SOURCEDIR)/test/data/cfg/lap3d_3x3x3_2x2x2.cfg

LAUNCH_PARAMS__lap3d_7p_10x10x10_1x1x1  := -g 7p -l $(SOURCEDIR)/test/data/cfg/lap3d_10x10x10_1x1x1.cfg
LAUNCH_PARAMS__lap3d_7p_10x10x10_2x1x1  := -g 7p -l $(SOURCEDIR)/test/data/cfg/lap3d_10x10x10_2x1x1.cfg
LAUNCH_PARAMS__lap3d_7p_10x10x10_2x2x1  := -g 7p -l $(SOURCEDIR)/test/data/cfg/lap3d_10x10x10_2x2x1.cfg
LAUNCH_PARAMS__lap3d_7p_10x10x10_2x2x2  := -g 7p -l $(SOURCEDIR)/test/data/cfg/lap3d_10x10x10_2x2x2.cfg

LAUNCH_PARAMS__lap3d_7p_50x50x50_1x1x1  := -g 7p -l $(SOURCEDIR)/test/data/cfg/lap3d_50x50x50_1x1x1.cfg
LAUNCH_PARAMS__lap3d_7p_50x50x50_2x1x1  := -g 7p -l $(SOURCEDIR)/test/data/cfg/lap3d_50x50x50_2x1x1.cfg
LAUNCH_PARAMS__lap3d_7p_50x50x50_2x2x1  := -g 7p -l $(SOURCEDIR)/test/data/cfg/lap3d_50x50x50_2x2x1.cfg
LAUNCH_PARAMS__lap3d_7p_50x50x50_2x2x2  := -g 7p -l $(SOURCEDIR)/test/data/cfg/lap3d_50x50x50_2x2x2.cfg

LAUNCH_PARAMS__lap3d_27p_3x3x3_1x1x1    := -g 27p -l $(SOURCEDIR)/test/data/cfg/lap3d_3x3x3_1x1x1.cfg
LAUNCH_PARAMS__lap3d_27p_3x3x3_2x1x1    := -g 27p -l $(SOURCEDIR)/test/data/cfg/lap3d_3x3x3_2x1x1.cfg
LAUNCH_PARAMS__lap3d_27p_3x3x3_2x2x1    := -g 27p -l $(SOURCEDIR)/test/data/cfg/lap3d_3x3x3_2x2x1.cfg
LAUNCH_PARAMS__lap3d_27p_3x3x3_2x2x2    := -g 27p -l $(SOURCEDIR)/test/data/cfg/lap3d_3x3x3_2x2x2.cfg

LAUNCH_PARAMS__lap3d_27p_10x10x10_1x1x1 := -g 27p -l $(SOURCEDIR)/test/data/cfg/lap3d_10x10x10_1x1x1.cfg
LAUNCH_PARAMS__lap3d_27p_10x10x10_2x1x1 := -g 27p -l $(SOURCEDIR)/test/data/cfg/lap3d_10x10x10_2x1x1.cfg
LAUNCH_PARAMS__lap3d_27p_10x10x10_2x2x1 := -g 27p -l $(SOURCEDIR)/test/data/cfg/lap3d_10x10x10_2x2x1.cfg
LAUNCH_PARAMS__lap3d_27p_10x10x10_2x2x2 := -g 27p -l $(SOURCEDIR)/test/data/cfg/lap3d_10x10x10_2x2x2.cfg

LAUNCH_PARAMS__lap3d_27p_50x50x50_1x1x1 := -g 27p -l $(SOURCEDIR)/test/data/cfg/lap3d_50x50x50_1x1x1.cfg
LAUNCH_PARAMS__lap3d_27p_50x50x50_2x1x1 := -g 27p -l $(SOURCEDIR)/test/data/cfg/lap3d_50x50x50_2x1x1.cfg
LAUNCH_PARAMS__lap3d_27p_50x50x50_2x2x1 := -g 27p -l $(SOURCEDIR)/test/data/cfg/lap3d_50x50x50_2x2x1.cfg
LAUNCH_PARAMS__lap3d_27p_50x50x50_2x2x2 := -g 27p -l $(SOURCEDIR)/test/data/cfg/lap3d_50x50x50_2x2x2.cfg

REF_SOL__bcsstk09                 := $(SOURCEDIR)/test/data/cmp/bcsstk09_sol.txt
REF_SOL__bcsstk13                 := $(SOURCEDIR)/test/data/cmp/bcsstk13_sol.txt
REF_SOL__bcsstk24                 := $(SOURCEDIR)/test/data/cmp/bcsstk24_sol.txt
REF_SOL__cfd2                     := $(SOURCEDIR)/test/data/cmp/cfd2_sol.txt
REF_SOL__msc23052                 := $(SOURCEDIR)/test/data/cmp/msc23052_sol.txt

REF_SOL__lap3d_7p_a50             := $(SOURCEDIR)/test/data/cmp/lap3d_7p_50x50x50_1x1x1_sol.txt

REF_SOL__lap3d_7p_10x10x10_1x1x1  := $(SOURCEDIR)/test/data/cmp/lap3d_7p_10x10x10_1x1x1_sol.txt
REF_SOL__lap3d_7p_10x10x10_2x1x1  := $(SOURCEDIR)/test/data/cmp/lap3d_7p_10x10x10_2x1x1_sol.txt
REF_SOL__lap3d_7p_10x10x10_2x2x1  := $(SOURCEDIR)/test/data/cmp/lap3d_7p_10x10x10_2x2x1_sol.txt
REF_SOL__lap3d_7p_10x10x10_2x2x2  := $(SOURCEDIR)/test/data/cmp/lap3d_7p_10x10x10_2x2x2_sol.txt

REF_SOL__lap3d_7p_50x50x50_1x1x1  := $(SOURCEDIR)/test/data/cmp/lap3d_7p_50x50x50_1x1x1_sol.txt
REF_SOL__lap3d_7p_50x50x50_2x1x1  := $(SOURCEDIR)/test/data/cmp/lap3d_7p_50x50x50_2x1x1_sol.txt
REF_SOL__lap3d_7p_50x50x50_2x2x1  := $(SOURCEDIR)/test/data/cmp/lap3d_7p_50x50x50_2x2x1_sol.txt
REF_SOL__lap3d_7p_50x50x50_2x2x2  := $(SOURCEDIR)/test/data/cmp/lap3d_7p_50x50x50_2x2x2_sol.txt

REF_SOL__lap3d_27p_10x10x10_1x1x1 := $(SOURCEDIR)/test/data/cmp/lap3d_27p_10x10x10_1x1x1_sol.txt
REF_SOL__lap3d_27p_10x10x10_2x1x1 := $(SOURCEDIR)/test/data/cmp/lap3d_27p_10x10x10_2x1x1_sol.txt
REF_SOL__lap3d_27p_10x10x10_2x2x1 := $(SOURCEDIR)/test/data/cmp/lap3d_27p_10x10x10_2x2x1_sol.txt
REF_SOL__lap3d_27p_10x10x10_2x2x2 := $(SOURCEDIR)/test/data/cmp/lap3d_27p_10x10x10_2x2x2_sol.txt

REF_SOL__lap3d_27p_50x50x50_1x1x1 := $(SOURCEDIR)/test/data/cmp/lap3d_27p_50x50x50_1x1x1_sol.txt
REF_SOL__lap3d_27p_50x50x50_2x1x1 := $(SOURCEDIR)/test/data/cmp/lap3d_27p_50x50x50_2x1x1_sol.txt
REF_SOL__lap3d_27p_50x50x50_2x2x1 := $(SOURCEDIR)/test/data/cmp/lap3d_27p_50x50x50_2x2x1_sol.txt
REF_SOL__lap3d_27p_50x50x50_2x2x2 := $(SOURCEDIR)/test/data/cmp/lap3d_27p_50x50x50_2x2x2_sol.txt

ifneq ($(NPROC__$(config)),)
TEST_NPROC = $(NPROC__$(config))
else
TEST_NPROC ?= $(nproc)
endif

ifeq ($(ENABLE_LOG), 1)
LOG_PARAM := -e $(TESTOUTDIR)/log_$(solver)_$(preconditioner)_$(config)
endif

launchAndCompare:
	@echo TEST_NPROC: $(TEST_NPROC)
	@echo nproc     : $(nproc)
# Validate parameters
ifndef TEST_NPROC
    $(error Missing parameter - nproc)
endif
ifndef solver
    $(error Missing parameter - solver)
endif
ifeq ($(filter $(solver),$(SOLVER_OPTS)),)
    $(error Parameter 'solver' must be in $(SOLVER_OPTS))
endif
ifndef preconditioner
    $(error Missing parameter - preconditioner)
endif
ifeq ($(filter $(preconditioner),$(PRECONDITIONER_OPTS)),)
    $(error Parameter 'preconditioner' must be in $(PRECONDITIONER_OPTS))
endif
ifndef config
    $(error Missing parameter - config)
endif
ifeq ($(filter $(config),$(CONFIG_OPTS)),)
    $(error Parameter 'config' must be in $(CONFIG_OPTS))
endif

# Launch
	$(MPIRUN) -np $(TEST_NPROC) $(CUDA_MEMCHECK) $(TARGETDIR)/main \
		$(LAUNCH_PARAMS__$(config)) \
		-s $(SOURCEDIR)/test/data/settings/$(solver)_$(preconditioner) \
		-o $(TESTOUTDIR)/$(solver)_$(preconditioner)_$(config)_sol.txt \
		-i $(TESTOUTDIR)/$(solver)_$(preconditioner)_$(config)_info.txt \
		$(LOG_PARAM)

# Compare solutions
ifneq ($(REF_SOL__$(config)),)
	$(TARGETDIR)/compareSolutions \
		$(REF_SOL__$(config)) \
		$(TESTOUTDIR)/$(solver)_$(preconditioner)_$(config)_sol.txt \
		"1.e-5"
endif
