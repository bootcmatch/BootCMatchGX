# parameters:
# - nproc: 1, 2, ...
# - solver: CGHS | FCG | CGS1 | CGS2 | CGS3 | CGS_CUBLAS2 | PipelinedCGS1 | PipelinedCGS2 | PipelinedCGS3
# - preconditioner: noPreconditioner | l1Jacobi | BCMG | AFSAI
# - config: bcsstk09 | bcsstk13 | bcsstk24 | cfd2 | msc23052 | ...

include config.mk

OVERSUBSCRIBE       := 0
DUMP_SOLUTION       := 0
ENABLE_LOG          := 0
DETAILED_METRICS    := 0
DETAILED_PROF       := 0
SUMMARY_PROF        := 0

SOLVER_OPTS         := CGHS FCG \
						CGS1 CGS2 CGS3 \
						CGS_CUBLAS2 \
						PipelinedCGS1 PipelinedCGS2 PipelinedCGS3
PRECONDITIONER_OPTS := noPreconditioner l1Jacobi BCMG AFSAI
CONFIG_OPTS         := bcsstk09 bcsstk13 bcsstk24 cfd2 msc23052 \
						lap3d_7p_a10 \
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
						lap3d_7p_100x100x100_1x1x1 \
						lap3d_7p_100x100x100_2x1x1 \
						lap3d_7p_100x100x100_2x2x1 \
						lap3d_7p_100x100x100_2x2x2 \
						lap3d_7p_160x160x160_1x1x1 \
						lap3d_7p_160x160x160_2x1x1 \
						lap3d_7p_160x160x160_2x2x1 \
						lap3d_7p_160x160x160_2x2x2 \
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
						lap3d_27p_50x50x50_2x2x2 \
						lap3d_27p_50x50x50_4x2x2 \
						lap3d_27p_50x50x50_4x4x2 \
						lap3d_27p_50x50x50_4x4x4 \
						lap3d_27p_100x100x100_1x1x1 \
						lap3d_27p_100x100x100_2x1x1 \
						lap3d_27p_100x100x100_2x2x1 \
						lap3d_27p_100x100x100_2x2x2 \
						lap3d_27p_160x160x160_1x1x1 \
						lap3d_27p_160x160x160_2x1x1 \
						lap3d_27p_160x160x160_2x2x1 \
						lap3d_27p_160x160x160_2x2x2 \
						lap3d_27p_160x160x160_4x2x2 \
						lap3d_27p_160x160x160_4x4x2 \
						lap3d_27p_160x160x160_4x4x4

NPROC__lap3d_7p_a10          := 1
NPROC__lap3d_7p_a50          := 1

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
NPROC__lap3d_7p_50x50x50_4x2x2  := 16
NPROC__lap3d_7p_50x50x50_4x4x2  := 32
NPROC__lap3d_7p_50x50x50_4x4x4  := 64

NPROC__lap3d_7p_100x100x100_1x1x1  := 1
NPROC__lap3d_7p_100x100x100_2x1x1  := 2
NPROC__lap3d_7p_100x100x100_2x2x1  := 4
NPROC__lap3d_7p_100x100x100_2x2x2  := 8

NPROC__lap3d_7p_160x160x160_1x1x1  := 1
NPROC__lap3d_7p_160x160x160_2x1x1  := 2
NPROC__lap3d_7p_160x160x160_2x2x1  := 4
NPROC__lap3d_7p_160x160x160_2x2x2  := 8
NPROC__lap3d_7p_160x160x160_4x2x2  := 16
NPROC__lap3d_7p_160x160x160_4x4x2  := 32
NPROC__lap3d_7p_160x160x160_4x4x4  := 64

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
NPROC__lap3d_27p_50x50x50_4x2x2 := 16
NPROC__lap3d_27p_50x50x50_4x4x2 := 32
NPROC__lap3d_27p_50x50x50_4x4x4 := 64

NPROC__lap3d_27p_100x100x100_1x1x1 := 1
NPROC__lap3d_27p_100x100x100_2x1x1 := 2
NPROC__lap3d_27p_100x100x100_2x2x1 := 4
NPROC__lap3d_27p_100x100x100_2x2x2 := 8

NPROC__lap3d_27p_160x160x160_1x1x1 := 1
NPROC__lap3d_27p_160x160x160_2x1x1 := 2
NPROC__lap3d_27p_160x160x160_2x2x1 := 4
NPROC__lap3d_27p_160x160x160_2x2x2 := 8
NPROC__lap3d_27p_160x160x160_4x2x2 := 16
NPROC__lap3d_27p_160x160x160_4x4x2 := 32
NPROC__lap3d_27p_160x160x160_4x4x4 := 64

LAUNCH_PARAMS__bcsstk09                 := -m $(SOURCEDIR)/test/data/mtx/bcsstk09.mtx
LAUNCH_PARAMS__bcsstk13                 := -m $(SOURCEDIR)/test/data/mtx/bcsstk13.mtx
LAUNCH_PARAMS__bcsstk24                 := -m $(SOURCEDIR)/test/data/mtx/bcsstk24.mtx
LAUNCH_PARAMS__cfd2                     := -m $(SOURCEDIR)/test/data/mtx/cfd2.mtx
LAUNCH_PARAMS__msc23052                 := -m $(SOURCEDIR)/test/data/mtx/msc23052.mtx

LAUNCH_PARAMS__lap3d_7p_a10             := -a 10
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
LAUNCH_PARAMS__lap3d_7p_50x50x50_4x2x2  := -g 7p -l $(SOURCEDIR)/test/data/cfg/lap3d_50x50x50_4x2x2.cfg
LAUNCH_PARAMS__lap3d_7p_50x50x50_4x4x2  := -g 7p -l $(SOURCEDIR)/test/data/cfg/lap3d_50x50x50_4x4x2.cfg
LAUNCH_PARAMS__lap3d_7p_50x50x50_4x4x4  := -g 7p -l $(SOURCEDIR)/test/data/cfg/lap3d_50x50x50_4x4x4.cfg

LAUNCH_PARAMS__lap3d_7p_100x100x100_1x1x1  := -g 7p -l $(SOURCEDIR)/test/data/cfg/lap3d_100x100x100_1x1x1.cfg
LAUNCH_PARAMS__lap3d_7p_100x100x100_2x1x1  := -g 7p -l $(SOURCEDIR)/test/data/cfg/lap3d_100x100x100_2x1x1.cfg
LAUNCH_PARAMS__lap3d_7p_100x100x100_2x2x1  := -g 7p -l $(SOURCEDIR)/test/data/cfg/lap3d_100x100x100_2x2x1.cfg
LAUNCH_PARAMS__lap3d_7p_100x100x100_2x2x2  := -g 7p -l $(SOURCEDIR)/test/data/cfg/lap3d_100x100x100_2x2x2.cfg

LAUNCH_PARAMS__lap3d_7p_160x160x160_1x1x1  := -g 7p -l $(SOURCEDIR)/test/data/cfg/lap3d_160x160x160_1x1x1.cfg
LAUNCH_PARAMS__lap3d_7p_160x160x160_2x1x1  := -g 7p -l $(SOURCEDIR)/test/data/cfg/lap3d_160x160x160_2x1x1.cfg
LAUNCH_PARAMS__lap3d_7p_160x160x160_2x2x1  := -g 7p -l $(SOURCEDIR)/test/data/cfg/lap3d_160x160x160_2x2x1.cfg
LAUNCH_PARAMS__lap3d_7p_160x160x160_2x2x2  := -g 7p -l $(SOURCEDIR)/test/data/cfg/lap3d_160x160x160_2x2x2.cfg
LAUNCH_PARAMS__lap3d_7p_160x160x160_4x2x2  := -g 7p -l $(SOURCEDIR)/test/data/cfg/lap3d_160x160x160_4x2x2.cfg
LAUNCH_PARAMS__lap3d_7p_160x160x160_4x4x2  := -g 7p -l $(SOURCEDIR)/test/data/cfg/lap3d_160x160x160_4x4x2.cfg
LAUNCH_PARAMS__lap3d_7p_160x160x160_4x4x4  := -g 7p -l $(SOURCEDIR)/test/data/cfg/lap3d_160x160x160_4x4x4.cfg

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
LAUNCH_PARAMS__lap3d_27p_50x50x50_4x2x2 := -g 27p -l $(SOURCEDIR)/test/data/cfg/lap3d_50x50x50_4x2x2.cfg
LAUNCH_PARAMS__lap3d_27p_50x50x50_4x4x2 := -g 27p -l $(SOURCEDIR)/test/data/cfg/lap3d_50x50x50_4x4x2.cfg
LAUNCH_PARAMS__lap3d_27p_50x50x50_4x4x4 := -g 27p -l $(SOURCEDIR)/test/data/cfg/lap3d_50x50x50_4x4x4.cfg

LAUNCH_PARAMS__lap3d_27p_100x100x100_1x1x1 := -g 27p -l $(SOURCEDIR)/test/data/cfg/lap3d_100x100x100_1x1x1.cfg
LAUNCH_PARAMS__lap3d_27p_100x100x100_2x1x1 := -g 27p -l $(SOURCEDIR)/test/data/cfg/lap3d_100x100x100_2x1x1.cfg
LAUNCH_PARAMS__lap3d_27p_100x100x100_2x2x1 := -g 27p -l $(SOURCEDIR)/test/data/cfg/lap3d_100x100x100_2x2x1.cfg
LAUNCH_PARAMS__lap3d_27p_100x100x100_2x2x2 := -g 27p -l $(SOURCEDIR)/test/data/cfg/lap3d_100x100x100_2x2x2.cfg

LAUNCH_PARAMS__lap3d_27p_160x160x160_1x1x1 := -g 27p -l $(SOURCEDIR)/test/data/cfg/lap3d_160x160x160_1x1x1.cfg
LAUNCH_PARAMS__lap3d_27p_160x160x160_2x1x1 := -g 27p -l $(SOURCEDIR)/test/data/cfg/lap3d_160x160x160_2x1x1.cfg
LAUNCH_PARAMS__lap3d_27p_160x160x160_2x2x1 := -g 27p -l $(SOURCEDIR)/test/data/cfg/lap3d_160x160x160_2x2x1.cfg
LAUNCH_PARAMS__lap3d_27p_160x160x160_2x2x2 := -g 27p -l $(SOURCEDIR)/test/data/cfg/lap3d_160x160x160_2x2x2.cfg
LAUNCH_PARAMS__lap3d_27p_160x160x160_4x2x2 := -g 27p -l $(SOURCEDIR)/test/data/cfg/lap3d_160x160x160_4x2x2.cfg
LAUNCH_PARAMS__lap3d_27p_160x160x160_4x4x2 := -g 27p -l $(SOURCEDIR)/test/data/cfg/lap3d_160x160x160_4x4x2.cfg
LAUNCH_PARAMS__lap3d_27p_160x160x160_4x4x4 := -g 27p -l $(SOURCEDIR)/test/data/cfg/lap3d_160x160x160_4x4x4.cfg

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

REF_SOL__lap3d_7p_100x100x100_1x1x1  := $(SOURCEDIR)/test/data/cmp/lap3d_7p_100x100x100_1x1x1_sol.txt
REF_SOL__lap3d_7p_100x100x100_2x1x1  := $(SOURCEDIR)/test/data/cmp/lap3d_7p_100x100x100_2x1x1_sol.txt
REF_SOL__lap3d_7p_100x100x100_2x2x1  := $(SOURCEDIR)/test/data/cmp/lap3d_7p_100x100x100_2x2x1_sol.txt
REF_SOL__lap3d_7p_100x100x100_2x2x2  := $(SOURCEDIR)/test/data/cmp/lap3d_7p_100x100x100_2x2x2_sol.txt

REF_SOL__lap3d_27p_10x10x10_1x1x1 := $(SOURCEDIR)/test/data/cmp/lap3d_27p_10x10x10_1x1x1_sol.txt
REF_SOL__lap3d_27p_10x10x10_2x1x1 := $(SOURCEDIR)/test/data/cmp/lap3d_27p_10x10x10_2x1x1_sol.txt
REF_SOL__lap3d_27p_10x10x10_2x2x1 := $(SOURCEDIR)/test/data/cmp/lap3d_27p_10x10x10_2x2x1_sol.txt
REF_SOL__lap3d_27p_10x10x10_2x2x2 := $(SOURCEDIR)/test/data/cmp/lap3d_27p_10x10x10_2x2x2_sol.txt

REF_SOL__lap3d_27p_50x50x50_1x1x1 := $(SOURCEDIR)/test/data/cmp/lap3d_27p_50x50x50_1x1x1_sol.txt
REF_SOL__lap3d_27p_50x50x50_2x1x1 := $(SOURCEDIR)/test/data/cmp/lap3d_27p_50x50x50_2x1x1_sol.txt
REF_SOL__lap3d_27p_50x50x50_2x2x1 := $(SOURCEDIR)/test/data/cmp/lap3d_27p_50x50x50_2x2x1_sol.txt
REF_SOL__lap3d_27p_50x50x50_2x2x2 := $(SOURCEDIR)/test/data/cmp/lap3d_27p_50x50x50_2x2x2_sol.txt

REF_SOL__lap3d_27p_100x100x100_1x1x1 := $(SOURCEDIR)/test/data/cmp/lap3d_27p_100x100x100_1x1x1_sol.txt
REF_SOL__lap3d_27p_100x100x100_2x1x1 := $(SOURCEDIR)/test/data/cmp/lap3d_27p_100x100x100_2x1x1_sol.txt
REF_SOL__lap3d_27p_100x100x100_2x2x1 := $(SOURCEDIR)/test/data/cmp/lap3d_27p_100x100x100_2x2x1_sol.txt
REF_SOL__lap3d_27p_100x100x100_2x2x2 := $(SOURCEDIR)/test/data/cmp/lap3d_27p_100x100x100_2x2x2_sol.txt

ifneq ($(NPROC__$(config)),)
TEST_NPROC = $(NPROC__$(config))
else
TEST_NPROC ?= $(nproc)
endif

ifeq ($(OVERSUBSCRIBE), 1)
OVERSUBSCRIBE_PARAM := --oversubscribe
endif

ifeq ($(USE_CUDA_PROFILER), 1)
CUDA_PROFILER_ARGS := --trace=cuda,nvtx,osrt,mpi -o $(TESTOUTDIR)/$(config)_$(solver)_$(preconditioner)_r$(repetition)_nsys_%q{OMPI_COMM_WORLD_RANK}
endif

ifeq ($(DUMP_SOLUTION), 1)
SOL_PARAM := -o $(TESTOUTDIR)/$(config)_$(solver)_$(preconditioner)_r$(repetition)_sol.txt
endif

ifeq ($(ENABLE_LOG), 1)
LOG_PARAM := -e $(TESTOUTDIR)/$(config)_$(solver)_$(preconditioner)_r$(repetition)_log_
endif

ifeq ($(DETAILED_PROF), 1)
DETAILED_PROF_PARAM := -P $(TESTOUTDIR)/$(config)_$(solver)_$(preconditioner)_r$(repetition)_prof_detail_
endif

ifeq ($(SUMMARY_PROF), 1)
SUMMARY_PROF_PARAM := -p $(TESTOUTDIR)/$(config)_$(solver)_$(preconditioner)_r$(repetition)_prof_summary_
endif

ifeq ($(DETAILED_METRICS), 1)
DETAILED_METRICS_PARAM := -M $(TESTOUTDIR)/$(config)_$(solver)_$(preconditioner)_r$(repetition)_time_
endif

launchAndCompare:
	@if [ ! -e $(MPIRUN) ]; then\
		echo "ERROR: Could not find $(MPIRUN)";\
		exit 1;\
	fi

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
ifndef repetition
    $(error Missing parameter - repetition)
endif
ifndef config
    $(error Missing parameter - config)
endif
ifeq ($(filter $(config),$(CONFIG_OPTS)),)
    $(error Parameter 'config' must be in $(CONFIG_OPTS))
endif

# Launch
	$(MPIRUN) $(OVERSUBSCRIBE_PARAM) -np $(TEST_NPROC) $(CUDA_MEMCHECK) $(CUDA_PROFILER) $(CUDA_PROFILER_ARGS) \
		$(TARGETDIR)/example/driverSolve \
		$(LAUNCH_PARAMS__$(config)) \
		-s $(SOURCEDIR)/test/data/settings/$(solver)_$(preconditioner).properties \
		-i $(TESTOUTDIR)/$(config)_$(solver)_$(preconditioner)_r$(repetition).info \
		$(SOL_PARAM) $(LOG_PARAM) $(DETAILED_METRICS_PARAM) $(DETAILED_PROF_PARAM) $(SUMMARY_PROF_PARAM)

# Compare solutions
ifneq ($(REF_SOL__$(config)),)
	if [ -f $(TESTOUTDIR)/$(config)_$(solver)_$(preconditioner)_r$(repetition)_sol.txt ] && [ -f $(REF_SOL__$(config)) ]; then \
		echo "Comparing $(TESTOUTDIR)/$(config)_$(solver)_$(preconditioner)_r$(repetition)_sol.txt with $(REF_SOL__$(config))";\
		$(TARGETDIR)/example/compareSolutions \
			$(REF_SOL__$(config)) \
			$(TESTOUTDIR)/$(config)_$(solver)_$(preconditioner)_r$(repetition)_sol.txt \
			"1.e-5"; \
	fi
endif
