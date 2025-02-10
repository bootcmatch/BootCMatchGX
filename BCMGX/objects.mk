# ==============================================================================

C_SRCS           := $(shell find ./$(SOURCEDIR) -name "*.c" ! -name "*.unused.c" ! -path "./$(SOURCEDIR)/test/*" ! -path "./$(SOURCEDIR)/example/*")
C_OBJS           := $(patsubst ./$(SOURCEDIR)/%.c,$(BUILDDIR)/%.o,$(C_SRCS))
C_DEPS           := $(C_OBJS:.o=.d)

CU_SRCS          := $(shell find ./$(SOURCEDIR) -name "*.cu" ! -name "*.unused.cu" ! -path "./$(SOURCEDIR)/test/*" ! -path "./$(SOURCEDIR)/example/*")
CU_OBJS          := $(patsubst ./$(SOURCEDIR)/%.cu,$(BUILDDIR)/%.o,$(CU_SRCS))
CU_DEPS          := $(CU_OBJS:.o=.d)

ALL_OBJS         := $(C_OBJS) $(CU_OBJS)
ALL_DEPS         := $(C_DEPS) $(CU_DEPS)

OBJECTS = \
	$(BUILDDIR)/EXTERNAL/nsparse.o \
	$(ALL_OBJS)

# ==============================================================================

info:
	@echo C_SRCS          : $(C_SRCS)
	@echo C_OBJS          : $(C_OBJS)
	@echo C_DEPS          : $(C_DEPS)
	@echo ""
	@echo CU_SRCS         : $(CU_SRCS)
	@echo CU_OBJS         : $(CU_OBJS)
	@echo CU_DEPS         : $(CU_DEPS)
	@echo ""
	@echo TEST_SRCS       : $(TEST_SRCS)
	@echo TEST_OBJS       : $(TEST_OBJS)
	@echo TEST_DEPS       : $(TEST_DEPS)
	@echo TEST_BINS       : $(TEST_BINS)
	@echo ""
	@echo EXAMPLE_SRCS    : $(EXAMPLE_SRCS)
	@echo EXAMPLE_OBJS    : $(EXAMPLE_OBJS)
	@echo EXAMPLE_DEPS    : $(EXAMPLE_DEPS)
	@echo EXAMPLE_BINS    : $(EXAMPLE_BINS)
	@echo ""

# ==============================================================================
# src/EXTERNAL
# ==============================================================================

$(BUILDDIR)/EXTERNAL/nsparse.o: $(NSPARSE_PATH)/src/kernel/kernel_spgemm_hash_d.cu
	$(MKDIR) $(@D)
	$(NVCC) -c -DDOUBLE -o $@ $(DEFINE) $(LIBS) $(INCLUDE) $(NSPARSE_GPU_ARCH) $(NVCC_FLAG) $^ $(NVCC_OPT)

# ==============================================================================
# Everything else
# ==============================================================================

-include $(ALL_DEPS)

.SECONDEXPANSION:

$(C_OBJS): %.o: $$(patsubst $(BUILDDIR)/$$(PERCENT),$(SOURCEDIR)/$$(PERCENT),%.c)
	$(MKDIR) $(@D)
	$(CC) -MMD -MP -c -o $@ $(DEFINE) $(INCLUDE) $< $(CC_OPT)

$(CU_OBJS): %.o: $$(patsubst $(BUILDDIR)/$$(PERCENT),$(SOURCEDIR)/$$(PERCENT),%.cu)
	$(MKDIR) $(@D)
	$(NVCC) -MMD -MP -c -DDOUBLE -o $@ $(DEFINE) $(LIBS) $(INCLUDE) $(NSPARSE_GPU_ARCH) $(NVCC_FLAG) $< $(NVCC_OPT)
