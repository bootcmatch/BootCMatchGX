testCore: \
	testDeviceFilter \
	testDeviceForEach \
	testDeviceMap \
	testDevicePrefixSum \
	testDeviceSort \
	testDeviceUnique

testDeviceFilter: $(TARGETDIR)/test/testDeviceFilter
	$(CUDA_MEMCHECK) $(TARGETDIR)/test/testDeviceFilter

testDeviceForEach: $(TARGETDIR)/test/testDeviceForEach
	$(CUDA_MEMCHECK) $(TARGETDIR)/test/testDeviceForEach

testDeviceMap: $(TARGETDIR)/test/testDeviceMap
	$(CUDA_MEMCHECK) $(TARGETDIR)/test/testDeviceMap

testDevicePrefixSum: $(TARGETDIR)/test/testDevicePrefixSum
	$(CUDA_MEMCHECK) $(TARGETDIR)/test/testDevicePrefixSum

testDeviceSort: $(TARGETDIR)/test/testDeviceSort
	$(CUDA_MEMCHECK) $(TARGETDIR)/test/testDeviceSort

testDeviceUnique: $(TARGETDIR)/test/testDeviceUnique
	$(CUDA_MEMCHECK) $(TARGETDIR)/test/testDeviceUnique

testProfiling: $(TARGETDIR)/test/testProfiling
	$(TARGETDIR)/test/testProfiling
