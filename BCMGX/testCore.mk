testCore: \
	testDeviceFilter \
	testDeviceForEach \
	testDeviceMap \
	testDevicePrefixSum \
	testDeviceSort \
	testDeviceUnique

testDeviceFilter: prepare $(TARGETDIR)/test/testDeviceFilter
	$(CUDA_MEMCHECK) $(TARGETDIR)/test/testDeviceFilter

testDeviceForEach: prepare $(TARGETDIR)/test/testDeviceForEach
	$(CUDA_MEMCHECK) $(TARGETDIR)/test/testDeviceForEach

testDeviceMap: prepare $(TARGETDIR)/test/testDeviceMap
	$(CUDA_MEMCHECK) $(TARGETDIR)/test/testDeviceMap

testDevicePrefixSum: prepare $(TARGETDIR)/test/testDevicePrefixSum
	$(CUDA_MEMCHECK) $(TARGETDIR)/test/testDevicePrefixSum

testDeviceSort: prepare $(TARGETDIR)/test/testDeviceSort
	$(CUDA_MEMCHECK) $(TARGETDIR)/test/testDeviceSort

testDeviceUnique: prepare $(TARGETDIR)/test/testDeviceUnique
	$(CUDA_MEMCHECK) $(TARGETDIR)/test/testDeviceUnique
