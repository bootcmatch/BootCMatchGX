// 2023 INFN APE Lab - Sezione di Roma
// cristian.rossi@roma1.infn.it

#include "GPowerU.h"


//CPU thread managing the parallel power data taking during the kernel execution
void *threadWork(void * arg) {
    unsigned int power[MAX_DEVICES];
    int i=0;
    bool not_enough=0;
    struct timeval tv_start, tv_aux;
    struct timeval *time = (struct timeval *) arg;
		
    tv_start=*time;
    printf("************STARTING THREAD**************\n");
    //printf("terminate_thread: %d\n",terminate_thread);
    //terminate_thread = 0;
    //printf("terminate_thread: %d\n",terminate_thread);

    while (!terminate_thread) {
	//GET POWER SAMPLES
	for(int d=0; d < device_count; d++){		       
            nvResult = nvmlDeviceGetPowerUsage(nvDevice[d], &power[d]);
	    if (NVML_SUCCESS != nvResult) {
		//printf("Failed to get power usage: %s [device %d]\n", nvmlErrorString(nvResult), d);
		if (nvResult == NVML_ERROR_UNINITIALIZED){
		    //printf("NVML_ERROR_UNINITIALIZED: the library has not been successfully initialized\n");
		    pthread_exit(NULL);
		}
		/*if (nvResult == NVML_ERROR_INVALID_ARGUMENT){
		    printf("NVML_ERROR_INVALID_ARGUMENT: device is invalid or power is NULL\n");
		}
		if (nvResult == NVML_ERROR_NOT_SUPPORTED){
		    printf("NVML_ERROR_NOT_SUPPORTED: the device does not support power readings\n");
		}
		if (nvResult == NVML_ERROR_GPU_IS_LOST){
		    printf("NVML_ERROR_GPU_IS_LOST: the target GPU has fallen off the bus or is otherwise inaccessible\n");
		}
		if (nvResult == NVML_ERROR_UNKNOWN){
		    printf("NVML_ERROR_UNKNOWN on any unexpected error\n");
		}*/
		if ( i>0 ) {
		  power[d] = thread_powers[d][i-1];
		}else{
		  power[d] = 0;
		}
	    }			
	    if(i < SAMPLE_MAX_SIZE_DEFAULT && (power[d] > POWER_THRESHOLD*1000.0  || thread_powers[d][0] > POWER_THRESHOLD*1000.0)) {
		gettimeofday(&tv_aux,NULL);
            	thread_powers[d][i] = power[d];			
		thread_times[d][i] = (tv_aux.tv_sec-tv_start.tv_sec)*1000000;
            	thread_times[d][i] += (tv_aux.tv_usec-tv_start.tv_usec);
		if(i==0) { 
		    printf("******STARTING GPU WORK******\n");
		}
		if(i==0) { 
	            FILE *fp_starttime;
		    fp_starttime = fopen("data/start_time.time", "a");
		    fprintf(fp_starttime,"%d;%ld;%ld\n", d, tv_aux.tv_sec, tv_aux.tv_usec);
		    fclose(fp_starttime);
		}

	     }
	     else{
	        if(i == SAMPLE_MAX_SIZE_DEFAULT) {
		    printf("ERROR: POWER VECTOR SIZE EXCEEDED!\n");
		    pthread_exit(NULL);
		}
		if(!not_enough){
		    printf("NOT ENOUGH POWER!\n");
		    not_enough=1;
		}
	     }	
	}
	i++;
	n_values = i;
	sleep(TIME_STEP);
    }	
    printf("************STOP THREAD**************\n");
    pthread_exit(NULL);
}


//Generate the output samples files
float DataOutput() {
   int values_threshold=0;
   float acc0 = 0.0;
   float p_average;
   double interval;
   double interval_GPU;
   int begin_gpu=-1, end_gpu=n_values-1;
   power_peak=0;
   FILE  *fp2;
	
   for(int d=0; d < device_count; d++){
	std::string s = "data/nvml_power_profile";
	s = s + std::to_string(d);
	s = s + ".csv";
	fp2 = fopen(s.c_str(), "w+");
	fprintf(fp2,"Timestamp [us];Power measure [W]");
	//fprintf(fp2,"#sep=;\n#Timestamp [us];Power measure [W]");

	for(int i=0; i<n_values; i++) {
            fprintf(fp2, "\n%.6f;%.4f", (thread_times[d][i]-thread_times[d][0])/1000000, thread_powers[d][i]/1000.0);	
            if (thread_powers[d][i] > power_peak) 
        		power_peak = thread_powers[d][i];
            if ( thread_powers[d][i]/1000.0 > 35 && begin_gpu == -1 ) begin_gpu = i;
            if ( thread_powers[d][i]/1000.0 < 35 && begin_gpu != -1 ) end_gpu = i;

            if (thread_powers[d][i]/1000.0 >= threshold) {
        	acc0 = acc0 + thread_powers[d][i];
         	values_threshold++;
            }
	}
	fclose(fp2);
   }
   if (values_threshold>0) {
      p_average = acc0 / (values_threshold*1.0);
   }
   else {
      printf("ERROR: DIVISION BY 0\n");
      exit(-1);
   }
    
   interval = thread_times[0][n_values-1] - thread_times[0][0];
   interval_GPU = thread_times[0][end_gpu] - thread_times[0][begin_gpu];
   printf("\tAt current frequency (%d,%d) MHz:  Average Power: %.2f W;  Max Power: %.2f W;  Sampling Duration: %.2f ms;  GPU active duration: %.2f ms \n", mem_clock, core_clock, p_average/1000.0, power_peak/1000.0, (interval)/1000, interval_GPU/1000);
   
   return 0;
}

//Initializations ==> enable the NVML library, starts CPU thread for the power monitoring. It is synchronized with the start time of the program
int GPowerU_init() {
    gettimeofday(&start_time,NULL);
    int a;
    int check = mkdir("data", 0777);
   
    FILE *fp_starttime;
    fp_starttime = fopen("data/start_time.time", "w");
    fprintf(fp_starttime,"device;tv_sec;tv_usec\n");
    //fprintf(fp_starttime,"#device;tv_sec;tv_usec\n");
    //fprintf(fp_starttime,"---;%ld;%ld\n", start_time.tv_sec, start_time.tv_usec);
    fclose(fp_starttime);
   
#if MULTIGPU_DISABLED
   for (int i = 0; i < MAX_CHECKPOINTS; i++) {
        kernel_checkpoints[i]= 0;
   }
#endif
     
        //printf("terminate_thread: %d\n",terminate_thread);	
	terminate_thread = 0;
        //printf("terminate_thread: %d\n",terminate_thread);	
	// NVML INITIALIZATIONS
	nvResult = nvmlInit();
	if (NVML_SUCCESS != nvResult)
    {
        printf("Failed to initialize NVML: %s\n", nvmlErrorString(nvResult));

        printf("Press ENTER to continue...\n");
        getchar();
        return -1;
    }

	nvResult = nvmlDeviceGetCount(&device_count);
    if (NVML_SUCCESS != nvResult)
    {
        printf("Failed to query device count: %s\n", nvmlErrorString(nvResult));
        return -1;
    }

	printf("Found %d device%s\n\n", device_count, device_count != 1 ? "s" : "");
    if (deviceID >= device_count) {
        printf("Device_id is out of range.\n");
        return -1;
    }

	for(int d=0; d<device_count; d++){
		nvResult = nvmlDeviceGetHandleByIndex(d, &nvDevice[d]);
		if (NVML_SUCCESS != nvResult)
		{
			printf("Failed to get handle for device %d: %s\n",d, nvmlErrorString(nvResult));
		 	return -1;
		}
	}

	//LAUNCH POWER SAMPLER
	a = pthread_create(&thread_sampler, NULL, threadWork, &start_time);
	if(a) {
		fprintf(stderr,"Error - pthread_create() return code: %d\n",a);
		return -1;
	}

	return 0;
}


//ROOT graph making function
#if ROOT_ENABLED
void grapher(){
	auto c1 = new TCanvas("c1","PowerMeas",200,10,700,500);
   c1->SetGrid();
	TGraphErrors *gr1  = new TGraphErrors("data/nvml_power_profile0.csv", "%lg;%lg");
   gr1->Draw("AP");
   gr1->SetTitle("GPU Power Measurement (GPowerU) ;" "Time (s);" "Power (W)");
   
#if MULTIGPU_DISABLED    
   TGraphErrors *gr2  = new TGraphErrors("data/power_checkpoints.csv", "%lg;%lg");
   gr2->SetMarkerColor(4);
   gr2->SetMarkerStyle(20);
	gr2->SetMarkerSize(1.5);
   gr2->Draw("P");
#endif
   
   c1->Print("data/gpu_graph.pdf");
}
#endif


//Checkpoint power measure __device__ function ==> last to be set =1 for the latest func call
__device__ void take_GPU_time(bool last = 0){
#if MULTIGPU_DISABLED
	static int i=0;
	if(kernel_checkpoints[i]==0){
		kernel_checkpoints[i]=1;
		i++;
		max_points++;
	}
	if(last) finish=1;
#endif
}



//Checkpoint power measure CPU function ==> it calls its own cudaDeviceSynchronize() 
void GPowerU_checkpoints(){
#if MULTIGPU_DISABLED
	unsigned int power;
 	FILE *fp2;
	//struct timespec time_aux;
   struct timeval tv_aux;
   
	fp2 = fopen("data/power_checkpoints.csv", "w");
   fprintf(fp2,"#sep=;\n Timestamp [s]; Power[W]");
   	
   	
   int n_saved_points = 0;
  	while (!finish && kernel_checkpoints[n_saved_points+1] == 0) {
  			for(int i = n_saved_points; i < max_points+1; i++){
  				if(kernel_checkpoints[i]==1) {
					gettimeofday(&tv_aux,NULL);
   				nvResult = nvmlDeviceGetPowerUsage(nvDevice[0], &power);
   				device_times[i] = (tv_aux.tv_sec - start_time.tv_sec )*1000000;
         		device_times[i] += (tv_aux.tv_usec - start_time.tv_usec)-thread_times[0][0];
         		device_powers[i] = power;
        			kernel_checkpoints[i]=0;
        			n_saved_points++;	
        	}
  		}
	}
#endif
	cudaDeviceSynchronize();

#if MULTIGPU_DISABLED
   for(int i = 0; i < max_points; i++) fprintf(fp2, "\n %.4f; %.4f", device_times[i]/1000000, device_powers[i]/1000);
   fclose(fp2);
   finish=0;
#endif
   
 }


//Ends power monitoring, returns data output files
int GPowerU_end(int zz=0) {
   sleep(zz);
	terminate_thread = 1;
	pthread_join(thread_sampler, NULL);
	DataOutput();
   //printf("File out_power_samples.csv created with all measured samples.\n");
   //printf("File out_power.txt average power consumption.\n");
   
	#if ROOT_ENABLED
	grapher();
	#endif
	//printf("Finished\n\n");

	return 0;
}
