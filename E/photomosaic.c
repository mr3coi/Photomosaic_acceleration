#include "photomosaic.h"

#include <CL/cl.h>
//#include "CL/cl_ext_collective.h"
#include <limits.h>
#include "helper_func.h"
#include <string.h>
#include <unistd.h>

#define dev_num			16
#define TNUM 			64
#define TILE_SIZE		3072
#define TILE_H			32
#define TILE_W			32
#define TILE_C			3

#if DEBUG==1
#define SLEEP_CHECK(sec) sleep(sec); printf(">>>>> Sleep complete\n");
#endif

void photomosaic(unsigned char *img, int width, int height, unsigned char *dataset, int *idx) {
	// Other variables
	unsigned int i, j;

	// Constants
	unsigned int		DEV_NUM			= dev_num;
	const unsigned int	swidth			= width / TILE_W,
		  				sheight			= height / TILE_H;
	const size_t		DATA_SIZE		= 60000 * TILE_C * TILE_H * TILE_W;
	const size_t		DB_SIZE			= TILE_H * width * TILE_C;		// H * W * C
	const size_t		DB_OUT_COUNT	= swidth,
		  				DB_OUT_SIZE		= DB_OUT_COUNT * sizeof(int);

	if (sheight < 24) DEV_NUM /= 4;
	if (sheight < 8) DEV_NUM /= 4;
#if DEBUG==1
	printf(">>> Final DEV_NUM : %d\n", DEV_NUM);
#endif

	// Region allocation among devices
	unsigned int		num_rows[DEV_NUM], cum_rows[DEV_NUM];
	i = sheight / DEV_NUM;
	for (j=0;j<DEV_NUM;j++) num_rows[j] = i;
	i = sheight - i*DEV_NUM;
	for (j=0;j<i;j++) num_rows[j]++;
	cum_rows[0] = 0;
	for (j=1;j<DEV_NUM;j++) cum_rows[j] = cum_rows[j-1] + num_rows[j-1];

	// OpenCL variables
	cl_platform_id		platform;
	cl_device_id		device_id[DEV_NUM];
	cl_context			context;
	cl_command_queue	command_queue[DEV_NUM][3];
	cl_program			program;
	cl_kernel			kernel;
	cl_mem				img_buf[DEV_NUM][sheight],
						cifar_buf[DEV_NUM],
						output[DEV_NUM][sheight];
	cl_int				err;
#if DEBUG==1
	cl_int				status;
#endif
	cl_event			read_event[DEV_NUM][sheight+1],
						kernel_event[DEV_NUM][sheight],
						write_event[DEV_NUM][sheight];
	char *kernel_src;
	const char *kernel_fname = "kernel.c", *kernel_name = "pm_kernel";
	const size_t kernel_src_len = read_kernel(kernel_fname, &kernel_src);
	size_t				g_wsize[3]		= {1, swidth, TNUM},
						l_wsize[3]		= {1, 1, TNUM};

	// ========================== OpenCL ============================
	// Find platform
	err = clGetPlatformIDs(1,&platform,NULL);
#if DEBUG==1
	checkError(err, __LINE__);
#endif
	
	// Find device
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, DEV_NUM, device_id, NULL);
#if DEBUG==1
	checkError(err, __LINE__);
#endif

	// Create a context for the platform and devices
	context = clCreateContext(0, DEV_NUM, device_id, NULL, NULL, &err);
#if DEBUG==1
	checkError(err, __LINE__);
#endif

	// Create a command_queue for each device
	for (i=0;i<DEV_NUM;i++) {
		for (j=0;j<3;j++) {
#if PROF==1
			command_queue[i][j] = clCreateCommandQueue(context,device_id[i],CL_QUEUE_PROFILING_ENABLE,&err);
#else
			command_queue[i][j] = clCreateCommandQueue(context,device_id[i],0,&err);
#endif
#if DEBUG==1
			checkError(err, __LINE__);
#endif
		}
	}

#if DEBUG==1
	printf(">>>>>>> Command queues generation complete\n");
#endif

	// Create and build a program with the given kernel source code
	program = clCreateProgramWithSource(context,1,(const char **)&kernel_src, &kernel_src_len, &err);
#if DEBUG==1
	checkError(err, __LINE__);
#endif

#if DEBUG==1
	printf(">>>>>>> program creation complete\n");
#endif

	//const char *args = NULL;
	char args[20], buf[10];
	strcpy(args, "-D TNUM=");
	sprintf(buf,"%d",TNUM);
	strcat(args,buf);
	size_t log_size;
	char *log;

	err = clBuildProgram(program, DEV_NUM, device_id, args, NULL, NULL);
	if (err != CL_SUCCESS) {
		if (err == CL_BUILD_PROGRAM_FAILURE) {
			for (i=0;i<DEV_NUM;i++) {
				//Determine the size of the log
				clGetProgramBuildInfo(program, device_id[i], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

				//Allocate memory for the log
				log = (char *) malloc(log_size);

				//Get the log
				clGetProgramBuildInfo(program, device_id[i], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

				//Print the log
				printf(">>> Build log for device %d:\n", i);
				printf("%s\n", log);
			}
		}
		else checkError(err, __LINE__);
	}

#if DEBUG==1
	printf(">>>>>>> program build complete\n");
#endif

	// Create a kernel for each devicee
	kernel = clCreateKernel(program, kernel_name, &err);
#if DEBUG==1
	checkError(err, __LINE__);
#endif

#if DEBUG==1
	printf(">>>>>>> Kernel generation complete\n");
#endif

	// Write CIFAR-10 data to global memory first => other buffers are written after these buffers
	for (i=0;i<DEV_NUM;i++) {
		cifar_buf[i]	= clCreateBuffer(context, CL_MEM_READ_ONLY, DATA_SIZE, NULL, &err);
		err = clEnqueueWriteBuffer(command_queue[i][0], cifar_buf[i], CL_TRUE,
								   0, DATA_SIZE, (void *)dataset,
								   0, NULL, write_event[i]+sheight);
	}

#if DEBUG==1
	for (i=0;i<DEV_NUM;i++) {
		err = clGetEventInfo(write_event[i][sheight], CL_EVENT_COMMAND_EXECUTION_STATUS,
							 sizeof(cl_int), &status, NULL);
		checkError(err, __LINE__);

		printf("(Device %d) cifar_write status : %d (Q:%d, S:%d, R:%d, C:%d)\n",
				i, status, CL_QUEUED, CL_SUBMITTED, CL_RUNNING, CL_COMPLETE);
	}

	printf(">>>>>>> cifar writing complete\n");
#endif

	// Create and write into rest of buffers
	for (i=0;i<DEV_NUM;i++) {
		for (j=0;j<num_rows[i];j++) {
			//printf(">>>>> Creating & writing buffers of device %d, DB %d\n", i, j);		// TODO Delete
			img_buf[i][j]	= clCreateBuffer(context, CL_MEM_READ_ONLY, DB_SIZE, NULL, &err);
			output[i][j]	= clCreateBuffer(context, CL_MEM_WRITE_ONLY, DB_OUT_SIZE, NULL, &err);

			err = clEnqueueWriteBuffer(command_queue[i][0], img_buf[i][j], CL_TRUE,
					0, DB_SIZE, (void *)(img + (cum_rows[i] + j) * DB_SIZE),
					1, write_event[i]+sheight, write_event[i]+j);
			//printf(">>>>> Finished creating & writing buffers of device %d, DB %d\n", i, j);	// TODO Delete
		}
	}

#if DEBUG==1
	for (i=0;i<DEV_NUM;i++) {
		for (j=0;j<num_rows[i];j++) {
			err = clGetEventInfo(write_event[i][j], CL_EVENT_COMMAND_EXECUTION_STATUS,
					sizeof(cl_int), &status, NULL);
			checkError(err, __LINE__);

			printf("(%d, %d) img_buf status : %d (Q:%d, S:%d, R:%d, C:%d)\n",
					i, j, status, CL_QUEUED, CL_SUBMITTED, CL_RUNNING, CL_COMPLETE);
		}
	}
	printf(">>>>>>> rest writing complete\n");
#endif

	// Create buffer, write into them, and
	// set them as arguments to respective kernels, and run kernels
	for (i=0;i<DEV_NUM;i++) {
		for (j=0;j<num_rows[i];j++) {
			err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)(img_buf[i]+j));
#if DEBUG==1
			checkError(err, __LINE__);
#endif
			err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)(cifar_buf+i));
#if DEBUG==1
			checkError(err, __LINE__);
#endif
			err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)(output[i]+j));
#if DEBUG==1
			checkError(err, __LINE__);
#endif

			err = clEnqueueNDRangeKernel(command_queue[i][1], kernel, 3, NULL,
										 g_wsize, l_wsize,
										 1, write_event[i]+j, kernel_event[i]+j);
#if DEBUG==1
			checkError(err, __LINE__);
			printf("======= kernel for device id %d launched, DB stage : %d =====\n", i, j);
#endif
		}
	}

#if DEBUG==1
	printf(">>>>>>> enqueuing complete\n");
#endif

	// Retrieve results from buffer to C
	for (i=0;i<DEV_NUM;i++) {
		for (j=0;j<num_rows[i];j++) {
			err = clEnqueueReadBuffer(command_queue[i][2], output[i][j], CL_TRUE,
					0, DB_OUT_SIZE, (void *)(idx + (cum_rows[i] + j) * DB_OUT_COUNT),
					1, kernel_event[i]+j, read_event[i]+j);
#if DEBUG==1
			checkError(err, __LINE__);
#endif
		}
	}

#if DEBUG==1
	printf(">>>>>>> enqueue reading complete\n");
#endif

	for (i=0;i<DEV_NUM;i++) {
		err = clFlush(command_queue[i][2]);
#if DEBUG==1
		checkError(err, __LINE__);
#endif
	}

	for (i=0;i<DEV_NUM;i++) {
		err = clFinish(command_queue[i][2]);
#if DEBUG==1
		checkError(err, __LINE__);
#endif
	}

#if DEBUG==1
	printf(">>>>>>> Flush & finish complete\n");
#endif

#if DEBUG==1
	for (i=0;i<DEV_NUM;i++) {
		for (j=0;j<num_rows[i];j++) {
			err = clGetEventInfo(write_event[i][j], CL_EVENT_COMMAND_EXECUTION_STATUS,
					sizeof(cl_int), &status, NULL);
			checkError(err, __LINE__);
			if (status != CL_COMPLETE) {
				printf("Write event %d of device %d is not yet complete.\n", j, i);
				err = clWaitForEvents(1, write_event[i]+j);
				checkError(err, __LINE__);
			}

			err = clGetEventInfo(kernel_event[i][j], CL_EVENT_COMMAND_EXECUTION_STATUS,
					sizeof(cl_int), &status, NULL);
			checkError(err, __LINE__);
			if (status != CL_COMPLETE) {
				printf("Kernel event %d of device %dis not yet complete.\n", j, i);
				err = clWaitForEvents(1, kernel_event[i]+j);
				checkError(err, __LINE__);
			}

			err = clGetEventInfo(read_event[i][j], CL_EVENT_COMMAND_EXECUTION_STATUS,
					sizeof(cl_int), &status, NULL);
			checkError(err, __LINE__);
			if (status != CL_COMPLETE) {
				printf("Read event %d of device %d is not yet complete.\n", j, i);
				err = clWaitForEvents(1, read_event[i]+j);
				checkError(err, __LINE__);
			}
		}
	}
	printf(">>>>>>> finished\n");
#endif

	// Cleanup
	for (i=0;i<DEV_NUM;i++) {
		for (j=0;j<3;j++) {
			err = clFlush(command_queue[i][j]);
			checkError(err, __LINE__);
			err = clFinish(command_queue[i][j]);
			checkError(err, __LINE__);
		}
	}
#if DEBUG==1
	printf("==================== Completed cleaning command_queues ===================\n");
#endif

	for (i=0;i<DEV_NUM;i++) {
		err = clReleaseMemObject(cifar_buf[i]);
		checkError(err, __LINE__);
		for (j=0;j<num_rows[i];j++) {
			err = clReleaseMemObject(img_buf[i][j]);
			checkError(err, __LINE__);
			err = clReleaseMemObject(output[i][j]);
			checkError(err, __LINE__);
		}
	}
#if DEBUG==1
	printf("==================== Completed clReleaseMemObject ===================\n");
#endif

	err = clReleaseProgram(program);
	checkError(err, __LINE__);
#if DEBUG==1
	printf("==================== Completed clReleaseProgram ===================\n");
#endif

	err = clReleaseKernel(kernel);
	checkError(err, __LINE__);
#if DEBUG==1
	printf("==================== Completed clReleaseKernel ===================\n");
#endif

	for (i=0;i<DEV_NUM;i++) {
		for (j=0;j<3;j++) {
			err = clReleaseCommandQueue(command_queue[i][j]);
			checkError(err, __LINE__);
		}
	}
#if DEBUG==1
	printf("==================== Completed clReleaseCommandQueue ===================\n");
#endif

	err = clReleaseContext(context);
	checkError(err, __LINE__);
#if DEBUG==1
	printf("==================== Completed clReleaseContext ===================\n");
#endif

	for (i=0;i<DEV_NUM;i++) {
		err = clReleaseDevice(device_id[i]);
		//checkError(err, __LINE__); printf("Device %d released.\n", i);		// TODO Delete
	}
#if DEBUG==1
	printf("==================== Completed releasing devices ===================\n");
#endif

	free(kernel_src);

#if DEBUG==1
	printf("==================== Completed freeing kernel code ===================\n");
#endif

#if DEBUG==1
	printf(">>>>>>> cleanup complete\n");
#endif

	return;
}
