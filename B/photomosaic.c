#include "photomosaic.h"

#include <CL/cl.h>
#include <limits.h>
#include "helper_func.h"
#include <string.h>
#include <omp.h>

#define TNUM 			64		// Maximum # of threads in a WG
#define TILE_SIZE		3072
#define TILE_H			32
#define TILE_W			32
#define TILE_C			3

void photomosaic(unsigned char *img, int width, int height, unsigned char *dataset, int *idx) {
	// Constants
	const unsigned int	swidth			= width / TILE_W,
		  				sheight			= height / TILE_H;
	if (sheight < 8 && swidth < 8) {
		unsigned char *thread_buf, tile_buf[TILE_C][TILE_H][TILE_W];
		unsigned int i, h, w, c, sw, sh;

		// Set number of threads to use in the program
		omp_set_num_threads(TNUM);

		#pragma omp parallel private(thread_buf, tile_buf, i, h, w, c, sw, sh)
		{
			// Allocate thread_buf for each thread
			thread_buf = (unsigned char *)malloc(TILE_H * TILE_W * TILE_C * swidth);

			#pragma omp for schedule(static) nowait
			for (sh = 0; sh < sheight; ++sh) {
#if DEBUG==1
				printf("================= sh : %d ================\n", sh);
#endif
				for (sw = 0; sw < swidth; ++sw) {
					// Copy each tile of the row to tile_buf w/ indices rearrangement
					for (h=0;h<TILE_H;h++) {
						for (w = 0; w < TILE_W; ++w) {
							for (c = 0; c < TILE_C; ++c) {
								tile_buf[c][h][w] = img[((sh * TILE_H + h) * width + (sw * TILE_W + w)) * TILE_C + c];
							}
						}
					}

					// Copy from tile_buf to thread_buf
					memcpy((void *)(thread_buf + sw*TILE_SIZE), tile_buf, TILE_SIZE);
				}
#if DEBUG==1
				printf("================= sh : %d, copy complete ================\n", sh);
#endif

				for (sw = 0; sw < swidth; ++sw) {
					int min_diff = INT_MAX, min_i = -1;

					// Iterate through all images in the CIFAR-10 dataset
					for (i = 0; i < 60000; ++i) {
						int diff = 0;

						// Compute L2 distance b/w tile and the given CIFAR-10 image
						for (c = 0; c < TILE_C; ++c) {
							for (h = 0; h < TILE_H; ++h) {
								for (w = 0; w < TILE_W; ++w) {
									int pixel_diff = (int)thread_buf[((sw*TILE_C+c)*TILE_H+h)*TILE_W+w] - (int)dataset[((i * TILE_C + c) * TILE_H + h) * TILE_W + w];
									diff += pixel_diff * pixel_diff;
								}
							}
						}
						// Update minimum if necessary
						if (min_diff > diff) {
							min_diff = diff;
							min_i = i;
						}
					}
					idx[sh * swidth + sw] = min_i;
				}
#if DEBUG==1
				printf("================= sh : %d, computation complete ================\n", sh);
#endif
			}	// end parallel for

			// Free thread_buf
			free(thread_buf);
		}	// end parallel
	}
	else {
		const size_t		DATA_SIZE		= 60000 * TILE_C * TILE_H * TILE_W;
		const size_t		DB_SIZE			= TILE_H * width * TILE_C;		// H * W * C
		const size_t		DB_OUT_COUNT	= swidth,
							DB_OUT_SIZE		= DB_OUT_COUNT * sizeof(int);

		// Other variables
		unsigned int i, j;

		// OpenCL variables
		cl_platform_id		platform;
		cl_device_id		device_id;
		cl_context			context;
		cl_command_queue	command_queue[3];
		cl_program			program;
		cl_kernel			kernel;
		cl_mem				img_buf[sheight],
							cifar_buf,
							output[sheight];
		cl_int				err;
#if DEBUG==1
		cl_int				status;
#endif
		cl_event			read_event[sheight+1],
							kernel_event[sheight],
							write_event[sheight];
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
		err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
#if DEBUG==1
		checkError(err, __LINE__);
#endif

		// Create a context for the platform and devices
		context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
#if DEBUG==1
		checkError(err, __LINE__);
#endif

		// Create a command_queue for each device
		for (j=0;j<3;j++) {
#if PROF==1
			command_queue[j] = clCreateCommandQueue(context,device_id,CL_QUEUE_PROFILING_ENABLE,&err);
#else
			command_queue[j] = clCreateCommandQueue(context,device_id,0,&err);
#endif
#if DEBUG==1
			checkError(err, __LINE__);
#endif
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

		err = clBuildProgram(program, 1, &device_id, args, NULL, NULL);
		if (err != CL_SUCCESS) {
			if (err == CL_BUILD_PROGRAM_FAILURE) {
				//Determine the size of the log
				size_t log_size;
				clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

				//Allocate memory for the log
				char *log = (char *) malloc(log_size);

				//Get the log
				clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

				//Print the log
				printf(">>> Build log for device %d:\n", 1);
				printf("%s\n", log);
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
		cifar_buf	= clCreateBuffer(context, CL_MEM_READ_ONLY, DATA_SIZE, NULL, &err);
		err = clEnqueueWriteBuffer(command_queue[0], cifar_buf, CL_FALSE,
								   0, DATA_SIZE, (void *)dataset,
								   0, NULL, write_event+sheight);

#if DEBUG==1
		err = clGetEventInfo(write_event[sheight], CL_EVENT_COMMAND_EXECUTION_STATUS,
							 sizeof(cl_int), &status, NULL);
		checkError(err, __LINE__);

		printf("cifar_write status : %d (Q:%d, S:%d, R:%d, C:%d)\n", status, CL_QUEUED, CL_SUBMITTED, CL_RUNNING, CL_COMPLETE);

		//if (status != CL_COMPLETE) err = clWaitForEvents(1, write_event + sheight);

		printf(">>>>>>> cifar writing complete\n");
#endif

		// Create and write into rest of buffers
		for (j=0;j<sheight;j++) {
			img_buf[j]	= clCreateBuffer(context, CL_MEM_READ_ONLY, DB_SIZE, NULL, &err);
			output[j]	= clCreateBuffer(context, CL_MEM_WRITE_ONLY, DB_OUT_SIZE, NULL, &err);

			err = clEnqueueWriteBuffer(command_queue[0], img_buf[j], CL_FALSE,
					0, DB_SIZE, (void *)(img + DB_SIZE*j),
					1, write_event+sheight, write_event+j);
		}

#if DEBUG==1
		printf(">>>>>>> rest writing complete\n");
#endif

		// Create buffer, write into them, and
		// set them as arguments to respective kernels, and run kernels
		for (j=0;j<sheight;j++) {
			err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)(img_buf+j));
			checkError(err, __LINE__);
			err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&cifar_buf);
			checkError(err, __LINE__);
			err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)(output+j));
			checkError(err, __LINE__);
			err = clEnqueueNDRangeKernel(command_queue[1], kernel, 3, NULL,
										 g_wsize, l_wsize,
										 1, write_event+j, kernel_event+j);
			checkError(err, __LINE__);
#if DEBUG==1
			printf("======= kernel for device id %d launched, DB stage : %d =====\n", 1, j);
#endif
		}

#if DEBUG==1
		printf(">>>>>>> enqueuing complete\n");
#endif

		// Retrieve results from buffer to C
		for (j=0;j<sheight;j++) {
			err = clEnqueueReadBuffer(command_queue[2], output[j], CL_TRUE,
					0, DB_OUT_SIZE, (void *)(idx + DB_OUT_COUNT*j),
					1, kernel_event+j, read_event+j);
#if DEBUG==1
			checkError(err, __LINE__);
#endif
		}

#if DEBUG==1
		printf(">>>>>>> enqueue reading complete\n");
#endif

		//err = clWaitForEvents(sheight, read_event);
#if DEBUG==1
		checkError(err, __LINE__);

		printf(">>>>>>> waiting complete\n");
#endif

		err = clFlush(command_queue[2]);
#if DEBUG==1
		checkError(err, __LINE__);
#endif

		err = clFinish(command_queue[2]);
#if DEBUG==1
		checkError(err, __LINE__);

		printf(">>>>>>> Flush & finish complete\n");
#endif

#if DEBUG==1
		for (i=0;i<sheight;i++) {
			err = clGetEventInfo(write_event[i], CL_EVENT_COMMAND_EXECUTION_STATUS,
								 sizeof(cl_int), &status, NULL);
			checkError(err, __LINE__);
			if (status != CL_COMPLETE) {
				printf("Write event %d is not yet complete.\n", i);
				err = clWaitForEvents(1, write_event+i);
				checkError(err, __LINE__);
			}

			err = clGetEventInfo(kernel_event[i], CL_EVENT_COMMAND_EXECUTION_STATUS,
								 sizeof(cl_int), &status, NULL);
			checkError(err, __LINE__);
			if (status != CL_COMPLETE) {
				printf("Kernel event %d is not yet complete.\n", i);
				err = clWaitForEvents(1, kernel_event+i);
				checkError(err, __LINE__);
			}

			err = clGetEventInfo(read_event[i], CL_EVENT_COMMAND_EXECUTION_STATUS,
								 sizeof(cl_int), &status, NULL);
			checkError(err, __LINE__);
			if (status != CL_COMPLETE) {
				printf("Read event %d is not yet complete.\n", i);
				err = clWaitForEvents(1, read_event+i);
				checkError(err, __LINE__);
			}
		}
		printf(">>>>>>> finished\n");
#endif

		// Cleanup
		const unsigned int D = sheight;
		const unsigned int dev_num = 1;
		const unsigned int log_flag = DEBUG;

		for (i=0;i<dev_num;i++) {
			err = clFlush(command_queue[i]);
			checkError(err, __LINE__);
			err = clFinish(command_queue[i]);
			checkError(err, __LINE__);
		}
		if (log_flag)
			printf("==================== Completed cleaning command_queues ===================\n");

		/*
		for (i=0;i<dev_num;i++) {
			err = clReleaseMemObject(cifar_buf[i]);
			checkError(err, __LINE__);
			for (j=0;j<D;j++) {
				err = clReleaseMemObject(img_buf[i][j]);
				checkError(err, __LINE__);
				err = clReleaseMemObject(output[i][j]);
				checkError(err, __LINE__);
			}
		}
		*/
		err = clReleaseMemObject(cifar_buf);
		checkError(err, __LINE__);
		for (j=0;j<D;j++) {
			err = clReleaseMemObject(img_buf[j]);
			checkError(err, __LINE__);
			err = clReleaseMemObject(output[j]);
			checkError(err, __LINE__);
		}
		if (log_flag)
			printf("==================== Completed clReleaseMemObject ===================\n");

		err = clReleaseProgram(program);
		checkError(err, __LINE__);
		if (log_flag)
			printf("==================== Completed clReleaseProgram ===================\n");

		err = clReleaseKernel(kernel);
		checkError(err, __LINE__);
		if (log_flag)
			printf("==================== Completed clReleaseKernel ===================\n");

		for (i=0;i<dev_num;i++) {
			for (j=0;j<3;j++) {
				err = clReleaseCommandQueue(command_queue[j]);
				checkError(err, __LINE__);
			}
		}
		if (log_flag)
			printf("==================== Completed clReleaseCommandQueue ===================\n");

		err = clReleaseContext(context);
		checkError(err, __LINE__);
		if (log_flag)
			printf("==================== Completed clReleaseContext ===================\n");

		//for (i=0;i<dev_num;i++) { err = clReleaseDevice(device_ids[i]); checkError(err, __LINE__); }
		err = clReleaseDevice(device_id);
		checkError(err, __LINE__);
		if (log_flag)
			printf("==================== Completed releasing devices ===================\n");

		free(kernel_src);

		if (log_flag)
			printf("==================== Completed freeing kernel code ===================\n");

#if DEBUG==1
		printf(">>>>>>> cleanup complete\n");
#endif
	}

	return;
}
