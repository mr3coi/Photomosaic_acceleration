#include "helper_func.h"

size_t read_kernel(const char *kernel_fname, char **kernel_dst) {
	FILE *kernel_fp;

	kernel_fp = fopen(kernel_fname,"r");
	assert(kernel_fp);
	fseek(kernel_fp,0L,SEEK_END);
	const size_t kernel_dst_len = ftell(kernel_fp);
	if (!kernel_dst_len) return 0;
	rewind(kernel_fp);

	*kernel_dst = (char *)malloc(kernel_dst_len+1);
	assert(*kernel_dst);
	assert(fread(*kernel_dst, kernel_dst_len, 1, kernel_fp));

	fclose(kernel_fp);

	return kernel_dst_len;
}

void checkError(cl_int error, int line) {
	if (error != CL_SUCCESS) {
		switch (error) {
			case CL_DEVICE_NOT_FOUND:                 printf("-- Error at %d:  Device not found.\n", line); break;
			case CL_DEVICE_NOT_AVAILABLE:             printf("-- Error at %d:  Device not available\n", line); break;
			case CL_COMPILER_NOT_AVAILABLE:           printf("-- Error at %d:  Compiler not available\n", line); break;
			case CL_MEM_OBJECT_ALLOCATION_FAILURE:    printf("-- Error at %d:  Memory object allocation failure\n", line); break;
			case CL_OUT_OF_RESOURCES:                 printf("-- Error at %d:  Out of resources\n", line); break;
			case CL_OUT_OF_HOST_MEMORY:               printf("-- Error at %d:  Out of host memory\n", line); break;
			case CL_PROFILING_INFO_NOT_AVAILABLE:     printf("-- Error at %d:  Profiling information not available\n", line); break;
			case CL_MEM_COPY_OVERLAP:                 printf("-- Error at %d:  Memory copy overlap\n", line); break;
			case CL_IMAGE_FORMAT_MISMATCH:            printf("-- Error at %d:  Image format mismatch\n", line); break;
			case CL_IMAGE_FORMAT_NOT_SUPPORTED:       printf("-- Error at %d:  Image format not supported\n", line); break;
			case CL_BUILD_PROGRAM_FAILURE:            printf("-- Error at %d:  Program build failure\n", line); break;
			case CL_MAP_FAILURE:                      printf("-- Error at %d:  Map failure\n", line); break;
			case CL_INVALID_VALUE:                    printf("-- Error at %d:  Invalid value\n", line); break;
			case CL_INVALID_DEVICE_TYPE:              printf("-- Error at %d:  Invalid device type\n", line); break;
			case CL_INVALID_PLATFORM:                 printf("-- Error at %d:  Invalid platform\n", line); break;
			case CL_INVALID_DEVICE:                   printf("-- Error at %d:  Invalid device\n", line); break;
			case CL_INVALID_CONTEXT:                  printf("-- Error at %d:  Invalid context\n", line); break;
			case CL_INVALID_QUEUE_PROPERTIES:         printf("-- Error at %d:  Invalid queue properties\n", line); break;
			case CL_INVALID_COMMAND_QUEUE:            printf("-- Error at %d:  Invalid command queue\n", line); break;
			case CL_INVALID_HOST_PTR:                 printf("-- Error at %d:  Invalid host pointer\n", line); break;
			case CL_INVALID_MEM_OBJECT:               printf("-- Error at %d:  Invalid memory object\n", line); break;
			case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:  printf("-- Error at %d:  Invalid image format descriptor\n", line); break;
			case CL_INVALID_IMAGE_SIZE:               printf("-- Error at %d:  Invalid image size\n", line); break;
			case CL_INVALID_SAMPLER:                  printf("-- Error at %d:  Invalid sampler\n", line); break;
			case CL_INVALID_BINARY:                   printf("-- Error at %d:  Invalid binary\n", line); break;
			case CL_INVALID_BUILD_OPTIONS:            printf("-- Error at %d:  Invalid build options\n", line); break;
			case CL_INVALID_PROGRAM:                  printf("-- Error at %d:  Invalid program\n", line); break;
			case CL_INVALID_PROGRAM_EXECUTABLE:       printf("-- Error at %d:  Invalid program executable\n", line); break;
			case CL_INVALID_KERNEL_NAME:              printf("-- Error at %d:  Invalid kernel name\n", line); break;
			case CL_INVALID_KERNEL_DEFINITION:        printf("-- Error at %d:  Invalid kernel definition\n", line); break;
			case CL_INVALID_KERNEL:                   printf("-- Error at %d:  Invalid kernel\n", line); break;
			case CL_INVALID_ARG_INDEX:                printf("-- Error at %d:  Invalid argument index\n", line); break;
			case CL_INVALID_ARG_VALUE:                printf("-- Error at %d:  Invalid argument value\n", line); break;
			case CL_INVALID_ARG_SIZE:                 printf("-- Error at %d:  Invalid argument size\n", line); break;
			case CL_INVALID_KERNEL_ARGS:              printf("-- Error at %d:  Invalid kernel arguments\n", line); break;
			case CL_INVALID_WORK_DIMENSION:           printf("-- Error at %d:  Invalid work dimensionsension\n", line); break;
			case CL_INVALID_WORK_GROUP_SIZE:          printf("-- Error at %d:  Invalid work group size\n", line); break;
			case CL_INVALID_WORK_ITEM_SIZE:           printf("-- Error at %d:  Invalid work item size\n", line); break;
			case CL_INVALID_GLOBAL_OFFSET:            printf("-- Error at %d:  Invalid global offset\n", line); break;
			case CL_INVALID_EVENT_WAIT_LIST:          printf("-- Error at %d:  Invalid event wait list\n", line); break;
			case CL_INVALID_EVENT:                    printf("-- Error at %d:  Invalid event\n", line); break;
			case CL_INVALID_OPERATION:                printf("-- Error at %d:  Invalid operation\n", line); break;
			case CL_INVALID_GL_OBJECT:                printf("-- Error at %d:  Invalid OpenGL object\n", line); break;
			case CL_INVALID_BUFFER_SIZE:              printf("-- Error at %d:  Invalid buffer size\n", line); break;
			case CL_INVALID_MIP_LEVEL:                printf("-- Error at %d:  Invalid mip-map level\n", line); break;
			case -1001:                               printf("-- Error at %d:  Code -1001: no GPU available?\n", line); break;
			default:                                  printf("-- Error at %d:  Unknown with code %d\n", line, error);
		}
		exit(1);
	}
}
