#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

size_t read_kernel(const char *kernel_fname, char **kernel_dst);
void checkError(cl_int error, int line);
/*
void cleanup(cl_device_id device_ids[],
			 cl_context context,
			 cl_command_queue command_queue[],
			 cl_program program,
			 cl_kernel kernel[],
			 cl_mem buffers[][3],
			 char *kernel_src,
			 const unsigned int dev_num, unsigned log_flag);

void cleanup(cl_device_id device_ids[],
			 cl_context context,
			 cl_command_queue command_queue[3],
			 cl_program program,
			 cl_kernel kernel,
			 cl_mem **img_buf,
			 cl_mem *cifar_buf,
			 cl_mem **output,
			 char *kernel_src,
			 const unsigned int dev_num,
			 const unsigned int D,
			 unsigned log_flag);
*/
