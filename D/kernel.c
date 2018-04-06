#define TILE_SIZE 3072
#define VEC 16
#define TILE_H 32
#define TILE_W 32
#define TILE_C 3
#define DATASET_SIZE 60000

__kernel void pm_kernel(__global uchar16 *img,
						__global uchar16 *cifar,
						__global int *output) {
	const unsigned int thread_idx = get_local_id(2);
	const unsigned int tile_idx = get_global_id(1);
	const unsigned int swidth = get_global_size(1);
	const unsigned int TPL = TNUM / TILE_H;				// Threads per each layer of tile
	const unsigned int VPL = TILE_W * TILE_C / VEC;		// Vectors per layer
	const unsigned int VPT = VPL / TPL;					// Vectors per thread
	const unsigned int WPT = TILE_W / TPL;				// widths per thread
	const unsigned int IPT = TILE_SIZE / TNUM;			// Items per thread
	unsigned int thread_loc_v, image_loc_v, thread_loc;
	unsigned int c, h, w, w_v, i, j;
	int thread_diff;

	__local int curr_diff[TNUM], min_diff;
	__local unsigned int min_i;
	__local union { uchar s[TILE_SIZE]; uchar16 v[TILE_SIZE/VEC]; } tile, data;

	h = thread_idx / TPL;
	w_v = (thread_idx % TPL) * VPT;
	w = (thread_idx % TPL) * WPT;
	min_i = DATASET_SIZE;
	min_diff = 256*256*TILE_SIZE;

	// Copy tile from global memory
	for (i=0;i<VPT;i++)
		tile.v[h * VPL + w_v + i] = img[(h * swidth + tile_idx) * VPL + w_v + i];

		// Barrier unnecessary (Each thread copies & transforms its allocated portion)

	// Transform copied tile
	for (i=w;i<w+WPT;i++) {
		for (c=0;c<3;c++)
			data.s[(c * 32 + h) * 32 + i] = tile.s[(h * 32 + i) * 3 + c];
	}

		// Barrier for copying & transforming
	barrier(CLK_LOCAL_MEM_FENCE);

		// For computational efficiency
	thread_loc_v = thread_idx * VPT;
	thread_loc = thread_loc_v * VEC;
	image_loc_v = 0;
	
	// Copy transformed tile back to original place
	//for (i=w_v;i<w_v+VPT;i++) tile.v[h * VPL + i] = data.v[h  * VPL + i];	// TODO Delete
	for (i=0;i<VPT;i++) tile.v[thread_loc_v + i] = data.v[thread_loc_v + i];

		// Barrier for copying
	//barrier(CLK_LOCAL_MEM_FENCE);

	// Iterate through CIFAR-10 images and conduct comparison
	for (i=0;i<DATASET_SIZE;i++) {
		// Initialize difference array
		curr_diff[thread_idx] = 0;

		// Copy image into local memory
		for (j=0;j<VPT;j++)
			data.v[thread_loc_v + j] = cifar[image_loc_v + thread_loc_v + j];

		// Compute diff. w/ current CIFAR-10 image for each thread
		for (j=0;j<IPT;j++) {
			//thread_diff = (int)(tile.s[(c*32+h)*32+w]) - (int)(data.s[(c*32+h)*32+w]);	// TODO Delete
			thread_diff = (int)(tile.s[thread_loc + j]) - (int)(data.s[thread_loc + j]);
			curr_diff[thread_idx] += thread_diff * thread_diff;
		}

		// Barrier for computing sums of sq. diffs
		barrier(CLK_LOCAL_MEM_FENCE);

		// Reduce the diffs of threads into one value
		for (j=TNUM/2;j>=1;j/=2) {
			if (thread_idx < j) curr_diff[thread_idx] += curr_diff[thread_idx + j];
		}

		if (thread_idx == 0) {
			// Reduce the diffs of threads into one value
			// (thread 0 stores the final result in its 'thread_diff'
//			__local int16 *red1 = (int16 *)curr_diff;
//			for (j=TNUM/VEC/2;j>=1;j/=2) { for (k=0;k<j;k++) red1[k] += red1[k+j]; }
//			__local int8 *red2 = (int8 *)curr_diff;
//			red2[0] += red2[1];
//			__local int4 *red3 = (int4 *)curr_diff;
//			red3[0] += red3[1];
//			__local int2 *red4 = (int2 *)curr_diff;
//			red4[0] += red4[1];

			thread_diff = curr_diff[0];

			//printf("tile_idx = %u, i = %u, curr_diff = %u, min_i = %u, min_diff = %u, curr_diff < min_diff = %d\n", tile_idx, i, thread_diff, min_i, min_diff, thread_diff<min_diff);	// TODO Delete

			// Update 'min_diff' and 'min_i' if applicable
			if (thread_diff < min_diff) {
				min_i = i;
				min_diff = thread_diff;
			}
		}

		// Update dataset image location
		image_loc_v += TILE_SIZE / VEC;
	}

	// Barrier not needed (only thread 0 manipulates 'min_i')

	// Copy 'min_i' to global memory
	if (thread_idx == 0) {
		output[tile_idx] = min_i;
		//printf(">>>>> Final : tile_idx = %u, min_diff = %u, min_i = %u\n", tile_idx, min_diff, min_i);		// TODO Delete
	}

	return;
}
