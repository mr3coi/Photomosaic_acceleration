#include "photomosaic.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#define TILE_H		32
#define TILE_W		32
#define TILE_C		3
#define TNUM		128
#define TILE_SIZE	3072

void photomosaic(unsigned char *img, int width, int height, unsigned char *dataset, int *idx) {
    const int swidth = width / TILE_W, sheight = height / TILE_H;
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
							// The other approach described in report
							// The order of loops of 'sw' and 'h' need to be changed, too
							//thread_buf[((sw * TILE_C + c) * TILE_H + h) * TILE_W + w] = img[((sh * TILE_H + h) * width + (sw * TILE_W + w)) * TILE_C + c];
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
