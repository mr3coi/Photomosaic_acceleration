#include <stdio.h>
#include <stdlib.h>
/* ================================================================ */
#include "mpi.h"
#include <assert.h>

#define GPU_NUM			4
#define THRES_PER_GPU	1		// Least number of rows for each GPU (o/w all done by CPU)
#define CHECK_VALID(err) if (DEBUG) assert(err == MPI_SUCCESS);
/* ================================================================ */

#include "photomosaic.h"
#include "timer.h"
#include "qdbmp.h"

int main(int argc, char **argv) {
	int i, j;

    if (argc != 3) {
        printf("Usage : %s [input.bmp] [output.bmp]\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    /*
     * read input image
     */

    BMP *bmp = BMP_ReadFile(argv[1]);
    BMP_CHECK_ERROR(stderr, EXIT_FAILURE);

    int width = BMP_GetWidth(bmp);
    int height = BMP_GetHeight(bmp);
    int depth = BMP_GetDepth(bmp);
    printf("image read success; image = %s, width = %d, height = %d, depth = %d\n", argv[1], width, height, depth);
    if (width % 32 != 0 || height % 32 != 0) {
        printf("width and height should be multiple of 32.\n");
        exit(EXIT_FAILURE);
    }
    if (depth != 24) {
        printf("depth should be 24.\n");
        exit(EXIT_FAILURE);
    }

    unsigned char *img = (unsigned char*)malloc(height * width * 3), *it = img;
    for (i = 0; i < height; ++i) {
        for (j = 0; j < width; ++j) {
            BMP_GetPixelRGB(bmp, j, i, it, it + 1, it + 2);
            it += 3;
        }
    }

    BMP_Free(bmp);

    /*
     * read cifar-10 dataset
     */

    unsigned char *dataset = (unsigned char*)malloc(60000 * 3 * 32 * 32);
    FILE *fin = fopen("../data/cifar-10.bin", "rb");
    if (!fin) {
        printf("cifar-10.bin not found\n");
        exit(EXIT_FAILURE);
    }
    fread(dataset, 1, 60000 * 3 * 32 * 32, fin);
    fclose(fin);

    printf("dataset read success\n");

    /*
     * photomosaic computation
     */

    int swidth = width / 32, sheight = height / 32;

    timer_start(0);
	/* ================================================================ */
	/* Variables for MPI */
	MPI_Comm in_comm;			// Intra-node communicator
	int rank,					// Rank of process in MPI_COMM_WORLD (all nodes)
		size,					// # processes in MPI_COMM_WORLD
		in_rank,				// Rank of process in 'in_comm'
		in_size,				// # processes in in_comm
		in_node,				// The number of node that the current process belongs to
		node_num;				// # nodes in this program
	int *indiv_idx,				// Container for individual results
		*idx = NULL;			// Container for final rsult. Only malloc'd in thread 0
	int flag, err;

	/* Initialize MPI and collect per-process information */
	err = MPI_Init(&argc, &argv);							CHECK_VALID(err);
	err = MPI_Initialized(&flag);							CHECK_VALID(err);
	assert(flag);
	err = MPI_Comm_rank(MPI_COMM_WORLD, &rank);				CHECK_VALID(err);
	err = MPI_Comm_size(MPI_COMM_WORLD, &size);				CHECK_VALID(err);

	err = MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
							  MPI_INFO_NULL, &in_comm);		CHECK_VALID(err);
	err = MPI_Comm_rank(in_comm, &in_rank);					CHECK_VALID(err);
	err = MPI_Comm_size(in_comm, &in_size);					CHECK_VALID(err);
	in_node = rank / in_size;
	node_num = size / in_size;

	/* Allocate rows among nodes */
	int node_num_rows[node_num], node_cum_rows[node_num];
	const size_t row_size = 32 * width * 3;

	i = sheight / node_num;
	for (j=0;j<node_num;j++) node_num_rows[j] = i;
	i = sheight - i * node_num;
	for (j=0;j<i;j++) node_num_rows[j]++;
	node_cum_rows[0] = 0;
	for (j=1;j<node_num;j++)
		node_cum_rows[j] = node_cum_rows[j-1] + node_num_rows[j-1];

	/* Allocate rows among devices in a node */
	int dev_num_rows[in_size], dev_cum_rows[in_size];
	/*
	j = node_num_rows[in_node] / GPU_NUM;
	if (j >= THRES_PER_GPU) {
		dev_num_rows[1] = GPU_NUM * j;
		dev_num_rows[0] = node_num_rows[in_node] - GPU_NUM * j;
		if (dev_num_rows[0] == 0 && j >= THRES_PER_GPU+1) {
			dev_num_rows[0] += GPU_NUM;
			dev_num_rows[1] -= GPU_NUM;
		}
	}
	else {
		dev_num_rows[0] = node_num_rows[in_node];
		dev_num_rows[1] = 0;
	}
	*/
	dev_num_rows[0] = 0;
	dev_num_rows[1] = node_num_rows[in_node];

	dev_cum_rows[0] = 0;
	dev_cum_rows[1] = dev_num_rows[0];

#if DEBUG==1
	printf("Proc %d (%d, %d) : (node) %d (%d), (dev) %d (%d)\n",
			rank, in_rank, in_node, node_num_rows[in_node], node_cum_rows[in_node],
			dev_num_rows[in_rank], dev_cum_rows[in_rank]);
#endif

	/* Allocate work among processes */
	indiv_idx = (int *)malloc(sizeof(int) * swidth * dev_num_rows[in_rank]);
    if (dev_num_rows[in_rank] > 0) {
		if (in_rank == 0) {
#if DEBUG==1
			printf("(CPU) Proc %d (%d) : (in rows) img offset : %u, height : %u\n",
					rank, in_rank,
					node_cum_rows[in_node] + dev_cum_rows[in_rank],
					dev_num_rows[in_rank]);
#endif
#if DEBUG==1
			timer_start(1);
#endif
			photomosaic_cpu(img + row_size * (node_cum_rows[in_node] + dev_cum_rows[in_rank]),
							width, dev_num_rows[in_rank] * 32,
							dataset, indiv_idx);
#if DEBUG==1
			printf(">>> Proc %d (%d), duration : %f\n", rank, in_rank, timer_stop(1));
#endif
		}
		else if (dev_num_rows[in_rank] > 0) {
#if DEBUG==1
			printf("(GPU) Proc %d (%d) : (in rows) img offset : %u, height : %u\n",
					rank, in_rank,
					node_cum_rows[in_node] + dev_cum_rows[in_rank],
					dev_num_rows[in_rank]);
#endif
#if DEBUG==1
			timer_start(1);
#endif
			photomosaic_gpu(img + row_size * (node_cum_rows[in_node] + dev_cum_rows[in_rank]),
							width, dev_num_rows[in_rank] * 32,
							dataset, indiv_idx);
#if DEBUG==1
			printf(">>> Proc %d (%d), duration : %f\n", rank, in_rank, timer_stop(1));
#endif
		}
	}

	/* Retrieve results from all devices */
	err = MPI_Barrier(MPI_COMM_WORLD);								CHECK_VALID(err);

	if (rank == 0) {
		idx = (int *)malloc(sizeof(int) * swidth * sheight);
		int *displs = (int *)malloc(sizeof(int)*size);
		int *recvcounts = (int *)malloc(sizeof(int)*size);
		MPI_Request	requests[size-1];
		MPI_Status	statuses[size-1];

		/* Compute recvcount & displs for Igatherv */
		recvcounts[0] = dev_num_rows[in_rank];
		for (i=1;i<size;i++) {
			MPI_Irecv((void *)(recvcounts+i), 1, MPI_INTEGER, i, i, MPI_COMM_WORLD, requests+i-1);
			CHECK_VALID(err);
		}
		err = MPI_Testall(size-1, requests, &flag, statuses);		CHECK_VALID(err);
		if (!flag) MPI_Waitall(size-1, requests, statuses);
		for(i=0;i<size;i++) recvcounts[i] *= swidth;

		displs[0] = 0;
		for (i=1;i<size;i++) displs[i] = displs[i-1] + recvcounts[i-1];

#if DEBUG==1
		printf("displs: ");
		for (i=0;i<8;i++) printf("%d ", displs[i]);
		printf("\nrecvcounts: ");
		for (i=0;i<8;i++) printf("%d ", recvcounts[i]);
		printf("\n");
#endif

		/* Continue when displs and recvcounts are ready */
		MPI_Barrier(MPI_COMM_WORLD);

		if (dev_num_rows[in_rank] > 0) {
#if DEBUG==1
			printf(">>> Proc : %d (%d), sendcount : %d, indiv_idx[16] : %d\n",
					rank, in_rank, dev_num_rows[in_rank], indiv_idx[16]);	// TODO Delete
#endif
		}

		/* Collect results from other processes */
		err = MPI_Gatherv(indiv_idx, dev_num_rows[in_rank] * swidth, MPI_INTEGER,
						  idx, recvcounts, displs, MPI_INTEGER,
						  0, MPI_COMM_WORLD);						CHECK_VALID(err);
		free(displs); free(recvcounts);
	}
	else {
		/* Send information for displ & recvcount */
		err = MPI_Send((void *)(dev_num_rows+in_rank), 1, MPI_INTEGER,
						0, rank, MPI_COMM_WORLD);					CHECK_VALID(err);

		/* Continue when displs and recvcounts are ready */
		MPI_Barrier(MPI_COMM_WORLD);

		if (dev_num_rows[in_rank] > 0) {
#if DEBUG==1
			printf(">>> Proc : %d (%d), sendcount : %d, indiv_idx[16] : %d\n",
					rank, in_rank, dev_num_rows[in_rank], indiv_idx[16]);	// TODO Delete
#endif
		}

		/* Send result to root process */
		err = MPI_Gatherv(indiv_idx, dev_num_rows[in_rank] * swidth, MPI_INTEGER,
						  NULL, NULL, NULL, NULL,
						  0, MPI_COMM_WORLD);						CHECK_VALID(err);
	}

    /*
     * construct output image
     */

	if (rank ==0) {
    	printf("Elapsed time: %f sec\n", timer_stop(0));

		bmp = BMP_Create(width, height, depth);
		for (int sh = 0; sh < sheight; ++sh) {
			for (int sw = 0; sw < swidth; ++sw) {
				//printf("sh:%d, sw:%d, val : %d\n", sh, sw, idx[sh*swidth+sw]);	// TODO Delete
				for (int h = 0; h < 32; ++h) {
					for (int w = 0; w < 32; ++w) {
						unsigned char rgb[3];
						for (int c = 0; c < 3; ++c) {
							rgb[c] = dataset[((idx[sh * swidth + sw] * 3 + c) * 32 + h) * 32 + w];
						}
						BMP_SetPixelRGB(bmp, sw * 32 + w, sh * 32 + h, rgb[0], rgb[1], rgb[2]);
					}
				}
			}
		}
		BMP_WriteFile(bmp, argv[2]);
		BMP_Free(bmp);
		printf("image write success\n");

    	free(idx);
	}

    /*
     * free resources
     */

    free(img);
    free(dataset);
	free(indiv_idx);

	/* Finalize MPI */
	MPI_Finalize();
	MPI_Finalized(&flag);
	assert(flag);
	/* ================================================================ */

    return 0;
}
