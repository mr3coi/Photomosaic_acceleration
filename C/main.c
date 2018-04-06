#include <stdio.h>
#include <stdlib.h>

#include "photomosaic.h"
#include "timer.h"
#include "qdbmp.h"

int main(int argc, char **argv) {
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
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
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
    int *idx = (int*)malloc(sheight * swidth * sizeof(int));
    timer_start(0);
    photomosaic(img, width, height, dataset, idx);
    printf("Elapsed time: %f sec\n", timer_stop(0));

    /*
     * construct output image
     */

    bmp = BMP_Create(width, height, depth);
    for (int sh = 0; sh < sheight; ++sh) {
        for (int sw = 0; sw < swidth; ++sw) {
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

    /*
     * free resources
     */

    free(img);
    free(dataset);
    free(idx);

    return 0;
}
