#pragma once

void photomosaic_gpu(unsigned char *img, int width, int height, unsigned char *dataset, int *idx);
void photomosaic_cpu(unsigned char *img, int width, int height, unsigned char *dataset, int *idx);
