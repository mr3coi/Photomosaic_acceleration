================================== README (for Github) =================================

This is a class final project on accelerating a naive sequential implementation of
a program that generates a photomosaic output from an input image using images from
the CIFAR-10 dataset, under various circumstances described below:

A) Single machine, CPU only
B) Single machine, single GPU
C) Single machine, 4 GPUs
D) 4 nodes on a server, 4 GPUs
E) 4 nodes on a server, 4 GPUs, using SnuCL mandatory

Algorithmic approach of any sort is not allowed - only acceleration via parallel
computing techniques is allowed.

Requirements (may have more):
- OpenCL 2.0
- OpenMP 3.1
- MPI
- SnuCL (for E) - refer to aces.snu.ac.kr

To run example cases (within each subdirectory A~E):
	make
	./main ../example/inputX.bmp ./output/outputX.bmp (X: 0/1)

Note:
- Only 24-Bit BMP files are allowed as inputs.
- Ignore the parts in Makefiles w/ 'thorq's, as they are specifically for the environment
	students used during class.
- The codes have been run on AMD GPUs and thus do not exploit CUDA or related libraries.
- cifar-10 binary dataset has been removed due to file size limits. Should originally
	be located in 'data' folder, with name 'cifar-10.bin'.

Important:
- This repository is solely for archiving purpose.
	Running the code outside the environment provided during class has not been tested,
	so it is NOT recommended to try running the codes in your computer.
	(You may try and debug on your own if you wish)

============================================================================================
