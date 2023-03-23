// 
// Utilities for moving data between host and CUDA devices
// 
// 2023, Jonathan Tainer
// 

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include "cn_util.h"

void* cn_malloc(unsigned long bytes, cn_location location) {
	void* ptr = NULL;
	switch (location) {
	case cn_host:
		ptr = malloc(bytes);
		break;
	case cn_device:
		cudaMalloc(&ptr, bytes);
		break;
	}
	return ptr;
}

void* cn_copy(void* origin, unsigned long bytes, cn_location target_location, cn_location origin_location) {
	void* target = cn_malloc(bytes, target_location);
	cn_memcpy(target, origin, bytes, target_location, origin_location);
	return target;
}

void cn_free(void* ptr, cn_location location) {
	switch (location) {
	case cn_host:
		free(ptr);
		break;
	case cn_device:
		cudaFree(ptr);
		break;
	}
}

void cn_memset(void* ptr, int val, unsigned long bytes, cn_location location) {
	switch (location) {
	case cn_host:
		memset(ptr, val, bytes);
		break;
	case cn_device:
		cudaMemset(ptr, val, bytes);
		break;

	}
}

void cn_memcpy(void* target, void* origin, unsigned long bytes, cn_location target_location, cn_location origin_location) {
	cudaMemcpyKind transfer_type;
	if (origin_location == cn_host && target_location == cn_device) {
		transfer_type = cudaMemcpyHostToDevice;
	}
	else if (origin_location == cn_device && target_location == cn_host) {
		transfer_type = cudaMemcpyDeviceToHost;
	}
	else if (origin_location == cn_device && target_location == cn_device) {
		transfer_type = cudaMemcpyDeviceToDevice;
	}
	else {
		transfer_type = cudaMemcpyHostToHost;
	}

	cudaMemcpy(target, origin, bytes, transfer_type);
}
