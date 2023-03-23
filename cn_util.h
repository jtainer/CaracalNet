// 
// Utilities for moving data between host and CUDA devices
// 
// 2023, Jonathan Tainer
// 

#ifndef CARACALNET_UTIL_H
#define CARACALNET_UTIL_H

#include "caracalnet.h"

void* cn_malloc(unsigned long bytes, cn_location location);

void* cn_copy(void* origin, unsigned long bytes, cn_location target_location, cn_location origin_location);

void cn_free(void* ptr, cn_location location);

void cn_memset(void* ptr, int val, unsigned long bytes, cn_location location);

void cn_memcpy(void* target, void* origin, unsigned long bytes, cn_location target_location, cn_location origin_location);

#endif
