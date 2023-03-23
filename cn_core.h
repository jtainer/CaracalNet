// 
// CaracalNet core training/inference operations
// 
// 2023, Jonathan Tainer
// 

#ifndef CARACALNET_CORE_H
#define CARACALNET_CORE_H

#include "caracalnet.h"

//*****************************************************************************
// CUDA backend
// Operations are called using CUDA kernel launching conventions
// Use CUDA backend only if cn_location == cn_device
//*****************************************************************************

// Perform forward pass on a single layer
__global__
void cn_device_forward_pass(cn_layer layer, float* input_vec, cn_activation activation);

// Calculate deltas for a layer
__global__
void cn_device_update_deltas(cn_layer layer_curr, cn_layer layer_next, float* target);

// Update weights for a layer
__global__
void cn_device_update_weights(cn_layer layer, float* input, float learning_rate);

//*****************************************************************************
// CPU backend
// Operations called using normal function calling conventions
// Used when cn_location == cn_host
//*****************************************************************************

// Perform forward pass on a single layer
__host__
void cn_host_forward_pass(cn_layer layer, float* input_vec, cn_activation activation);

// Calculate deltas for a layer
__host__
void cn_host_update_deltas(cn_layer layer_curr, cn_layer layer_next, float* target);

// Update weights for a layer
__host__
void cn_host_update_weights(cn_layer layer, float* input, float learning_rate);


#endif
