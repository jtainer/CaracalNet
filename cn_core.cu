// 
// CaracalNet core training/inference operations
// 
// 2023, Jonathan Tainer
// 

#include <cuda.h>
#include <math.h>
#include <stdint.h>
#include "cn_math.h"
#include "cn_core.h"

// CUDA backend
// Operations are called using CUDA kernel launching conventions
// Use CUDA backend only if cn_location == cn_device

// Perform forward pass on a single layer
__global__
void cn_device_forward_pass(cn_layer layer, float* input_vec, cn_activation activation) {

	// Figure out which neuron this thread is handling
	uint32_t tid = threadIdx.x + (blockIdx.x * blockDim.x);
	if (tid >= layer.num_nodes) return;

	// Multiply input vector by weight vector
	float* weight_vec = layer.weight + (tid * layer.num_inputs);
	float res = cn_vector_dot_product(weight_vec, input_vec, layer.num_inputs);

	// Apply bias and neuron activation function
	res += layer.bias[tid];
	switch (activation) {
	case cn_log:
		res = cn_sigmoid(res);
		break;
	}

	// Write results back to output vector
	layer.output[tid] = res;
}

// Calculate deltas for a layer
// 
// If target!=NULL then we assume layer_curr is the output layer
// and layer_next is not used.
// 
// If target==NULL then we assume layer_curr is a hidden layer
// and its deltas are calculated using layer_next.
__global__
void cn_device_update_deltas(cn_layer layer_curr, cn_layer layer_next, float* target) {

	// Figure out which neuron this thread is handling
	uint32_t tid = threadIdx.x + (blockIdx.x * blockDim.x);
	if (tid >= layer_curr.num_nodes) return;

	// Check if layer_curr is an output layer
	if (target != NULL) {

		// Compute delta of current output node
		float out = layer_curr.output[tid];
		float delta = out * (1.f - out) * (target[tid] - out);
		layer_curr.delta[tid] = delta;
	}

	// Otherwise layer_curr is a hidden layer
	else {
		float weighted_sum = 0.f;
		for (uint32_t i = 0; i < layer_next.num_nodes; i++) {
			weighted_sum += layer_next.delta[i] * layer_next.weight[(layer_next.num_inputs * i) + tid];
		}

		// Compute delta at current node
		float out = layer_curr.output[tid];
		float delta = out * (1.f - out) * weighted_sum;
		layer_curr.delta[tid] = delta;
	}
}

// Update weights for a layer
// 
// Deltas must have already been computed
__global__
void cn_device_update_weights(cn_layer layer, float* input_vec, float learning_rate) {
	uint32_t tid = threadIdx.x + (blockIdx.x * blockDim.x);
	if (tid >= layer.num_nodes) return;

	// Calculate gradient at current node
	float grad = learning_rate * layer.delta[tid];

	// Update weights of current node
	for (uint32_t i = 0; i < layer.num_inputs; i++) {
		layer.weight[(tid * layer.num_inputs) + i] += input_vec[i] * grad;
	}

	// Update bias
	layer.bias[tid] += grad;
}

// CPU backend
// Operations called using normal function calling conventions
// Used when cn_location == cn_host

// Perform forward pass on a single layer
__host__
void cn_host_forward_pass(cn_layer layer, float* input_vec, cn_activation activation) {
	// TODO: Use a thread pool to parallelize this instead of iterating.
	// Or maybe not, who does ML on CPU anyways. Just get a GPU lmao.
	for (uint32_t tid = 0; tid < layer.num_nodes; tid++) {

		// Multiply input vector by weight vector
		float* weight_vec = layer.weight + (tid * layer.num_inputs);
		float res = cn_vector_dot_product(weight_vec, input_vec, layer.num_inputs);
		
		// Apply bias and neuron activation function
		res += layer.bias[tid];
		switch(activation) {
		case cn_log:
			res = cn_sigmoid(res);
			break;
		}

		// Write results back to output vector
		layer.output[tid] = res;
	}
}

// Calculate deltas for a layer
__host__
void cn_host_update_deltas(cn_layer layer_curr, cn_layer layer_next, float* target) {
	for (uint32_t tid = 0; tid < layer_curr.num_nodes; tid++) {

		// Check if layer_curr is an output layer
		if (target != NULL) {

			// Compute delta of current output node
			float out = layer_curr.output[tid];
			float delta = out * (1.f - out) * (target[tid] - out);
			layer_curr.delta[tid] = delta;
		}

		// Otherwise layer_curr is a hidden layer
		else {

			float weighted_sum = 0.f;
			for (unsigned int i = 0; i < layer_next.num_nodes; i++) {
				weighted_sum += layer_next.delta[i] * layer_next.weight[(layer_next.num_inputs * i) + tid];
			}

			// Compute delta at current node
			float out = layer_curr.output[tid];
			float delta = out * (1.f - out) * weighted_sum;
			layer_curr.delta[tid] = delta;
		}
	}
}

// Update weights for a layer
__host__
void cn_host_update_weights(cn_layer layer, float* input_vec, float learning_rate) {
	for (uint32_t tid = 0; tid < layer.num_nodes; tid++) {

		// Calculate gradient at current node
		float grad = learning_rate * layer.delta[tid];

		// Update weights of current node
		for (unsigned int i = 0; i < layer.num_inputs; i++) {
			layer.weight[(tid * layer.num_inputs) + i] += input_vec[i] * grad;
		}
	
		// Update bias
		layer.bias[tid] += grad;
	}
}

