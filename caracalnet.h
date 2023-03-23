// 
// CaracalNet: my most recent attempt at making a usable multilayer perceptron
//
// 2023, Jonathan Tainer
//

#ifndef CARACALNET_H
#define CARACALNET_H

#include <stdint.h>

// CaracalNet uses data structures that can exist in either device or host memory,
// and automatically moves data objects between each memory system when needed.
typedef enum cn_location {
	cn_host = 0,
	cn_device
} cn_location;

// CaracalNet provides a variety of common neuron activation functions.
typedef enum cn_activation {
	cn_log = 0,
	cn_tanh,
	cn_relu,
	cn_silu,
	cn_elu
} cn_activation;

// Layers are composed of neurons which each contain a weight vector and a bias.
// The output of a node is calculated by taking the dot product of the input vector
// and the weight vector, adding the bias for that neuron, and applying an
// activation function. This is simplified by arranging the weight vectors of
// several neurons into a single weight matrix which is multiplied by the input
// vector.
typedef struct cn_layer {
	// If location == cn_device, then weight, output, and delta buffers are allocated
	// in device memory. Entire layer structure must be passed into CUDA kernel.
	cn_location location;
	float* weight;	// Multiply input vector by weight matrix
	float* bias;	// Add bias vector
	float* output;	// Apply activation function to get output vector
	float* delta;	// Compare output and target vectors to calculate deltas
	uint32_t num_nodes;
	uint32_t num_inputs;
} cn_layer;

// A network is just a sequence of layers that feed into each other.
typedef struct cn_network {
	cn_location location;
	cn_layer* layer;
	uint32_t num_layers;
	uint32_t num_inputs;
	uint32_t num_outputs;
} cn_network;

#if defined(__cplusplus)
extern "C" {
#endif

// Layer handling operations:
// These probably shouldn't be exposed since they are mainly intended for internal use.
// But I will leave them here for now :)

cn_layer cn_layer_create(uint32_t num_nodes, uint32_t num_inputs, cn_location location);

cn_layer cn_layer_copy(cn_layer* origin, cn_location target_location);

void cn_layer_destroy(cn_layer* layer);

// Network handling operations:
// These are what you should actually use for your application

cn_network cn_network_create(uint32_t num_inputs, uint32_t num_outputs, uint32_t num_layers, uint32_t layer_width, cn_location location);

cn_network cn_network_copy(cn_network* origin, cn_location target_location);

void cn_network_destroy(cn_network* network);

// Training and inference functions

// Input and output buffers may be in host or device memory, as indicated by data_location,
// but they must both be in the same memory space. The network can be in a different memory
// space from the input and output buffers.
//
// If output buffer is NULL then the result of the forward pass is stored in the final
// layer of the network, but not copied anywhere else.
void cn_inference(cn_network* network, float* input, float* output, cn_location data_location);

// Updates neural network weights based on the target vector and the results of the
// previous forward pass.
void cn_backprop(cn_network* network, float* input, float* target, cn_location data_location);

#if defined(__cplusplus)
}
#endif


#endif
