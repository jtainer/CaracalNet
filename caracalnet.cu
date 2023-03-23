// 
// CaracalNet: my most recent attempt at making a usable multilayer perceptron
// 
// 2023, Jonathan Tainer
// 

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include "caracalnet.h"
#include "cn_core.h"
#include "cn_util.h"

cn_layer cn_layer_create(uint32_t num_nodes, uint32_t num_inputs, cn_location location) {
	// Allocate either device or host memory for all buffers needed by a layer.
	cn_layer layer = { location, NULL, NULL, NULL, NULL, num_nodes, num_inputs };
	
	layer.weight = (float*) cn_malloc(sizeof(float) * num_nodes * num_inputs, location);
	layer.bias = (float*) cn_malloc(sizeof(float) * num_nodes, location);
	layer.output = (float*) cn_malloc(sizeof(float) * num_nodes, location);
	layer.delta = (float*) cn_malloc(sizeof(float) * num_nodes, location);
	cn_memset(layer.weight, 0, sizeof(float) * num_nodes * num_inputs, location);
	
	return layer;
}

cn_layer cn_layer_copy(cn_layer* origin, cn_location target_location) {
	cn_layer target = cn_layer_create(origin->num_nodes, origin->num_inputs, target_location);
	cn_memcpy(target.weight, origin->weight, sizeof(float) * origin->num_nodes * origin->num_inputs, target.location, origin->location);
	cn_memcpy(target.bias, origin->bias, sizeof(float) * origin->num_nodes, target.location, origin->location);
	cn_memcpy(target.output, origin->output, sizeof(float) * origin->num_nodes, target.location, origin->location);
	cn_memcpy(target.delta, origin->delta, sizeof(float) * origin->num_nodes, target.location, origin->location);

	return target;
}

void cn_layer_destroy(cn_layer* layer) {
	cn_free(layer->weight, layer->location);
	cn_free(layer->bias, layer->location);
	cn_free(layer->output, layer->location);
	cn_free(layer->delta, layer->location);

	layer->weight = NULL;
	layer->bias = NULL;
	layer->output = NULL;
	layer->delta = NULL;
	layer->num_nodes = 0;
	layer->num_inputs = 0;
}

cn_network cn_network_create(uint32_t num_inputs, uint32_t num_outputs, uint32_t num_layers, uint32_t layer_width, cn_location location) {
	cn_network network = { location, NULL, num_layers, num_inputs, num_outputs };
	network.layer = (cn_layer*) malloc(sizeof(cn_layer) * num_layers);
	for (int32_t i = 0; i < num_layers; i++) {
		unsigned int tmp_num_inputs = (i==0) ? num_inputs : layer_width;
		unsigned int tmp_num_nodes = (i==num_layers-1) ? num_outputs : layer_width;
		network.layer[i] = cn_layer_create(tmp_num_nodes, tmp_num_inputs, location);
	}
	return network;
}

cn_network cn_network_copy(cn_network* origin, cn_location target_location) {
	cn_network target = *origin;
	target.location = target_location;
	target.layer = (cn_layer*) malloc(sizeof(cn_layer) * origin->num_layers);
	for (int32_t i = 0; i < origin->num_layers; i++) {
		target.layer[i] = cn_layer_copy(&origin->layer[i], target_location);
	}
	return target;
}

void cn_network_destroy(cn_network* network) {
	for (int32_t i = 0; i < network->num_layers; i++) {
		cn_layer_destroy(&network->layer[i]);
	}
	network->layer = NULL;
	network->num_layers = 0;
	network->num_inputs = 0;
	network->num_outputs = 0;
}

static void cn_host_inference(cn_network* network, float* input, float* output) {
	// If using CPU backend, we assume the buffers point to host memory system, not VRAM
	float* tmp_input = input;
	for (int32_t i = 0; i < network->num_layers; i++) {
		cn_host_forward_pass(network->layer[i], tmp_input, cn_log);
		tmp_input = network->layer[i].output;
	}
	memcpy(output, tmp_input, sizeof(float) * network->num_outputs);
}

static void cn_device_inference(cn_network* network, float* input, float* output, cn_location data_location) {
	// Copy input buffer into GPU memory if needed
	float* dev_input = NULL;
	float* dev_output = network->layer[network->num_layers-1].output;
	if (data_location == cn_host) {
		dev_input = (float*) cn_copy(input, sizeof(float) * network->num_inputs, cn_device, cn_host);
	}
	else {
		dev_input = input;
	}

	// Run forward pass
	float* tmp_input = dev_input;
	for (int32_t i = 0; i < network->num_layers; i++) {
		uint32_t num_threads = network->layer[i].num_nodes;
		const uint32_t block_size = 32;
		uint32_t num_blocks = num_threads / block_size;
		if (num_blocks * block_size < num_threads) num_blocks++;
		cn_device_forward_pass<<<num_blocks, block_size>>>(network->layer[i], tmp_input, cn_log);
		tmp_input = network->layer[i].output;
	}

	// Delete temporary input buffer
	if (data_location == cn_host) {
		cudaFree(dev_input);
	}

	// Write back output buffer
	cn_memcpy(output, dev_output, sizeof(float) * network->num_outputs, data_location, cn_device);

}

void cn_inference(cn_network* network, float* input, float* output, cn_location data_location) {
	switch (network->location) {
	case cn_host:
		cn_host_inference(network, input, output);
		break;
	case cn_device:
		cn_device_inference(network, input, output, data_location);
		break;
	}
}

static void cn_host_backprop(cn_network* network, float* input, float* target) {
	// Assume input and target buffers are in host memory

	// Calculate deltas
	for (int32_t i = network->num_layers - 1; i >= 0; i--) {
		cn_layer layer_curr = network->layer[i];
		cn_layer layer_next = (i < network->num_layers - 1) ? network->layer[i + 1] : network->layer[i];
		float* tmp_target = (i == network->num_layers - 1) ? target : NULL;
		cn_host_update_deltas(layer_curr, layer_next, tmp_target);
	}
	
	// Update weights
	for (int32_t i = 0; i < network->num_layers; i++) {
		cn_layer tmp_layer = network->layer[i];
		float* tmp_input = (i == 0) ? input : network->layer[i-1].output;
		cn_host_update_weights(tmp_layer, tmp_input, 1.f);
	}
}

static void cn_device_backprop(cn_network* network, float* input, float* target, cn_location data_location) {
	// Copy target vector into GPU memory if needed
	float* dev_target = NULL;
	if (data_location == cn_host) {
		dev_target = (float*) cn_copy(target, sizeof(float) * network->num_outputs, cn_device, cn_host);
	}
	else {
		dev_target = target;
	}

	// Calculate deltas
	for (int32_t i = network->num_layers - 1; i >= 0; i--) {
		cn_layer layer_curr = network->layer[i];
		cn_layer layer_next = (i < network->num_layers - 1) ? network->layer[i + 1] : network->layer[i];
		float* tmp_target = (i == network->num_layers - 1) ? dev_target : NULL;
		
		uint32_t num_threads = layer_curr.num_nodes;
		const uint32_t block_size = 32;
		uint32_t num_blocks = num_threads / block_size;
		if (num_blocks * block_size < num_threads) num_blocks++;
		cn_device_update_deltas<<<num_blocks, block_size>>>(layer_curr, layer_next, tmp_target);
	}

	// Unload target buffer
	if (data_location == cn_host) {
		cudaFree(dev_target);
	}
	
	// Copy input vector into GPU memory if needed
	float* dev_input = NULL;
	if (data_location == cn_host) {
		dev_input = (float*) cn_copy(input, sizeof(float) * network->num_inputs, cn_device, cn_host);
	}
	else {
		dev_input = input;
	}

	// Update weights
	for (int32_t i = 0; i < network->num_layers; i++) {
		cn_layer tmp_layer = network->layer[i];
		float* tmp_input = (i == 0) ? dev_input : network->layer[i - 1].output;

		uint32_t num_threads = tmp_layer.num_nodes;
		const uint32_t block_size = 32;
		uint32_t num_blocks = num_threads / block_size;
		if (num_blocks * block_size < num_threads) num_blocks++;
		cn_device_update_weights<<<num_blocks, block_size>>>(tmp_layer, tmp_input, 1.f);
	}

	// Unload input buffer
	if (data_location == cn_host) {
		cudaFree(dev_input);
	}

}

void cn_backprop(cn_network* network, float* input, float* target, cn_location data_location) {
	switch (network->location) {
	case cn_host:
		cn_host_backprop(network, input, target);
		break;
	case cn_device:
		cn_device_backprop(network, input, target, data_location);
		break;
	}
}
