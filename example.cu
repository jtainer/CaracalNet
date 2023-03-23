// 
// Writing test cases here. Don't expect anything written here to work or make sense.
// 
// 2023, Jonathan Tainer
// 

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "caracalnet.h"

void print_weights(cn_network network) {
	cn_network net = cn_network_copy(&network, cn_host);

	for (int l = 0; l < net.num_layers; l++) {
		printf("Layer %d: ", l);
		for (int i = 0; i < net.layer[l].num_nodes * net.layer[l].num_inputs; i++) {
			printf("%f, ", net.layer[l].weight[i]);
		}
		printf("\n");
	}

	cn_network_destroy(&net);
}

int main() {
	srand(time(NULL));

	const int num_inputs = 1;
	const int num_outputs = 2;
	const int num_layers = 2;
	const int num_nodes = 2;
	cn_network net = cn_network_create(num_inputs, num_outputs, num_layers, num_nodes, cn_device);

	float input_vec[num_inputs] = { 2.f };
	float output_vec[num_outputs] = { 0.f, 0.f };
	float target_vec[num_outputs] = { 0.7f, 0.2f };

	for (int i = 0; i < 50; i++) {
		cn_inference(&net, input_vec, output_vec, cn_host);
		cn_backprop(&net, input_vec, target_vec, cn_host);
		printf("output = %f\t%f\n", output_vec[0], output_vec[1]);
		print_weights(net);
	}

	cn_network_destroy(&net);
	return 0;
}
