// 
// CaracalNet math functions used by both host and device
// 
// 2023, Jonathan Tainer
// 

#include <cuda.h>
#include <math.h>
#include "cn_math.h"

// Various activation functions and their derivatives
// TODO: Currently only have sigmoid, need to implement the other ones
__host__ __device__
float cn_sigmoid(float x) {
	return 1.f / (1.f + expf(-x));
}

__host__ __device__
float cn_sigmoid_prime(float x) {
	return cn_sigmoid(x) * (1.f - cn_sigmoid(x));
}

// Helper functions for forward propagation
__host__ __device__
float cn_vector_dot_product(float* a, float* b, unsigned int n) {
	float sum = 0.f;
	for (unsigned int i = 0; i < n; i++) {
		sum += a[i] * b[i];
	}
	return sum;
}

