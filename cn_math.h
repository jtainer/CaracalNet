// 
// CaracalNet math functions used by both host and device
// 
// 2023, Jonathan Tainer
// 

#ifndef CARACALNET_MATH_H
#define CARACALNET_MATH_H

// Various activation functions and their derivatives
// TODO: Currently only have sigmoid, need to implement the other ones
__host__ __device__
float cn_sigmoid(float x);

__host__ __device__
float cn_sigmoid_prime(float x);

// Helper functions for forward propagation
__host__ __device__
float cn_vector_dot_product(float* a, float* b, unsigned int n);

#endif
