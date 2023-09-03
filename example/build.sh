nvcc --relocatable-device-code=true -L/usr/local/cuda/lib64 example.cu ../caracalnet.cu ../cn_core.cu ../cn_math.cu ../cn_util.cu -lcudart -lm
