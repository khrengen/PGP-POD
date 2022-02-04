#include <stdio.h>
#include <stdlib.h>

__global__ void kernel(double* arr, int n) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;			// Абсолютный номер потока
	int offset = blockDim.x * gridDim.x;						// Общее кол-во потоков
	for (int i = idx; i < n; i += offset) {
		arr[i] *= arr[i];
	}
}

int main() {
	int n = 0;
	scanf("%d", &n);
	double* arr = (double*)malloc(sizeof(double) * n);
	for (int i = 0; i < n; i++) {
		scanf("%lf", &arr[i]);
	}

	double* dev_arr;
	cudaMalloc(&dev_arr, sizeof(double) * n);
	cudaMemcpy(dev_arr, arr, sizeof(double) * n, cudaMemcpyHostToDevice);

	kernel<<<256,256>>>(dev_arr, n);

	cudaMemcpy(arr, dev_arr, sizeof(double) * n, cudaMemcpyDeviceToHost);
	cudaFree(dev_arr);
	for (int i = 0; i < n; i++) {
		printf("%.10e ", arr[i]);
	}
	printf("\n");
	free(arr);
	return 0;
}