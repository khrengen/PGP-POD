#include <stdio.h>
#include <stdlib.h>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>


#define CSC(call)                                                   \
do {                                                                \
    cudaError_t res = call;                                         \
    if (res != cudaSuccess) {                                       \
        fprintf(stderr, "ERROR in %s:%d. Message: %s\n",            \
                __FILE__, __LINE__, cudaGetErrorString(res));       \
        exit(0);                                                    \
    }                                                               \
} while(0)


struct comparator {
    __host__ __device__ bool operator()(double a, double b) {
        return fabs(a) < fabs(b);
    }
};

__global__ void kernel_gaussian(double* a, int w, int h, int i, int j) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    for (int k = idy + j + 1; k < h; k += offsety) {
        for (int q = idx + i + 1; q < w; q+= offsetx) {
            a[k*w + q] -= a[k*w + i] * a[j*w + q]/a[j*w+i];
        }
    }
}

__global__ void kernel_swap(double* dev_data, int w, int h, int j, int i, int ind) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offsetx = blockDim.x * gridDim.x;
    for (int k = idx+j; k < h; k+= offsetx) {
        double tmp = dev_data[k*w+i];
        dev_data[k*w+i] = dev_data[k*w+ind];
        dev_data[k*w+ind] = tmp; 
    }
}

int main() {
    comparator comp;
    int w, h;
    scanf("%d%d", &w,&h);
            
    double *data = (double *)malloc(sizeof(double) * w * h);

    for (int i = 0; i < w; i++) {
            for (int j = 0; j < h; j++) {
                scanf("%lf", &data[j*w + i]);
            }
    }

    double *dev_data;
    CSC(cudaMalloc(&dev_data, sizeof(double) * w * h));
    CSC(cudaMemcpy(dev_data, data, sizeof(double) * w * h, cudaMemcpyHostToDevice));
 
    int i = 0;
    for (int j = 0; j < h; j++) {
        thrust::device_ptr<double> col_beg = thrust::device_pointer_cast(dev_data + j*w);
        thrust::device_ptr<double> max_ptr = thrust::max_element(col_beg+i, col_beg+w, comp);
        if (fabs(*max_ptr) > 1e-7) {
            if (i == w-1) { // обработка последней строки 
                i++;
                break;
            }

            int ind = (int)(max_ptr-col_beg);
            if (ind != i) {
                kernel_swap<<<16,32>>>(dev_data, w, h, j, i, ind);
                CSC(cudaGetLastError());
            }

            kernel_gaussian<<<dim3(16, 32), dim3(16, 32)>>>(dev_data, w, h, i, j);
            CSC(cudaGetLastError());
            i++;
        }
    }
    
    printf("%d\n", i);

    CSC(cudaFree(dev_data));
    free(data);
    return 0;
}