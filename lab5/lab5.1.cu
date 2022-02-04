#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/scan.h>
#include <vector>

#define BLOCK_SIZE 1024
#define GRID_SIZE 1024

#define CSC(call)                                                   \
do {                                                                \
    cudaError_t res = call;                                         \
    if (res != cudaSuccess) {                                       \
        fprintf(stderr, "ERROR in %s:%d. Message: %s\n",            \
                __FILE__, __LINE__, cudaGetErrorString(res));       \
        exit(0);                                                    \
    }                                                               \
} while(0)


__global__ void kernel_reduce_max(float* data, int n, float* out) {
  volatile __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int offsetx = blockDim.x * gridDim.x;
    for (int i = idx; i < (n/BLOCK_SIZE+1)*BLOCK_SIZE; i+=offsetx) {
        sdata[tid] = -FLT_MAX;
        if (i < n) {
            sdata[tid] = data[i];
        }
        __syncthreads();

        for (int s = blockDim.x/2; s > 32; s>>=1) {
            if (tid < s) {
                sdata[tid] = max(sdata[tid],sdata[tid+s]);
            }
            __syncthreads();
        }

        if (tid < 32) {
            sdata[tid] = max(sdata[tid],sdata[tid+32]);
            sdata[tid] = max(sdata[tid],sdata[tid+16]);
            sdata[tid] = max(sdata[tid],sdata[tid+8]);
            sdata[tid] = max(sdata[tid],sdata[tid+4]);
            sdata[tid] = max(sdata[tid],sdata[tid+2]);
            sdata[tid] = max(sdata[tid],sdata[tid+1]);
        }

        if (tid == 0) {
            out[blockIdx.x] = max(sdata[0], out[blockIdx.x]);
        }
    }
}

__global__ void kernel_reduce_min(float* data,int n, float* out) {
    volatile __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int offsetx = blockDim.x * gridDim.x;
    for (int i = idx; i < (n/BLOCK_SIZE+1)*BLOCK_SIZE; i+=offsetx) {
        sdata[tid] = FLT_MAX;
        if (i < n) {
            sdata[tid] = data[i];
        }
        __syncthreads();

        for (int s = blockDim.x/2; s > 32; s>>=1) {
            if (tid < s) {
                sdata[tid] = min(sdata[tid],sdata[tid+s]);
            }
            __syncthreads();
        }

        if (tid < 32) {
            sdata[tid] = min(sdata[tid],sdata[tid+32]);
            sdata[tid] = min(sdata[tid],sdata[tid+16]);
            sdata[tid] = min(sdata[tid],sdata[tid+8]);
            sdata[tid] = min(sdata[tid],sdata[tid+4]);
            sdata[tid] = min(sdata[tid],sdata[tid+2]);
            sdata[tid] = min(sdata[tid],sdata[tid+1]);
        }

        if (tid == 0) {
            out[blockIdx.x] = min(sdata[0], out[blockIdx.x]);
        }
    }
}

__global__ void kernel_hist(float* data, int* hist, int n, float min, float max, int n_split) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offsetx = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i+=offsetx) {
        atomicAdd(hist + (int)((data[i]-min)/(max-min)*n_split*0.99999), 1);
    }
}

__global__ void kernel_pre_sort(float* data, int* count, float* out,int n, float min, float max, int n_split) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offsetx = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i+=offsetx) {
        out[atomicAdd(count + (int)((data[i]-min)/(max-min)*n_split*0.99999), -1)-1] = data[i];
    }
}

__global__ void kerel_bitonic_sort(float* data, int n, int* begin_arr, int pocet_num) {
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offsetx = gridDim.x * blockDim.x;
    int grid = 0;

    for (int i = idx; i < pocet_num*BLOCK_SIZE; i+=offsetx, grid+=gridDim.x) {
        int begin = begin_arr[grid+blockIdx.x];
        if (begin_arr[grid+blockIdx.x+1]-begin > BLOCK_SIZE) {
            continue;
        }
        sdata[tid] = FLT_MAX;
        if (tid < begin_arr[grid+blockIdx.x+1]-begin){ 
            sdata[tid] = data[tid+begin];
        }    
        __syncthreads();
    
        for (int k = 2; k <= BLOCK_SIZE; k*=2) {
            for (int j = k/2; j > 0; j/=2) {
                int ixj = tid ^ j;
                if (ixj > tid) {
                    if ((tid & k) == 0) {
                        if (sdata[tid] > sdata[ixj]) {
                            float tmp = sdata[tid];
                            sdata[tid] = sdata[ixj];
                            sdata[ixj] = tmp;
                        }
                    } else {
                        if (sdata[tid] < sdata[ixj]) {
                            float tmp = sdata[tid];
                            sdata[tid] = sdata[ixj];
                            sdata[ixj] = tmp;
                        }
                    }
                }
                __syncthreads();
            }
        }
        if (tid < begin_arr[grid+blockIdx.x+1]-begin){
            data[tid+begin] = sdata[tid];
        }
    }
}

void BucketSort(float* dev_data, int n, std::vector<int> &bucket_begins) {
    float* max_arr = (float*)malloc(sizeof(float)*GRID_SIZE);
    float* min_arr = (float*)malloc(sizeof(float)*GRID_SIZE);

    std::fill(max_arr, max_arr+GRID_SIZE, -FLT_MAX);
    std::fill(min_arr, min_arr+GRID_SIZE, FLT_MAX);

    float* dev_max_arr;
    float* dev_min_arr;
    CSC(cudaMalloc(&dev_max_arr, sizeof(float)*GRID_SIZE));
    CSC(cudaMalloc(&dev_min_arr, sizeof(float)*GRID_SIZE));
    CSC(cudaMemcpy(dev_max_arr, max_arr, sizeof(float) * GRID_SIZE, cudaMemcpyHostToDevice));
    CSC(cudaMemcpy(dev_min_arr, min_arr, sizeof(float) * GRID_SIZE, cudaMemcpyHostToDevice));

    kernel_reduce_max<<<GRID_SIZE, BLOCK_SIZE>>>(dev_data, n, dev_max_arr);
    CSC(cudaGetLastError());
    kernel_reduce_min<<<GRID_SIZE, BLOCK_SIZE>>>(dev_data, n, dev_min_arr);
    CSC(cudaGetLastError());

    CSC(cudaMemcpy(max_arr, dev_max_arr, sizeof(float) * GRID_SIZE, cudaMemcpyDeviceToHost));
    CSC(cudaMemcpy(min_arr, dev_min_arr, sizeof(float) * GRID_SIZE, cudaMemcpyDeviceToHost));
    
    for (int i = 1; i < GRID_SIZE; i++) {
        max_arr[0] = max(max_arr[0], max_arr[i]);
        min_arr[0] = min(min_arr[0], min_arr[i]);
    }
    float max = max_arr[0];
    float min = min_arr[0];

    free(max_arr);
    free(min_arr);
    CSC(cudaFree(dev_max_arr));
    CSC(cudaFree(dev_min_arr));

    if (max == min){
        bucket_begins.push_back(bucket_begins.back()+n);

        return;
    }

    int split_size = 16;
    int n_split = (n - 1) / split_size + 1;
    int *count_data = (int *)malloc(sizeof(int) * n_split);
    int* dev_count_data;
    CSC(cudaMalloc(&dev_count_data, sizeof(int) * n_split));
    CSC(cudaMemset(dev_count_data, 0, sizeof(int) * n_split));

    kernel_hist<<<GRID_SIZE, BLOCK_SIZE>>>(dev_data, dev_count_data, n, min, max, n_split);
    CSC(cudaGetLastError());
    thrust::device_ptr<int> ptr = thrust::device_pointer_cast(dev_count_data);
    thrust::inclusive_scan(ptr, ptr + n_split, ptr);
    CSC(cudaMemcpy(count_data, dev_count_data, sizeof(int) * n_split, cudaMemcpyDeviceToHost));
    float* dev_presort_data;
    CSC(cudaMalloc(&dev_presort_data, sizeof(float) * n));
    kernel_pre_sort<<<GRID_SIZE,BLOCK_SIZE>>>(dev_data, dev_count_data, dev_presort_data, n, min, max, n_split);
    CSC(cudaGetLastError());
    CSC(cudaMemcpy(dev_data, dev_presort_data, sizeof(float) * n, cudaMemcpyDeviceToDevice));

    int pocet_size = BLOCK_SIZE;
    int begin = 0;
    int step = bucket_begins.back();
    int cur_pocket_end = 0;
    for (int i = 0; i < n_split; i++) {
        if (count_data[i] - begin <= pocet_size) {
            cur_pocket_end = count_data[i];
        } else {
            if (cur_pocket_end == 0) {
                BucketSort(dev_data + begin, count_data[i]-begin, bucket_begins);
                begin = count_data[i];
            } else {
                bucket_begins.push_back(cur_pocket_end + step);
                begin = cur_pocket_end;
                cur_pocket_end = 0;
                i--;
            }
        }
    }
    if (cur_pocket_end != 0) {
        bucket_begins.push_back(step+n);
    }

    free(count_data);
    CSC(cudaFree(dev_count_data));
    CSC(cudaFree(dev_presort_data));
    return;
}

int main() {
    int n;
    fread(&n, sizeof(int), 1, stdin);
    float *data = (float *)malloc(sizeof(float) * n);
    fread(data, sizeof(float), n, stdin);
    float *dev_data;
    CSC(cudaMalloc(&dev_data, sizeof(float) * n));
    CSC(cudaMemcpy(dev_data, data, sizeof(float) * n, cudaMemcpyHostToDevice));

    std::vector<int> bucket_begins(1,0); 
    BucketSort(dev_data, n, bucket_begins);

    int *dev_bucket_begins;
    CSC(cudaMalloc(&dev_bucket_begins, sizeof(int) * bucket_begins.size()));
    CSC(cudaMemcpy(dev_bucket_begins, &bucket_begins[0], sizeof(int) * bucket_begins.size(), cudaMemcpyHostToDevice));
    kerel_bitonic_sort<<<GRID_SIZE, BLOCK_SIZE>>>(dev_data, n, dev_bucket_begins, bucket_begins.size()-1);
    CSC(cudaGetLastError());

    CSC(cudaMemcpy(data, dev_data, sizeof(float) * n, cudaMemcpyDeviceToHost));

    fwrite(data, sizeof(float), n, stdout);

    CSC(cudaFree(dev_data));
    CSC(cudaFree(dev_bucket_begins));
    free(data);
    return 0;
}