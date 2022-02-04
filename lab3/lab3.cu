#include <stdio.h>
#include <stdlib.h>
#include <float.h>


const int MAX_AVG_SIZE = 32;
__constant__ float3 dev_avg[MAX_AVG_SIZE];

#define CSC(call)                                                   \
do {                                                                \
    cudaError_t res = call;                                         \
    if (res != cudaSuccess) {                                       \
        fprintf(stderr, "ERROR in %s:%d. Message: %s\n",            \
                __FILE__, __LINE__, cudaGetErrorString(res));       \
        exit(0);                                                    \
    }                                                               \
} while(0)


struct coord{
    int x;
    int y;
};

void avg(float3* out, int* np, coord** coords, uchar4* data, int w, int nc) {
    for (int i = 0; i < nc; i++) {
        float3 ch;
        ch.x = 0;
        ch.y = 0;
        ch.z = 0;
        for (int j = 0; j < np[i]; j++) {
            ch.x += data[coords[i][j].x + coords[i][j].y*w].x;
            ch.y += data[coords[i][j].x + coords[i][j].y*w].y;
            ch.z += data[coords[i][j].x + coords[i][j].y*w].z;
        }

        ch.x /= np[i];
        ch.y /= np[i];
        ch.z /= np[i];
        out[i] = ch;
    }
}

__global__ void kernel(uchar4 *data, int w, int h, int nc) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offsetx = blockDim.x * gridDim.x;
    for (int i = idx; i < w*h; i += offsetx) {
        float max = -FLT_MAX;
        for (int j = 0; j < nc; j++) {
            float3 dif;
            dif.x = data[i].x - dev_avg[j].x;
            dif.y = data[i].y - dev_avg[j].y;
            dif.z = data[i].z - dev_avg[j].z;     
            float argj = -dif.x*dif.x - dif.y*dif.y - dif.z*dif.z;

            if (argj > max) {
                max = argj;
                data[i].w = j;
            }
        }
    }
}

int main() {
    int w, h;
    char strIn[50];
    char strOut[50];
    int nc;

    scanf("%s\n%s", strIn, strOut);
    scanf("%d\n", &nc);

    int* np = (int*)malloc(sizeof(int) * nc);
    coord** coords = (coord**)malloc(sizeof(coord*) * nc);
    for (int i = 0; i < nc; i++) {
        scanf("%d", &np[i]);
        coords[i] = (coord*)malloc(sizeof(coord) * np[i]);
        for (int j = 0; j < np[i]; j++) {
            int x, y;
            scanf("%d%d", &x,&y);
            coord crd;
            crd.x = x;
            crd.y = y;
            coords[i][j] = crd;
        }
    }

    FILE *fp = fopen(strIn, "rb");
    fread(&w, sizeof(int), 1, fp);
    fread(&h, sizeof(int), 1, fp);
    uchar4 *data = (uchar4 *)malloc(sizeof(uchar4) * w * h);
    fread(data, sizeof(uchar4), w * h, fp);
    fclose(fp);
    
    float3 h_avg[MAX_AVG_SIZE];
    avg(h_avg, np, coords, data, w, nc);
    uchar4 *dev_data;
    CSC(cudaMalloc(&dev_data, sizeof(uchar4) * w * h));
    CSC(cudaMemcpy(dev_data, data, sizeof(uchar4) * w * h, cudaMemcpyHostToDevice));

    CSC(cudaMemcpyToSymbol(dev_avg, h_avg, sizeof(float3) * MAX_AVG_SIZE));

    kernel<<<1024, 1024>>>(dev_data, w, h, nc);
    CSC(cudaGetLastError());

    CSC(cudaMemcpy(data, dev_data, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));

    free(np);
    for (int i = 0; i < nc; i++) {
        free(coords[i]);
    }
    free(coords);
    CSC(cudaFree(dev_data));

    fp = fopen(strOut, "wb");
    fwrite(&w, sizeof(int), 1, fp);
    fwrite(&h, sizeof(int), 1, fp);
    fwrite(data, sizeof(uchar4), w * h, fp);
    fclose(fp);

    free(data);
    return 0;
}