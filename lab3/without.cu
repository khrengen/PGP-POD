#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <time.h>

const int MAX_AVG_SIZE = 32;
__constant__ float3 dev_avg[MAX_AVG_SIZE];

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

void ker(uchar4 *data, float3 *h_avg, int w, int h, int nc) {
    for (int i = 0; i < w*h; i ++) {
        float max = -FLT_MAX;
        for (int j = 0; j < nc; j++) {
            float3 dif;
            dif.x = data[i].x - h_avg[j].x;
            dif.y = data[i].y - h_avg[j].y;
            dif.z = data[i].z - h_avg[j].z;     
            float argj = -dif.x*dif.x - dif.y*dif.y - dif.z*dif.z;

            if (argj > max) {
                max = argj;
                data[i].w = j;
            }
        }
    }
}

__global__ void kernel(uchar4 *data, float3 *dev_avg, int w, int h, int nc) {
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
    printf("%d, %d\n", w, h);
    uchar4 *data = (uchar4 *)malloc(sizeof(uchar4) * w * h);
    fread(data, sizeof(uchar4), w * h, fp);
    fclose(fp);
    
    float3 h_avg[MAX_AVG_SIZE];
    avg(h_avg, np, coords, data, w, nc);
    
    clock_t begin = clock();
    ker(data, h_avg, w, h, nc);
    clock_t end = clock();
    printf("%f",(double)(end - begin) / CLOCKS_PER_SEC*1000000);

    for (int i = 0; i < w*h; i++) {
        if(data[i].w == 0) {
            data[i].x = 200;
            data[i].y = 0;
            data[i].z = 0;
        } else if (data[i].w == 1) {
            data[i].x = 0;
            data[i].y = 200;
            data[i].z = 0;
        } else if(data[i].w == 2) {
            data[i].x = 0;
            data[i].y = 0;
            data[i].z = 200;
        } else if(data[i].w == 3) {
            data[i].x = 200;
            data[i].y = 200;
            data[i].z = 0;
        } else {
            data[i].x = 0;
            data[i].y = 200;
            data[i].z = 200;
        }
    }
    free(np);
    for (int i = 0; i < nc; i++) {
        free(coords[i]);
    }
    free(coords);

    fp = fopen(strOut, "wb");
    fwrite(&w, sizeof(int), 1, fp);
    fwrite(&h, sizeof(int), 1, fp);
    fwrite(data, sizeof(uchar4), w * h, fp);
    fclose(fp);

    free(data);
    printf("ok\n");
    return 0;
}