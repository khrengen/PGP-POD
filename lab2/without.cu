#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CSC(call)                                                   \
do {                                                                \
    cudaError_t res = call;                                         \
    if (res != cudaSuccess) {                                       \
        fprintf(stderr, "ERROR in %s:%d. Message: %s\n",            \
                __FILE__, __LINE__, cudaGetErrorString(res));       \
        exit(0);                                                    \
    }                                                               \
} while(0)

double intensity2(uchar4 p) {
    return 0.299 * p.x + 0.587 * p.y + 0.114 * p.z;
}
void kernel2(uchar4 *out, uchar4 *in, int w, int h) {

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            double w11 = intensity2(in[max(min(y-1,h-1),0)*w + max(min(x-1,w-1),0)]);
            double w12 = intensity2(in[max(min(y,h-1),0)*w + max(min(x-1,w-1),0)]);
            double w13 = intensity2(in[max(min(y+1,h-1),0)*w + max(min(x-1,w-1),0)]);
            double w21 = intensity2(in[max(min(y-1,h-1),0)*w + max(min(x,w-1),0)]);
            double w22 = intensity2(in[max(min(y,h-1),0)*w + max(min(x,w-1),0)]);
            double w23 = intensity2(in[max(min(y+1,h-1),0)*w + max(min(x,w-1),0)]);
            double w31 = intensity2(in[max(min(y-1,h-1),0)*w + max(min(x+1,w-1),0)]);
            double w32 = intensity2(in[max(min(y,h-1),0)*w + max(min(x+1,w-1),0)]);
            double w33 = intensity2(in[max(min(y+1,h-1),0)*w + max(min(x+1,w-1),0)]);

            double gx = w13 + 2*w23 + w33 - w11 - 2*w21 - w31;
            double gy = w31 + 2*w32 + w33 - w11 - 2*w12 - w13;

            int grad = (int)sqrt(gx*gx + gy*gy);
            int result = min(255, grad);

            out[y * w + x] = make_uchar4(result, result, result, 0);
        }
    }
}

// текстурная ссылка <тип элементов, размерность, режим нормализации>
texture<uchar4, 2, cudaReadModeElementType> tex;

__device__ double intensity(uchar4 p) {
  return 0.299 * p.x + 0.587 * p.y + 0.114 * p.z;
}

__global__ void kernel(uchar4 *out, int w, int h) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    for (int y = idy; y < h; y += offsety) {
        for (int x = idx; x < w; x += offsetx) {
            double w11 = intensity(tex2D(tex, x-1, y-1));
            double w12 = intensity(tex2D(tex, x-1, y));
            double w13 = intensity(tex2D(tex, x-1, y+1));
            double w21 = intensity(tex2D(tex, x, y-1));
            double w22 = intensity(tex2D(tex, x, y));
            double w23 = intensity(tex2D(tex, x, y+1));
            double w31 = intensity(tex2D(tex, x+1, y-1));
            double w32 = intensity(tex2D(tex, x+1, y));
            double w33 = intensity(tex2D(tex, x+1, y+1));

            double gx = w13 + 2*w23 + w33 - w11 - 2*w21 - w31;
            double gy = w31 + 2*w32 + w33 - w11 - 2*w12 - w13;

            int grad = (int)sqrt(gx*gx + gy*gy);
            int result = min(255, grad);

            out[y * w + x] = make_uchar4(result, result, result, 0);
        }
    }
}

int main() {
    int w, h;
    char strIn[50];
    char strOut[50];
    scanf("%s\n%s", strIn, strOut);
    FILE *fp = fopen(strIn, "rb");
    fread(&w, sizeof(int), 1, fp);
    fread(&h, sizeof(int), 1, fp);
    uchar4 *data = (uchar4 *)malloc(sizeof(uchar4) * w * h);
    uchar4 *out = (uchar4 *)malloc(sizeof(uchar4) * w * h);
    fread(data, sizeof(uchar4), w * h, fp);
    fclose(fp);

    clock_t begin = clock();
    kernel2(out, data, w,h);
    clock_t end = clock();
    printf("%f",(double)(end - begin) / CLOCKS_PER_SEC*1000000);
    // Подготовка данных для текстуры
    cudaArray *arr;
    cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
    CSC(cudaMallocArray(&arr, &ch, w, h));

    CSC(cudaMemcpyToArray(arr, 0, 0, data, sizeof(uchar4) * w * h, cudaMemcpyHostToDevice));

    // Подготовка текстурной ссылки, настройка интерфейса работы с данными
    tex.addressMode[0] = cudaAddressModeClamp;  // Политика обработки выхода за границы по каждому измерению
    tex.addressMode[1] = cudaAddressModeClamp;
    tex.channelDesc = ch;
    tex.filterMode = cudaFilterModePoint;       // Без интерполяции при обращении по дробным координатам
    tex.normalized = false;                     // Режим нормализации координат: без нормализации

    // Связываем интерфейс с данными
    CSC(cudaBindTextureToArray(tex, arr, ch));

    uchar4 *dev_out;
    CSC(cudaMalloc(&dev_out, sizeof(uchar4) * w * h));

    kernel<<<dim3(16, 16), dim3(32, 32)>>>(dev_out, w, h);
    CSC(cudaGetLastError());

    CSC(cudaMemcpy(data, dev_out, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));

    // Отвязываем данные от текстурной ссылки
    CSC(cudaUnbindTexture(tex));


    CSC(cudaFreeArray(arr));
    CSC(cudaFree(dev_out));

    fp = fopen(strOut, "wb");
    fwrite(&w, sizeof(int), 1, fp);
    fwrite(&h, sizeof(int), 1, fp);
    fwrite(out, sizeof(uchar4), w * h, fp);
    fclose(fp);

    free(data);
    free(out);

    return 0;
}