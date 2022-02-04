#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>



void gaussian(double* a, int w, int h, int i, int j) {
    for (int k = j + 1; k < h; k++) {
        for (int q = i + 1; q < w; q++) {
            a[k*w + q] -= a[k*w + i] * a[j*w + q]/a[j*w+i];
        }
    }
}

void swap(double* dev_data, int w, int h, int j, int i, int ind) {
    for (int k = j; k < h; k++) {
        double tmp = dev_data[k*w+i];
        dev_data[k*w+i] = dev_data[k*w+ind];
        dev_data[k*w+ind] = tmp; 
    }
}

double* max_element(int i, int w, double* col) {
    double max = -100000000; 
    double* answ;
    for (double* q = col + i; q < col+w; q++) {
        if (max < *q) {
            max = *q;
            answ = q;
        }
    }
    return answ;
}

int main() {
    int w, h;
    scanf("%d%d", &w,&h);
            
    double *data = (double *)malloc(sizeof(double) * w * h);

    for (int i = 0; i < w; i++) {
            for (int j = 0; j < h; j++) {
                scanf("%lf", &data[j*w + i]);
            }
    }

 
    int i = 0;
    double time = 0;
    for (int j = 0; j < h; j++) {
        double* col_beg = data + j*w;
        double* max_ptr = max_element(i, w, col_beg);
        if (fabs(*max_ptr) > 1e-7) {
            if (i == w-1) { // обработка последней строки 
                i++;
                break;
            }

            int ind = (int)(max_ptr-col_beg);
            if (ind != i) {
                swap(data, w, h, j, i, ind);
            }
            clock_t begin = clock();
            gaussian(data, w, h, i, j);
            clock_t end = clock();
            time += (double)end-begin;
            i++;
        }
    }
    printf("%lf",time / CLOCKS_PER_SEC*1000);
    printf("%d\n", i);
    free(data);
    return 0;
}