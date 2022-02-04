#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>
#include <time.h>


float kernel_max(float* data, int n) {
    float maxim = data[0]; 
    for (int i = 1; i < n; i++) {
        if (maxim < data[i]) {
            maxim = data[i];
        }
    }
    return maxim;
}

float kernel_min(float* data,int n) {
    float minim = data[0]; 
    for (int i = 1; i < n; i++) {
        if (minim > data[i]) {
            minim = data[i];
        }
    }
    return minim;
}

void kernel_hist(float* data, int* hist, int n, float min, float max, int n_split) {

    for (int i = 0; i < n; i++) {
        hist[(int)((data[i]-min)/(max-min)*n_split*0.99999)]++;
    }
}

void kernel_scan(int* scan, int n_split) {
    for (int i = 1; i < n_split; i++) {
        scan[i] += scan[i-1];
    }
}

void kernel_pre_sort(float* data, int* count, float* out,int n, float min, float max, int n_split) {
    for (int i = 0; i < n; i++) {
        out[count[(int)((data[i]-min)/(max-min)*n_split*0.99999)]--] = data[i];
    }
}

void kerel_bitonic_sort(float* data, int n, int begin) {
    float sdata[1024];
    for (int i = 0; i < 1024; i++) {
        if (i < n) {
            sdata[i] = data[i+begin];
        } else {
            sdata[i] = FLT_MAX;
        }
    }

    for (int k = 2; k <= 1024; k*=2) {
        for (int j = k/2; j > 0; j/=2) {
            for (int tid = 0; tid < 1024; tid++){
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
            }
        }
    }

    for (int i = 0; i < n; i++) {
        data[i+begin] = sdata[i];
    }

}


void BucketSort(float* data, int n, std::vector<int> &bucket_begins) {
    
    float max = kernel_max(data, n);
    float min = kernel_min(data, n);


    if (max == min){
        bucket_begins.push_back(bucket_begins.back()+n);
        return;
    }

    int split_size = 16;
    int n_split = (n - 1) / split_size + 1;
    int *count_data = (int *)malloc(sizeof(int) * n_split);
    std::fill(count_data, count_data+n_split, 0);


    kernel_hist(data, count_data, n, min, max, n_split);

    kernel_scan(count_data, n_split);

    int* dev_count_data = (int *)malloc(sizeof(int)*n_split);
    float* dev_presort_data = (float *)malloc(sizeof(float)*n);
    std::copy(count_data, count_data+n_split, dev_count_data);


    kernel_pre_sort(data, dev_count_data, dev_presort_data, n, min, max, n_split);
    std::copy(dev_presort_data, dev_presort_data+n, data);


    int pocet_size = 1024;
    int begin = 0;
    int step = bucket_begins.back();
    int cur_pocket_end = 0;
    for (int i = 0; i < n_split; i++) {
        if (count_data[i] - begin <= pocet_size) {
            cur_pocket_end = count_data[i];
        } else {
            if (cur_pocket_end == 0) {
                BucketSort(data + begin, count_data[i]-begin, bucket_begins);
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
    free(dev_count_data);
    free(dev_presort_data);
    return;
}

int main() {
    int n;
    scanf("%d", &n);//fread(&n, sizeof(int), 1, stdin);
    float *data = (float *)malloc(sizeof(float) * n);
    for (int i = 0; i < n; i++) {
        scanf("%f", data+i);
    }//fread(data, sizeof(float), n, stdin);
 

    std::vector<int> bucket_begins(1,0); 
    clock_t beginc = clock();
    BucketSort(data, n, bucket_begins);
    //for (int i = 0; i < bucket_begins.size(); i++) {
       // printf("%d\n", bucket_begins[i]);
    //}
    for (int i = 0; i < bucket_begins.size()-1; i++) {
        kerel_bitonic_sort(data, bucket_begins[i+1]-bucket_begins[i], bucket_begins[i]);
    }

    clock_t end = clock();
    printf("%lf",(double)(end-beginc)/ CLOCKS_PER_SEC*1000);

//fwrite(data, sizeof(float), n, stdout);

    free(data);
    return 0;
}