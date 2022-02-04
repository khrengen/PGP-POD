#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void kernel(double* arr, int n) {
    for (int i = 0; i < n; i++) {
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
    clock_t begin = clock();
    kernel(arr, n);
    clock_t end = clock();
    printf("%f",(double)(end - begin) / CLOCKS_PER_SEC*1000000);
    //for (int i = 0; i < n; i++) {
    //  printf("%.10e ", arr[i]);
    //}
    printf("\n");
    free(arr);
    return 0;
}