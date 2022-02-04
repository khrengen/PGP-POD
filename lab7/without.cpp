#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <math.h>
#include <time.h>

#define _i(i, j, k) (((k) + 1) * (nx + 2) * (ny + 2) + ((j) + 1) * (nx + 2) + (i) + 1)
#define _ib(i, j, k) ((k) * nbx * nby + (j) * nbx + (i))

int main(int argc, char* argv[]) {
	int ib, jb, kb, nbx, nby, nbz, nx, ny, nz;
	int id, numproc;
	double lx, ly, lz, hx, hy, hz, bc_down, bc_up, bc_left, bc_right, bc_front, bc_back, u0;
	double eps, cur_eps;
	double *data, *temp, *next, *buff;
	char fname[100];

	
	scanf("%d %d %d", &nbx, &nby, &nbz);
	scanf("%d %d %d", &nx, &ny, &nz);
	scanf("%s", fname);
	scanf("%lf", &eps);
	scanf("%lf %lf %lf", &lx, &ly, &lz);
	scanf("%lf %lf %lf %lf %lf %lf", &bc_down, &bc_up, &bc_left, &bc_right, &bc_front, &bc_back);
	scanf("%lf", &u0);

	clock_t startTime, endTime;
	startTime = clock();

	kb = id / (nbx * nby);
	jb = id % (nbx * nby) / nbx;
	ib = id % (nbx * nby) % nbx;

	hx = lx / (nx * nbx);
	hy = ly / (ny * nby);
	hz = lz / (nz * nbz);

	data = (double*)malloc(sizeof(double) * (nx + 2) * (ny + 2) * (nz + 2));
	next = (double*)malloc(sizeof(double) * (nx + 2) * (ny + 2) * (nz + 2));

	for (int i = 0; i < nx; i++) {					// Инициализация блока
		for (int j = 0; j < ny; j++) {
			for (int k = 0; k < nz; k++) {
				data[_i(i, j, k)] = u0;
			}
		}
	}

	for (int j = 0; j < ny; j++) {
		for (int k = 0; k < nz; k++) {
			data[_i(-1, j, k)] = bc_left;
			next[_i(-1, j, k)] = bc_left;
		}
	}

	for (int i = 0; i < nx; i++) {
		for (int k = 0; k < nz; k++) {
			data[_i(i, -1, k)] = bc_front;
			next[_i(i, -1, k)] = bc_front;
		}
	}

	for (int i = 0; i < nx; i++) {
		for (int j = 0; j < ny; j++) {
			data[_i(i, j, -1)] = bc_down;
			next[_i(i, j, -1)] = bc_down;
		}
	}

	for (int j = 0; j < ny; j++) {
		for (int k = 0; k < nz; k++) {
			data[_i(nx, j, k)] = bc_right;
			next[_i(nx, j, k)] = bc_right;
		}
	}

	for (int i = 0; i < nx; i++) {
		for (int k = 0; k < nz; k++) {
			data[_i(i, ny, k)] = bc_back;
			next[_i(i, ny, k)] = bc_back;
		}
	}

	for (int i = 0; i < nx; i++) {
		for (int j = 0; j < ny; j++) {
			data[_i(i, j, nz)] = bc_up;
			next[_i(i, j, nz)] = bc_up;
		}
	}

	cur_eps = eps + 1;
	while (cur_eps >= eps) {

		cur_eps = 0.0;
		for (int i = 0; i < nx; i++) {
			for (int j = 0; j < ny; j++) {
				for (int k = 0; k < nz; k++) {
					next[_i(i, j, k)] = 0.5 * ((data[_i(i + 1, j, k)] + data[_i(i - 1, j, k)]) / (hx * hx) +
						(data[_i(i, j + 1, k)] + data[_i(i, j - 1, k)]) / (hy * hy) +
						(data[_i(i, j, k + 1)] + data[_i(i, j, k - 1)]) / (hz * hz)) /
						(1.0 / (hx * hx) + 1.0 / (hy * hy) + 1.0 / (hz * hz));

					cur_eps = fmax(cur_eps, fabs(next[_i(i, j, k)] - data[_i(i, j, k)]));
				}
			}
		}

		temp = next;
		next = data;
		data = temp;

	}

	endTime = clock();
	printf("%f ms \n", ((double)endTime - startTime)/CLOCKS_PER_SEC*1000);

	FILE* f = fopen(fname, "w");
		for (int k = 0; k < nz; k++) {
				for (int j = 0; j < ny; j++) {
						for (int i = 0; i < nx; i++) {
							fprintf(f, "%.6e ", data[_i(i, j, k)]);
						
						}
					
				}
			
		}
		fclose(f);

	free(data);
	free(next);
	return 0;
}
