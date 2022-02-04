#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <math.h>
#include "mpi.h"

#define _i(i, j, k) (((k) + 1) * (nx + 2) * (ny + 2) + ((j) + 1) * (nx + 2) + (i) + 1)
#define _ib(i, j, k) ((k) * nbx * nby + (j) * nbx + (i))

int main(int argc, char* argv[]) {
	int ib, jb, kb, nbx, nby, nbz, nx, ny, nz;
	int id, numproc;
	double lx, ly, lz, hx, hy, hz, bc_down, bc_up, bc_left, bc_right, bc_front, bc_back, u0;
	double eps, cur_eps;
	double *data, *temp, *next, *buff;
	char fname[100];

	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);

	MPI_Barrier(MPI_COMM_WORLD);

	if (id == 0) {
		scanf("%d %d %d", &nbx, &nby, &nbz);
		scanf("%d %d %d", &nx, &ny, &nz);
		scanf("%s", fname);
		scanf("%lf", &eps);
		scanf("%lf %lf %lf", &lx, &ly, &lz);
		scanf("%lf %lf %lf %lf %lf %lf", &bc_down, &bc_up, &bc_left, &bc_right, &bc_front, &bc_back);
		scanf("%lf", &u0);
	}

	MPI_Bcast(&nx, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&ny, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&nz, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&nbx, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&nby, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&nbz, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&lx, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&ly, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&lz, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&bc_down, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&bc_up, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&bc_left, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&bc_right, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&bc_front, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&bc_back, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&eps, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&u0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	kb = id / (nbx * nby);
	jb = id % (nbx * nby) / nbx;
	ib = id % (nbx * nby) % nbx;

	hx = lx / (nx * nbx);
	hy = ly / (ny * nby);
	hz = lz / (nz * nbz);

	data = (double*)malloc(sizeof(double) * (nx + 2) * (ny + 2) * (nz + 2));
	next = (double*)malloc(sizeof(double) * (nx + 2) * (ny + 2) * (nz + 2));
	int buf_side1 = std::max(nx, ny);
	int buf_side2 = std::max(ny, nz);

	buff = (double*)malloc(sizeof(double) * buf_side1 * buf_side2);
	int buffer_size;
	MPI_Pack_size(buf_side1 * buf_side2, MPI_DOUBLE, MPI_COMM_WORLD, &buffer_size);
	buffer_size = 6 * (buffer_size + MPI_BSEND_OVERHEAD);
	double* buffer = (double*)malloc(buffer_size);
	MPI_Buffer_attach(buffer, buffer_size);

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
		MPI_Barrier(MPI_COMM_WORLD);

		if (ib + 1 < nbx) {
			for (int j = 0; j < ny; j++) {
				for (int k = 0; k < nz; k++) {
					buff[j * nz + k] = data[_i(nx - 1, j, k)];
				}
			}
			MPI_Bsend(buff, ny * nz, MPI_DOUBLE, _ib(ib + 1, jb, kb), id, MPI_COMM_WORLD);
		}

		if (jb + 1 < nby) {
			for (int i = 0; i < nx; i++) {
				for (int k = 0; k < nz; k++) {
					buff[i * nz + k] = data[_i(i, ny - 1, k)];
				}
			}
			MPI_Bsend(buff, nx * nz, MPI_DOUBLE, _ib(ib, jb + 1, kb), id, MPI_COMM_WORLD);
		}

		if (kb + 1 < nbz) {
			for (int i = 0; i < nx; i++) {
				for (int j = 0; j < ny; j++) {
					buff[i * ny + j] = data[_i(i, j, nz - 1)];
				}
			}
			MPI_Bsend(buff, nx * ny, MPI_DOUBLE, _ib(ib, jb, kb + 1), id, MPI_COMM_WORLD);
		}

		if (ib > 0) {
			for (int j = 0; j < ny; j++) {
				for (int k = 0; k < nz; k++) {
					buff[j * nz + k] = data[_i(0, j, k)];
				}
			}
			MPI_Bsend(buff, ny * nz, MPI_DOUBLE, _ib(ib - 1, jb, kb), id, MPI_COMM_WORLD);
		}

		if (jb > 0) {
			for (int i = 0; i < nx; i++) {
				for (int k = 0; k < nz; k++) {
					buff[i * nz + k] = data[_i(i, 0, k)];
				}
			}
			MPI_Bsend(buff, nx * nz, MPI_DOUBLE, _ib(ib, jb - 1, kb), id, MPI_COMM_WORLD);
		}

		if (kb > 0) {
			for (int i = 0; i < nx; i++) {
				for (int j = 0; j < ny; j++) {
					buff[i * ny + j] = data[_i(i, j, 0)];
				}
			}
			MPI_Bsend(buff, nx * ny, MPI_DOUBLE, _ib(ib, jb, kb - 1), id, MPI_COMM_WORLD);
		}

		/*---------------------------------------------------------------------------------------------------------*/
		if (ib > 0) {
			MPI_Recv(buff, ny * nz, MPI_DOUBLE, _ib(ib - 1, jb, kb), _ib(ib - 1, jb, kb), MPI_COMM_WORLD, &status);
			for (int j = 0; j < ny; j++) {
				for (int k = 0; k < nz; k++) {
					data[_i(-1, j, k)] = buff[j * nz + k];
				}
			}
		}

		if (jb > 0) {
			MPI_Recv(buff, nx * nz, MPI_DOUBLE, _ib(ib, jb - 1, kb), _ib(ib, jb - 1, kb), MPI_COMM_WORLD, &status);
			for (int i = 0; i < nx; i++) {
				for (int k = 0; k < nz; k++) {
					data[_i(i, -1, k)] = buff[i * nz + k];
				}
			}
		}

		if (kb > 0) {
			MPI_Recv(buff, nx * ny, MPI_DOUBLE, _ib(ib, jb, kb - 1), _ib(ib, jb, kb - 1), MPI_COMM_WORLD, &status);
			for (int i = 0; i < nx; i++) {
				for (int j = 0; j < ny; j++) {
					data[_i(i, j, -1)] = buff[i * ny + j];
				}
			}
		}

		if (ib + 1 < nbx) {
			MPI_Recv(buff, ny * nz, MPI_DOUBLE, _ib(ib + 1, jb, kb), _ib(ib + 1, jb, kb), MPI_COMM_WORLD, &status);
			for (int j = 0; j < ny; j++) {
				for (int k = 0; k < nz; k++) {
					data[_i(nx, j, k)] = buff[j * nz + k];
				}
			}
		}

		if (jb + 1 < nby) {
			MPI_Recv(buff, nx * nz, MPI_DOUBLE, _ib(ib, jb + 1, kb), _ib(ib, jb + 1, kb), MPI_COMM_WORLD, &status);
			for (int i = 0; i < nx; i++) {
				for (int k = 0; k < nz; k++) {
					data[_i(i, ny, k)] = buff[i * nz + k];
				}
			}
		}

		if (kb + 1 < nbz) {
			MPI_Recv(buff, nx * ny, MPI_DOUBLE, _ib(ib, jb, kb + 1), _ib(ib, jb, kb + 1), MPI_COMM_WORLD, &status);
			for (int i = 0; i < nx; i++) {
				for (int j = 0; j < ny; j++) {
					data[_i(i, j, nz)] = buff[i * ny + j];
				}
			}
		}

		MPI_Barrier(MPI_COMM_WORLD);
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

		MPI_Allreduce(MPI_IN_PLACE, &cur_eps, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	}

	if (id != 0) {
		for (int k = 0; k < nz; k++) {
			for (int j = 0; j < ny; j++) {
				for (int i = 0; i < nx; i++) {
					buff[i] = data[_i(i, j, k)];
				}
				MPI_Send(buff, nx, MPI_DOUBLE, 0, id, MPI_COMM_WORLD);
			}
		}
	} else {
		FILE* f = fopen(fname, "w");
		for (int kb = 0; kb < nbz; kb++) {
			for (int k = 0; k < nz; k++) {
				for (int jb = 0; jb < nby; jb++) {
					for (int j = 0; j < ny; j++) {
						for (int ib = 0; ib < nbx; ib++) {
							if (_ib(ib, jb, kb) == 0) {
								for (int i = 0; i < nx; i++) {
									buff[i] = data[_i(i, j, k)];
								}
							} else {
								MPI_Recv(buff, nx, MPI_DOUBLE, _ib(ib, jb, kb), _ib(ib, jb, kb), MPI_COMM_WORLD, &status);
							}
							for (int i = 0; i < nx; i++) {
								fprintf(f, "%.6e ", buff[i]);
							}
						}
					}
				}
			}
		}
		fclose(f);
	}
	MPI_Finalize();
	free(data);
	free(next);
	free(buff);
	return 0;
}
