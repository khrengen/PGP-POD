#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <math.h>
#include <mpi.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>

#define CSC(call)                                                   \
do {                                                                \
    cudaError_t res = call;                                         \
    if (res != cudaSuccess) {                                       \
        fprintf(stderr, "ERROR in %s:%d. Message: %s\n",            \
                __FILE__, __LINE__, cudaGetErrorString(res));       \
        exit(0);                                                    \
    }                                                               \
} while(0)


#define _i(i, j, k) (((k) + 1) * (nx + 2) * (ny + 2) + ((j) + 1) * (nx + 2) + (i) + 1)
#define _ib(i, j, k) ((k) * nbx * nby + (j) * nbx + (i))

const int N_DIMS = 3;
const dim3 GRID_SIZE = dim3(32,32);
const dim3 BLOCK_SIZE = dim3(32,32);


__global__ void kernel_init(double* data, int nx, int ny, int nz, double u0) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int idz = blockDim.z * blockIdx.z + threadIdx.z;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;
    int offsetz = blockDim.z * gridDim.z;

    for (int k = idz; k < nz; k+=offsetz) {
    	for (int j = idy; j < ny; j+=offsety) {
    		for (int i = idx; i < nx; i+=offsetx) {
    			data[_i(i, j, k)] = u0;
    		}
    	}
    }
}

__global__ void kernel_LR_bc(double* data, int nx, int ny, int nz, double bc, int x_ind) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    for (int k = idy; k < nz; k+=offsety) {
    	for (int j = idx; j < ny; j+=offsetx) {
    		data[_i(x_ind, j, k)] = bc;
    	}
    }
}

__global__ void kernel_FB_bc(double* data, int nx, int ny, int nz, double bc, int y_ind) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    for (int k = idy; k < nz; k+=offsety) {
    	for (int i = idx; i < nx; i+=offsetx) {
    		data[_i(i, y_ind, k)] = bc;
    	}
    }
}

__global__ void kernel_DU_bc(double* data, int nx, int ny, double bc, int z_ind) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    for (int j = idy; j < ny; j+=offsety) {
    	for (int i = idx; i < nx; i+=offsetx) {
    		data[_i(i, j, z_ind)] = bc;
    	}
    }
}

__global__ void kernel_copy_send_LR(double* buf, double* data, int nx, int ny, int nz, int x_ind) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    for (int k = idy; k < nz; k+=offsety) {
    	for (int j = idx; j < ny; j+=offsetx) {
    		buf[k * ny + j] = data[_i(x_ind, j, k)];
    	}
    }
}

__global__ void kernel_copy_recive_LR(double* buf, double* data, int nx, int ny, int nz, int x_ind) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    for (int k = idy; k < nz; k+=offsety) {
    	for (int j = idx; j < ny; j+=offsetx) {
    		data[_i(x_ind, j, k)] = buf[k * ny + j];
    	}
    }
}

__global__ void kernel_copy_send_FB(double* buf, double* data, int nx, int ny, int nz, int y_ind) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    for (int k = idy; k < nz; k+=offsety) {
    	for (int i = idx; i < nx; i+=offsetx) {
    		buf[k * nx + i] = data[_i(i, y_ind, k)];
    	}
    }
}

__global__ void kernel_copy_recive_FB(double* buf, double* data, int nx, int ny, int nz, int y_ind) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    for (int k = idy; k < nz; k+=offsety) {
    	for (int i = idx; i < nx; i+=offsetx) {
    		data[_i(i, y_ind, k)] = buf[k * nx + i];
    	}
    }
}

__global__ void kernel_copy_send_DU(double* buf, double* data, int nx, int ny, int z_ind) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    for (int j = idy; j < ny; j+=offsety) {
    	for (int i = idx; i < nx; i+=offsetx) {
    		buf[j * nx + i] = data[_i(i, j, z_ind)];
    	}
    }
}

__global__ void kernel_copy_recive_DU(double* buf, double* data, int nx, int ny, int z_ind) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    for (int j = idy; j < ny; j+=offsety) {
    	for (int i = idx; i < nx; i+=offsetx) {
    		data[_i(i, j, z_ind)] = buf[j * nx + i];
    	}
    }
}

__global__ void kernel_get_vals(double* data, double* next, int nx, int ny, int nz, double hx, double hy, double hz) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int idz = blockDim.z * blockIdx.z + threadIdx.z;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;
    int offsetz = blockDim.z * gridDim.z;

    for (int k = idz; k < nz; k+=offsetz) {
    	for (int j = idy; j < ny; j+=offsety) {
    		for (int i = idx; i < nx; i+=offsetx) {
    			next[_i(i, j, k)] = 0.5 * ((data[_i(i + 1, j, k)] + data[_i(i - 1, j, k)]) / (hx * hx) +
						(data[_i(i, j + 1, k)] + data[_i(i, j - 1, k)]) / (hy * hy) +
						(data[_i(i, j, k + 1)] + data[_i(i, j, k - 1)]) / (hz * hz)) /
						(1.0 / (hx * hx) + 1.0 / (hy * hy) + 1.0 / (hz * hz));
    		}
    	}
    }
}

__global__ void kernel_get_diffs(double* data, double* next, int nx, int ny, int nz) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int idz = blockDim.z * blockIdx.z + threadIdx.z;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;
    int offsetz = blockDim.z * gridDim.z;

    for (int k = idz - 1; k <= nz; k+=offsetz) {
    	for (int j = idy - 1; j <= ny; j+=offsety) {
    		for (int i = idx - 1; i <= nx; i+=offsetx) {
    			data[_i(i, j, k)] = ((i != -1) && (j != -1) && (k != -1) && (i != nx) && (j != ny) && (k != nz)) * fabs(next[_i(i, j, k)] - data[_i(i, j, k)]);
    		}
    	}
    }
}


int main(int argc, char* argv[]) {
	int ib, jb, kb, nbx, nby, nbz, nx, ny, nz;
	int id, numproc;
	double lx, ly, lz, hx, hy, hz, bc_down, bc_up, bc_left, bc_right, bc_front, bc_back, u0;
	double eps, cur_eps;
	double *temp, *buff;
	char fname[100];

	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);

	int deviceCount;
	CSC(cudaGetDeviceCount(&deviceCount));
	CSC(cudaSetDevice(id % deviceCount));

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
	MPI_Bcast(fname, 100, MPI_CHAR, 0, MPI_COMM_WORLD);

	kb = id / (nbx * nby);
	jb = id % (nbx * nby) / nbx;
	ib = id % (nbx * nby) % nbx;

	hx = lx / (nx * nbx);
	hy = ly / (ny * nby);
	hz = lz / (nz * nbz);

	double *dev_data;
	double *dev_next;
	CSC(cudaMalloc(&dev_data, sizeof(double) * (nx + 2) * (ny + 2) * (nz + 2)));
	CSC(cudaMalloc(&dev_next, sizeof(double) * (nx + 2) * (ny + 2) * (nz + 2)));

	int buf_side1 = std::max(nx, ny);
	int buf_side2 = std::max(ny, nz);

	buff = (double*)malloc(sizeof(double) * buf_side1 * buf_side2);
	int buffer_size;
	MPI_Pack_size(buf_side1 * buf_side2, MPI_DOUBLE, MPI_COMM_WORLD, &buffer_size);
	buffer_size = 6 * (buffer_size + MPI_BSEND_OVERHEAD);
	double* buffer = (double*)malloc(buffer_size);
	MPI_Buffer_attach(buffer, buffer_size);
	double* dev_buff;
	CSC(cudaMalloc(&dev_buff, sizeof(double) * buf_side1 * buf_side2));

	kernel_init<<<dim3(8, 8, 8), dim3(32, 4, 4)>>>(dev_data, nx, ny, nz, u0);
	CSC(cudaGetLastError());

	cur_eps = eps + 1;
	while (cur_eps >= eps) {
		MPI_Barrier(MPI_COMM_WORLD);

		if (ib + 1 < nbx) {
			kernel_copy_send_LR<<<GRID_SIZE, BLOCK_SIZE>>>(dev_buff, dev_data, nx, ny, nz, nx-1);
			CSC(cudaGetLastError());
			CSC(cudaMemcpy(buff, dev_buff, sizeof(double) * ny * nz, cudaMemcpyDeviceToHost));
			MPI_Bsend(buff, ny * nz, MPI_DOUBLE, _ib(ib + 1, jb, kb), id, MPI_COMM_WORLD);
		}

		if (jb + 1 < nby) {
			kernel_copy_send_FB<<<GRID_SIZE, BLOCK_SIZE>>>(dev_buff, dev_data, nx, ny, nz, ny-1);
			CSC(cudaGetLastError());
			CSC(cudaMemcpy(buff, dev_buff, sizeof(double) * nx * nz, cudaMemcpyDeviceToHost));
			MPI_Bsend(buff, nx * nz, MPI_DOUBLE, _ib(ib, jb + 1, kb), id, MPI_COMM_WORLD);
		}

		if (kb + 1 < nbz) {
			kernel_copy_send_DU<<<GRID_SIZE, BLOCK_SIZE>>>(dev_buff, dev_data, nx, ny, nz-1);
			CSC(cudaGetLastError());
			CSC(cudaMemcpy(buff, dev_buff, sizeof(double) * nx * ny, cudaMemcpyDeviceToHost));
			MPI_Bsend(buff, nx * ny, MPI_DOUBLE, _ib(ib, jb, kb + 1), id, MPI_COMM_WORLD);
		}

		if (ib > 0) {
			kernel_copy_send_LR<<<GRID_SIZE, BLOCK_SIZE>>>(dev_buff, dev_data, nx, ny, nz, 0);
			CSC(cudaGetLastError());
			CSC(cudaMemcpy(buff, dev_buff, sizeof(double) * ny * nz, cudaMemcpyDeviceToHost));
			MPI_Bsend(buff, ny * nz, MPI_DOUBLE, _ib(ib - 1, jb, kb), id, MPI_COMM_WORLD);
		}

		if (jb > 0) {
			kernel_copy_send_FB<<<GRID_SIZE, BLOCK_SIZE>>>(dev_buff, dev_data, nx, ny, nz, 0);
			CSC(cudaGetLastError());
			CSC(cudaMemcpy(buff, dev_buff, sizeof(double) * nx * nz, cudaMemcpyDeviceToHost));
			MPI_Bsend(buff, nx * nz, MPI_DOUBLE, _ib(ib, jb - 1, kb), id, MPI_COMM_WORLD);
		}

		if (kb > 0) {
			kernel_copy_send_DU<<<GRID_SIZE, BLOCK_SIZE>>>(dev_buff, dev_data, nx, ny, 0);
			CSC(cudaGetLastError());
			CSC(cudaMemcpy(buff, dev_buff, sizeof(double) * nx * ny, cudaMemcpyDeviceToHost));
			MPI_Bsend(buff, nx * ny, MPI_DOUBLE, _ib(ib, jb, kb - 1), id, MPI_COMM_WORLD);
		}

		/*-----------------------------------------------------------------------------------*/
		if (ib > 0) {
			MPI_Recv(buff, ny * nz, MPI_DOUBLE, _ib(ib - 1, jb, kb), _ib(ib - 1, jb, kb), MPI_COMM_WORLD, &status);
			CSC(cudaMemcpy(dev_buff, buff, sizeof(double) * ny * nz, cudaMemcpyHostToDevice));
			kernel_copy_recive_LR<<<GRID_SIZE, BLOCK_SIZE>>>(dev_buff, dev_data, nx, ny, nz, -1);
			CSC(cudaGetLastError());
		} else {
			kernel_LR_bc<<<GRID_SIZE, BLOCK_SIZE>>>(dev_data, nx, ny, nz, bc_left, -1);
			CSC(cudaGetLastError());
		}

		if (jb > 0) {
			MPI_Recv(buff, nx * nz, MPI_DOUBLE, _ib(ib, jb - 1, kb), _ib(ib, jb - 1, kb), MPI_COMM_WORLD, &status);
			CSC(cudaMemcpy(dev_buff, buff, sizeof(double) * nx * nz, cudaMemcpyHostToDevice));
			kernel_copy_recive_FB<<<GRID_SIZE, BLOCK_SIZE>>>(dev_buff, dev_data, nx, ny, nz, -1);
			CSC(cudaGetLastError());
		} else {
			kernel_FB_bc<<<GRID_SIZE, BLOCK_SIZE>>>(dev_data, nx, ny, nz, bc_front, -1);
			CSC(cudaGetLastError());
		}

		if (kb > 0) {
			MPI_Recv(buff, nx * ny, MPI_DOUBLE, _ib(ib, jb, kb - 1), _ib(ib, jb, kb - 1), MPI_COMM_WORLD, &status);
			CSC(cudaMemcpy(dev_buff, buff, sizeof(double) * nx * ny, cudaMemcpyHostToDevice));
			kernel_copy_recive_DU<<<GRID_SIZE, BLOCK_SIZE>>>(dev_buff, dev_data, nx, ny, -1);
			CSC(cudaGetLastError());
		} else {
			kernel_DU_bc<<<GRID_SIZE, BLOCK_SIZE>>>(dev_data, nx, ny, bc_down, -1);
			CSC(cudaGetLastError());
		}

		if (ib + 1 < nbx) {
			MPI_Recv(buff, ny * nz, MPI_DOUBLE, _ib(ib + 1, jb, kb), _ib(ib + 1, jb, kb), MPI_COMM_WORLD, &status);
			CSC(cudaMemcpy(dev_buff, buff, sizeof(double) * ny * nz, cudaMemcpyHostToDevice));
			kernel_copy_recive_LR<<<GRID_SIZE, BLOCK_SIZE>>>(dev_buff, dev_data, nx, ny, nz, nx);
			CSC(cudaGetLastError());
		} else {
			kernel_LR_bc<<<GRID_SIZE, BLOCK_SIZE>>>(dev_data, nx, ny, nz, bc_right, nx);
			CSC(cudaGetLastError());
		}

		if (jb + 1 < nby) {
			MPI_Recv(buff, nx * nz, MPI_DOUBLE, _ib(ib, jb + 1, kb), _ib(ib, jb + 1, kb), MPI_COMM_WORLD, &status);
			CSC(cudaMemcpy(dev_buff, buff, sizeof(double) * nx * nz, cudaMemcpyHostToDevice));
			kernel_copy_recive_FB<<<GRID_SIZE, BLOCK_SIZE>>>(dev_buff, dev_data, nx, ny, nz, ny);
			CSC(cudaGetLastError());
		} else {
			kernel_FB_bc<<<GRID_SIZE, BLOCK_SIZE>>>(dev_data, nx, ny, nz, bc_back, ny);
			CSC(cudaGetLastError());
		}

		if (kb + 1 < nbz) {
			MPI_Recv(buff, nx * ny, MPI_DOUBLE, _ib(ib, jb, kb + 1), _ib(ib, jb, kb + 1), MPI_COMM_WORLD, &status);
			CSC(cudaMemcpy(dev_buff, buff, sizeof(double) * nx * ny, cudaMemcpyHostToDevice));
			kernel_copy_recive_DU<<<GRID_SIZE, BLOCK_SIZE>>>(dev_buff, dev_data, nx, ny, nz);
			CSC(cudaGetLastError());
		} else {
			kernel_DU_bc<<<GRID_SIZE, BLOCK_SIZE>>>(dev_data, nx, ny, bc_up, nz);
			CSC(cudaGetLastError());
		}

		MPI_Barrier(MPI_COMM_WORLD);
		cur_eps = 0.0;

		kernel_get_vals<<<dim3(8, 8, 8), dim3(32, 4, 4)>>>(dev_data, dev_next, nx, ny, nz, hx, hy, hz);
		CSC(cudaGetLastError());

		kernel_get_diffs<<<dim3(8, 8, 8), dim3(32, 4, 4)>>>(dev_data, dev_next, nx, ny, nz);
		CSC(cudaGetLastError());

		thrust::device_ptr<double> diffs = thrust::device_pointer_cast(dev_data);
        thrust::device_ptr<double> max_eps = thrust::max_element(diffs, diffs + (nx + 2) * (ny + 2) * (nz + 2));
        cur_eps =  *max_eps;
        MPI_Allreduce(MPI_IN_PLACE, &cur_eps, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

		temp = dev_next;
		dev_next = dev_data;
		dev_data = temp;
		
	}

	double* data = (double *)malloc(sizeof(double) * (nx + 2) * (ny + 2) * (nz + 2));
	cudaMemcpy(data, dev_data, sizeof(double) * (nx + 2) * (ny + 2) * (nz + 2), cudaMemcpyDeviceToHost);

	int n_size = 14;	//знак(-) + мантисса + порядок + '\0' = 1 + 8 + 4 + 1
	char* bf = (char*)malloc(sizeof(char) * nx * ny * nz * n_size);
	memset(bf, ' ', sizeof(char) * nx * ny * nz * n_size);

	for (int k = 0; k < nz; k++) {
		for (int j = 0; j < ny; j++) {
			for (int i = 0; i < nx; i++) {
				sprintf(bf + (k * nx * ny + j * nx + i) * n_size, "%.6e", data[_i(i, j, k)]);
			}
		}
	}
	
	for (int i = 0; i < nx * ny * nz * n_size; i++) {
		if (bf[i] == '\0') {
			bf[i] = ' ';
		}
	}
	
	MPI_File fp;
	MPI_Datatype filetype;
	int sizes[] = {nz * nbz, ny * nby, nx * nbx * n_size};
	int subsizes[] = {nz, ny, nx * n_size};
	int starts[] = {id / (nbx * nby) * nz, id % (nbx * nby) / nbx * ny,  id % (nbx * nby) % nbx * nx * n_size};
	MPI_Type_create_subarray(N_DIMS, sizes, subsizes, starts, MPI_ORDER_C, MPI_CHAR, &filetype);
	MPI_Type_commit(&filetype);

	MPI_File_delete(fname, MPI_INFO_NULL);
	MPI_File_open(MPI_COMM_WORLD, fname, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fp);
	MPI_File_set_view(fp, 0, MPI_CHAR, filetype, "native", MPI_INFO_NULL);
	MPI_File_write_all(fp, bf, nx * ny * nz * n_size, MPI_CHAR, MPI_STATUS_IGNORE);
	MPI_File_close(&fp);

	MPI_Type_free(&filetype);
	MPI_Finalize();
    CSC(cudaFree(dev_data));
	CSC(cudaFree(dev_next));
	CSC(cudaFree(dev_buff));
	free(data);
	free(buff);
	free(bf);
	return 0;
}