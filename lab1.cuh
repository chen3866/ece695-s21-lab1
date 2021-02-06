#include "cuda_runtime.h"
#include <stdlib.h>
#include <iostream>
#include<time.h>
#include<iomanip>
using namespace std;

__global__
void saxpy_gpu(float* x, float* y, float scale, size_t size) {
	//	Insert GPU SAXPY kernel code here
	int i = blockDim.x * blockIdx.x + threadIdx.x; //vector is 1-dim, blockDim means the number of thread in a block

	for (int i = 0; i < size; i++) {
		y[i] = scale * x[i] + y[i];
	}
}

int runGpuSaxpy(int vectorSize) {

	//cout << "Hello GPU Saxpy!" << endl;

	//	Insert code here

	int num = vectorSize; // size of vector
	size_t size = num * sizeof(float);

	// host memery
	float* x = (float*)malloc(size);
	float* y = (float*)malloc(size);

	float* temp = (float*)malloc(size);
	float* error = (float*)malloc(size);

	// init the vector
	for (int i = 1; i < num; ++i) {
		x[i] = rand() / (float)RAND_MAX;
		y[i] = rand() / (float)RAND_MAX;
		temp[i] = y[i];
	}
	float scale = rand() / (float)RAND_MAX;

	// copy the host memery to device memery
	float* device_x = NULL;
	float* device_y = NULL;

	cudaMalloc((void**)&device_x, size);
	cudaMalloc((void**)&device_y, size);

	cudaMemcpy(device_x, x, size, cudaMemcpyHostToDevice);
	cudaMemcpy(device_y, y, size, cudaMemcpyHostToDevice);

	// launch function add kernel
	int threadPerBlock = 256;
	int blockPerGrid = (num + threadPerBlock - 1) / threadPerBlock;
	printf("threadPerBlock: %d \nblockPerGrid: %d \n", threadPerBlock, blockPerGrid);

	saxpy_gpu << < blockPerGrid, threadPerBlock >> > (device_x, device_y, scale, size);

	//copy the device result to host
	cudaMemcpy(y, device_y, size, cudaMemcpyDeviceToHost);

	// Verify that the result vector is correct
	int cnt = 0;
	for (int i = 0; i < num; i++) {
		error[i] = x[i] * scale + temp[i] - y[i];
		if (fabs(error[i]) > 1e-5) {
			//cout << i << " Result verification failed at element and error is " << error[i] << endl;
			cnt++;
		}
	}
	if (cnt == 0) cout << "test passed" << endl;



	// Free device global memory
	cudaFree(device_x);
	cudaFree(device_y);

	// Free host memory
	free(x);
	free(y);

	return 0;
}

//////////////////////////////////////

//_global__
int runCpuSaxpy(int vectorSize)
{
	//cout << "Hello CPU Saxpy!" << endl;
	int num = vectorSize;
	size_t size = num * sizeof(float);

	float* x, * y;
	x = (float*)malloc(size);
	y = (float*)malloc(size);

	float* temp = (float*)malloc(size);
	float* error = (float*)malloc(size);

	float scale = rand() / (float)RAND_MAX;

	for (int i = 0; i < num; i++)
	{
		x[i] = rand() / (float)RAND_MAX;
		y[i] = rand() / (float)RAND_MAX;
		temp[i] = y[i];
		y[i] = scale * x[i] + y[i];

	}

	int cnt = 0;
	for (int i = 0; i < num; i++) {
		error[i] = x[i] * scale + temp[i] - y[i];
		if (fabs(error[i]) > 1e-5) {
			cout << i << " Result verification failed at element and error is " << error[i] << endl;
			cnt++;
		}
	}
	if (cnt == 0) cout << "test passed" << endl;
	return 0;
}

//////////////////////////

int runCpuMCPi(int vectorSize)
{
	//cout << "Hello run Gpu MCPi!" << endl;
	int num = vectorSize;
	size_t size = num * sizeof(float);

	float* x, * y;
	x = (float*)malloc(size);
	y = (float*)malloc(size);

	int cnt = 0;

	for (int i = 0; i < num; i++)
	{
		x[i] = rand() / (float)RAND_MAX;
		y[i] = rand() / (float)RAND_MAX;
		if ((x[i] * x[i] + y[i] * y[i]) <= 1) cnt++;
	}

	cout << "pi is" << float(4 * cnt / num) << endl;

	return 0;
}
/////////////////////////////////


using namespace std;

// vectorAdd run in device
__global__ void
MCpi_gpu(float* a, float* b, float* c, int num) {

	int idx = blockDim.x * blockIdx.x + threadIdx.x; //vector is 1-dim, blockDim means the number of thread in a block

	int cnt = 0;
	for (int i = 0; i < num; i++) {
		c[i] = a[i] * a[i] + b[i] * b[i];
	}

}
// main run in host
int
runGpuMCPi(int vectorSize) {
	float num = vectorSize; // size of vector
	size_t size = num * sizeof(float);

	// host memery
	float* a = (float*)malloc(size);
	float* b = (float*)malloc(size);
	float* c = (float*)malloc(size);

	// init the vector
	for (int i = 1; i < num; i++) {
		a[i] = rand() / (float)RAND_MAX;
		b[i] = rand() / (float)RAND_MAX;

		//cout << a[i] << " " << b[i] << endl;
	}

	// copy the host memery to device memery
	float* da = NULL;
	float* db = NULL;
	float* dc = NULL;

	cudaMalloc((void**)&da, size);
	cudaMalloc((void**)&db, size);
	cudaMalloc((void**)&dc, size);

	cudaMemcpy(da, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(db, b, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dc, c, size, cudaMemcpyHostToDevice);

	// launch function add kernel
	int threadPerBlock = 256;
	int blockPerGrid = (num + threadPerBlock - 1) / threadPerBlock;
	printf("threadPerBlock: %d \nblockPerGrid: %d \n", threadPerBlock, blockPerGrid);

	MCpi_gpu << < blockPerGrid, threadPerBlock >> > (da, db, dc, num);

	//copy the device result to host
	cudaMemcpy(c, dc, size, cudaMemcpyDeviceToHost);


	int cnt = 0;
	for (int i = 0; i < num; i++) {
		if (c[i] <= 1)	cnt++;
	}

	float pi = 4 * cnt / num;

	cout << "pi is " << pi << endl;


	printf("Test PASSED\n");

	// Free device global memory
	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);
	// Free host memory
	free(a);
	free(b);
	free(c);
	return 0;
}

