#include <stdio.h>

//CUDA kernel
__global__ void vectorAdd(const float *a, const float *b, float *c, int num_elems){
	//Thread ID retrieval
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	
	if (i < num_elems){
		c[i] = a[i] + b[i];
	}
}

//GPU space
int main(void){
	//CUDA call error code
	cudaError_t err = cudaSuccess;
	char p_name[5] = "vAdd";
	
	int num_elems = 10000;
	size_t size = num_elems*sizeof(float);
	printf("vectorAdd( num_elems = %d )\n", num_elems);
	
	//Host input vectors alloc
	float *h_a = (float *)malloc(size);
	float *h_b = (float *)malloc(size);
	float *h_c = (float *)malloc(size);
	
	if(h_a == NULL || h_b == NULL || h_c == NULL){
		fprintf(stderr, "%s: Failed host vector allocation.\n", p_name);
		exit(EXIT_FAILURE);
	}

	//Host vector init
	for (int i = 0; i < num_elems; ++i)
    {
        h_a[i] = rand()/(float)RAND_MAX;
        h_b[i] = rand()/(float)RAND_MAX;
    }

	//Device vectors alloc
	float *d_a = NULL;
    err = cudaMalloc((void **)&d_a, size);
	if (err != cudaSuccess){
        fprintf(stderr, "%s: Failed device vector \"a\" allocation.\n", p_name);
		exit(EXIT_FAILURE);
    }
	float *d_b = NULL;
    err = cudaMalloc((void **)&d_b, size);
	if (err != cudaSuccess){
        fprintf(stderr, "%s: Failed device vector \"b\" allocation.\n", p_name);
		exit(EXIT_FAILURE);
    }
	float *d_c = NULL;
    err = cudaMalloc((void **)&d_c, size);
	if (err != cudaSuccess){
        fprintf(stderr, "%s: Failed device vector \"c\" allocation.\n", p_name);
		exit(EXIT_FAILURE);
    }

	//Copying from host to device
	err = cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "%s: Failed to copy vector \"a\" from host to device (error code %s),\n", p_name, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	err = cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "%s: Failed to copy vector \"b\" from host to device (error code %s),\n", p_name, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	//Launch vAdd kernel
	int threads_per_block = 256;
	// The threads_per_block - 1 pattern ensures ceiling for the division  
    int blocks_per_grid = (num_elems + threads_per_block - 1) / threads_per_block;
	printf("%s: CUDA kernel launch with %d blocks of %d threads\n", p_name, blocks_per_grid, threads_per_block);
	vectorAdd<<<blocks_per_grid, threads_per_block>>>(d_a, d_b, d_c, num_elems);
	err = cudaGetLastError();
    if (err != cudaSuccess){
		fprintf(stderr, "%s: Failed to launch vectorAdd kernel (error code %s)!\n", p_name, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
	}

	//Device to Host result
    err = cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "%s: Failed to copy vector C from device to host (error code %s)!\n", p_name, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	//Error for each element < e^{-5}
	for (int i = 0; i < num_elems; ++i)
    {
        if (fabs(h_a[i] + h_b[i] - h_c[i]) > 1e-5)
        {
            fprintf(stderr, "%s: Result verification failed at element %d!\n", p_name, i);
            exit(EXIT_FAILURE);
        }
    }
    printf("%s: Test PASSED\n", p_name);

	//Free device vectors
	err = cudaFree(d_a);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "%s: Failed to free device vector a (error code %s)!\n", p_name, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	err = cudaFree(d_b);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "%s: Failed to free device vector b (error code %s)!\n", p_name, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	err = cudaFree(d_c);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "%s: Failed to free device vector c (error code %s)!\n", p_name, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	// Free host memory
    free(h_a);
    free(h_b);
    free(h_c);

    printf("%s: Finished succesfully.\n", p_name);
    return 0;
}