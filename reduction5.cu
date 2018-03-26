
__global__ void reduce5(int *g_idata, int *g_odata) {
	extern __shared__ int sdata[];
	
	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
	sdata[tid] = g_idata[i] + g)idata[i+blockDim.x];
	__syncthreads();
	
	// do reduction in shared mem
	for(unsigned int s=blockDim.x/2; s>32;s >>=1) {
	   
	   if (tid < s) {
	      sdata[tid] += sdata[tid + s];
	   }
	   __syncthreads();
	}
	
	if (tid < 32){
	  sdata[tid] += sdata[tid + 32];
	  sdata[tid] += sdata[tid + 16];
	  sdata[tid] += sdata[tid + 8];
	  sdata[tid] += sdata[tid + 4];
	  sdata[tid] += sdata[tid + 2];
	  sdata[tid] += sdata[tid + 1];
	}
	
	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}