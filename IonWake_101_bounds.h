#ifndef PK4_101_INCLUDED
#define PK4_101_INCLUDED

#include "cuda_runtime.h"
#include <curand_kernel.h>

__global__ void replaceOutOfBoundsIons(float3* d_posIon, float3* d_velIon,
	curandState_t* statesThread, curandState_t* statesBlock,
	float* const d_RAD_SIM_SQRD, float* const d_RAD_SIM, unsigned int* const d_NUM_ION,
	unsigned int* const d_NUM_DUST, float3* d_posDust, float* const d_RAD_DUST_SQRD);
__global__ void init(unsigned int, curandState_t*);

#endif // PK4_101_INCLUDED