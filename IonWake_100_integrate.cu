#include "IonWake_100_integrate.h"

__global__ void stepForward(float3* d_posIon, float3* d_velIon, float3* d_accIon, float* const d_HALF_TIME_STEP)
{
	int IDion = blockIdx.x * blockDim.x + threadIdx.x;

	// calculate velocity
	d_velIon[IDion].x += ((*d_HALF_TIME_STEP) * d_accIon[IDion].x);
	d_velIon[IDion].y += ((*d_HALF_TIME_STEP) * d_accIon[IDion].y);
	d_velIon[IDion].z += ((*d_HALF_TIME_STEP) * d_accIon[IDion].z);

	__syncthreads();

	// calculate position
	d_accIon[IDion].x += ((*d_HALF_TIME_STEP) * d_velIon[IDion].x);
	d_accIon[IDion].y += ((*d_HALF_TIME_STEP) * d_velIon[IDion].y);
	d_accIon[IDion].z += ((*d_HALF_TIME_STEP) * d_velIon[IDion].z);
	
	__syncthreads();

	// reset acceleration
	d_accIon[IDion].x = 0;
	d_accIon[IDion].y = 0;
	d_accIon[IDion].z = 0;

	__syncthreads();

}
