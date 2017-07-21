#ifndef IONWAKE_108
#define IONWAKE_108

#include "cuda_runtime.h"

__global__ void calcIonDustForces(float3* , float3* , unsigned int * const ,
	float * const ,
	float * const , unsigned int* const, float3*);

#endif