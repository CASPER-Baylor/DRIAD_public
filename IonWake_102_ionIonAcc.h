#ifndef PK4_102_INCLUDED
#define PK4_102_INCLUDED

#include "cuda_runtime.h"

__global__ void calcIonIonForces(float3*, float3*, unsigned int * const,
	float * const, float * const, float * const);

#endif // PK4_102_INCLUDED
