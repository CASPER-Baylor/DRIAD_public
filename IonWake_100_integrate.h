#ifndef PK4_100_INCLUDED
#define PK4_100_INCLUDED

#include "cuda_runtime.h"

__global__ void stepForward(float3*, float3*, float3*, float* const);

#endif // PK4_100_INCLUDED
