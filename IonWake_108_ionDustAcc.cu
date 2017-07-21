
#include "IonWake_108_ionDustAcc.h"

__global__ void
calcIonDustForces(float3* d_posIon, float3* d_accIon, unsigned int * const d_NUM_ION,
	float * const d_SOFT_RAD_SQRD,
	float * const d_INV_DEBYE, unsigned int* const d_NUM_DUST, float3* d_posDust)
{
	
	// a holders distances
	float3 dist;
	float distSqrd;
	double softDistCubed;
	float softDist;
	float exponent;
	float hardDist;
	float forceInter;
	
	// Index of the current ion
	int IDcrntIon = blockIdx.x * blockDim.x + threadIdx.x;
	
	// position of the current ion
	float3 posCrntIon = d_posIon[IDcrntIon];

	// initialize a variable to store the acceleration for 
	// the current ion in
	float3 accCrntIon = { 0.0, 0.0, 0.0 };

	// wait for all positions to be copied
	__syncthreads();

	// loop over all of the dust particles
	for (int i = 0; i < *d_NUM_DUST; i++)
	{
		float3 dist;

		// calculate the distance between the current ion
		// and the ion in the shared memory
		dist.x = d_posDust[i].x - posCrntIon.x;
		dist.y = d_posDust[i].y - posCrntIon.y;
		dist.z = d_posDust[i].z - posCrntIon.z;

		// softened distance between the two ions squared
		distSqrd = dist.x*dist.x + dist.y*dist.y + dist.z*dist.z;

		// softened scaler distance between the two ions
		softDist = __fsqrt_rn(distSqrd + *d_SOFT_RAD_SQRD);

		// un-softend scaler distance between the two ions
		hardDist = __fsqrt_rn(distSqrd);

		// tcalculate the force multiplier
		forceInter = 10e-10 *  
			(1.f + hardDist * (*d_INV_DEBYE)) *
			__expf(-hardDist * (*d_INV_DEBYE)) /
			__fsqrt_rn(softDist * softDist * softDist);

		// calculate the vector accelartion
		accCrntIon.x += dist.x * forceInter;
		accCrntIon.y += dist.y * forceInter;
		accCrntIon.z += dist.z * forceInter;
	}

	// wait for all ptheads to finish calculating
	__syncthreads();
	

	// Save the result in global memory.
	d_accIon[IDcrntIon].x += accCrntIon.x;
	d_accIon[IDcrntIon].y += accCrntIon.y;
	d_accIon[IDcrntIon].z += accCrntIon.z;

	__syncthreads();
}