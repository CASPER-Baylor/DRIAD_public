
#include "IonWake_102_ionIonAcc.h"

__global__ void
calcIonIonForces(float3* d_posIon, float3* d_accIon, unsigned int * const d_NUM_ION,
	float * const d_SOFT_RAD_SQRD, float * const d_ION_ION_ACC_MULT,
	float * const d_INV_DEBYE)
{
	// allocate memory for set of ion positions
	extern __shared__ float3 sh_posIon[];

	// Index of the current ion
	int IDcrntIon = blockIdx.x * blockDim.x + threadIdx.x;

	// position of the current ion
	float3 posCrntIon = d_posIon[IDcrntIon];

	// initialize a variable to store the acceleration for 
	// the current ion in
	float3 accCrntIon = { 0.0f, 0.0f, 0.0f };

	int IDcopyIon, tile;
	unsigned int i;
	float distSqrd;
	float softDist;
	float hardDist;
	float forceInter;

	// loop over each tile
	for (i = 0, tile = 0; i < *d_NUM_ION; i += blockDim.x, tile++)
	{
		// Index for the ion to be copied 
		IDcopyIon = tile * blockDim.x + threadIdx.x;

		// copy the ion possition into shared memory
		sh_posIon[threadIdx.x] = d_posIon[IDcopyIon];

		// wait for all positions to be copied
		__syncthreads();

		// loop over all of the ions in the shared memory
		for (int i = 0; i < blockDim.x; i++)
		{
			float3 dist;

			// calculate the distance between the current ion
			// and the ion in the shared memory
			dist.x = sh_posIon[i].x - posCrntIon.x;
			dist.y = sh_posIon[i].y - posCrntIon.y;
			dist.z = sh_posIon[i].z - posCrntIon.z;

			// softened distance between the two ions squared
			distSqrd = dist.x*dist.x + dist.y*dist.y + dist.z*dist.z;

			// softened scaler distance between the two ions
			softDist = __fsqrt_rn(distSqrd + *d_SOFT_RAD_SQRD);

			// un-softend scaler distance between the two ions
			hardDist = __fsqrt_rn(distSqrd);

			// tcalculate the force multiplier
			forceInter = 3e11 * 30000 *
				(1.f + hardDist * (*d_INV_DEBYE)) *
				__expf(-hardDist * (*d_INV_DEBYE)) /
				__fsqrt_rn(softDist * softDist * softDist);

			// calculate the vector accelartion
			accCrntIon.x += dist.x * forceInter;
			accCrntIon.y += dist.y * forceInter;
			accCrntIon.z += dist.z * forceInter;
		}
	
	}

	// wait for all threads to finish calcTile()
	__syncthreads();


	// Save the result in global memory.
	d_accIon[i] = accCrntIon;
	
}