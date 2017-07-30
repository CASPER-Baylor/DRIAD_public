
#include "IonWake_102_ionIonAcc.h"

__global__ void
calcIonIonForces(float3* d_posIon, float3* d_accIon, unsigned int * const d_NUM_ION,
	float * const d_SOFT_RAD_SQRD, float * const d_ION_ION_ACC_MULT,
	float * const d_INV_DEBYE)
{
	// the thread block size is assumed to be 256
	// to change the thread block size change the value 
	// of t_block_dim
	const int t_block_dim = 256;

	// index of the current ion
	int IDcrntIon = blockIdx.x * blockDim.x + threadIdx.x;

	// allocate variables
	float3 dist;
	float distSquared;
	float hardDist;
	float softDist;
	float linForce;
	int tileThreadID;

	// allocate shared memory
	__shared__ float3 sharedPos[t_block_dim];

	// loop over all of the ions by ussing tiles. Where each tile is a section
	// of the ions that is loaded into shared memory. Each tile consists of 
	// as many ions as the block size. Each thread is responsible for loading 
	// one ion position for the tile.
	for (int tileOffset = 0; tileOffset < *d_NUM_ION; tileOffset += blockDim.x)
	{
		// the index of the ion for the thread to load
		// for the current tile
		tileThreadID = tileOffset + threadIdx.x; 

		// load in an ion position
		sharedPos[threadIdx.x] = d_posIon[tileThreadID];
		
		// wait for all threads to load the current position
		__syncthreads();

		// loop over all of the ions loaded in the tile
		for (int h = 0; h < t_block_dim; h++)
		{

			// calculate the distance between the ion in shared
			// memory and the current thread's ion
			dist.x = d_posIon[IDcrntIon].x - sharedPos[h].x;
			dist.y = d_posIon[IDcrntIon].y - sharedPos[h].y;
			dist.z = d_posIon[IDcrntIon].z - sharedPos[h].z;

			// calculate the distance squared
			distSquared = dist.x*dist.x + dist.y*dist.y + dist.z*dist.z;

			// calculate the hard distance
			hardDist = __fsqrt_rn(distSquared);

			// calculate the soft distance
			softDist = __fsqrt_rn(distSquared + *d_SOFT_RAD_SQRD);

			// calculate a scaler intermediat
			linForce = *d_ION_ION_ACC_MULT*(1 + (hardDist**d_INV_DEBYE))
				*__expf(-hardDist**d_INV_DEBYE) / (softDist*softDist*softDist);

			// add the acceleration to the current ion's acceleration
			d_accIon[IDcrntIon].x += linForce*dist.x;
			d_accIon[IDcrntIon].y += linForce*dist.y;
			d_accIon[IDcrntIon].z += linForce*dist.z;
		} // end loop over ion in tile

		// wait for all trheads to finish calculations
		__syncthreads();
	} // end loop over tiles

}