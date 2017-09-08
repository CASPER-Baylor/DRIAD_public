/*
* Project: IonWake
* File Type: function library implemtation
* File Name: IonWake_102_ionIonAcc.cu
*
* Created: 6/13/2017
* Last Modified: 8/28/2017
*
* Description:
*	Includes functions for handeling ion-ion accelerations 
*
* Functions:
*	calcIonIonForces()
*
*/

// header file
#include "IonWake_102_ionIonAcc.h"

/*
* Name: calcIonIonForces
* Created: 6/13/2017
* last edit: 8/28/2017
*
* Editors
*	Name: Dustin Sanford
*	Contact: Dustin_Sanford@baylor.edu
*	last edit: 8/28/2017
*
* Description:
*	Calculates the accelerations due to ion-ion 
*	interactions modled as Yakawa particles
*
* Input:
*	d_posIon: the positions of the ions
*	d_accIon: the accelerations of the ions
*	d_NUM_ION: the number of ions
*	d_SOFT_RAD_SQRD: the squared softening radius
*	d_ION_ION_ACC_MULT: a constant multiplier for the yakawa interaction
*	d_INV_DEBYE: the inverse of the debye
*
* Output (void):
*	d_accIon: the acceleration due to all of the other ions
*		is added to the initial ion acceleration
*
* Asumptions:
*	All inputs are real values
*	All ions have the parameters specified in the creation of the 
*		d_ION_ION_ACC_MULT value
*
* Includes:
*	cuda_runtime.h
*	device_launch_parameters.h
*
*/
__global__ void calcIonIonForces
	(float3* d_posIon, float3* d_accIon, unsigned int * const d_NUM_ION,
	float * const d_SOFT_RAD_SQRD, float * const d_ION_ION_ACC_MULT,
	float * const d_INV_DEBYE)
{

	// index of the current ion
	int IDcrntIon = blockIdx.x * blockDim.x + threadIdx.x;

	// allocate variables
	float3 dist;
	float3 accCrntIon = { 0,0,0 };
	float distSquared;
	float hardDist;
	float softDist;
	float linForce;
	int tileThreadID;

	// allocate shared memory
	extern __shared__ float3 sharedPos[];

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
		sharedPos[threadIdx.x].x = d_posIon[tileThreadID].x;
		sharedPos[threadIdx.x].y = d_posIon[tileThreadID].y;
		sharedPos[threadIdx.x].z = d_posIon[tileThreadID].z;
		
		// wait for all threads to load the current position
		__syncthreads();

		// DEBUGING // 
		/*
		// PTX code ussed to access shared memory sizes
		// which are save to "ret"
		unsigned ret;
		asm volatile ("mov.u32 %0, %total_smem_size;" : "=r"(ret));
		*/

		// loop over all of the ions loaded in the tile
		for (int h = 0; h < blockDim.x; h++)
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
			accCrntIon.x += linForce * dist.x;
			accCrntIon.y += linForce * dist.y;
			accCrntIon.z += linForce * dist.z;
		} // end loop over ion in tile

		// wait for all threads to finish calculations
		__syncthreads();
	} // end loop over tiles

	// save to global memory
	d_accIon[IDcrntIon].x += accCrntIon.x;
	d_accIon[IDcrntIon].y += accCrntIon.y;
	d_accIon[IDcrntIon].z += accCrntIon.z;
}