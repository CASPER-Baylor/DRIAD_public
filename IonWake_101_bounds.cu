
#include "IonWake_101_bounds.h"

/*
* Name: replaceOutOfBoundsIons
* Created: 6/20/2017
* last edit: 7/19/2017
*
* Editors
*	Name: Dustin Sanford
*	Contact: Dustin_Sanford@baylor.edu
*	last edit: 7/19/2017
*
* Description:
*
* Input:
*
* Output:
*
* Data Abstraction:
*
* Asumptions:
*
* Includes:
*
*/
__global__ void replaceOutOfBoundsIons(float3* d_posIon, float3* d_velIon, 
	curandState_t* statesThread, curandState_t* statesBlock, 
	float* const d_RAD_SIM_SQRD, float* const d_RAD_SIM, unsigned int* const d_NUM_ION,
	unsigned int* const d_NUM_DUST, float3* d_posDust, float* const d_RAD_DUST_SQRD)
{
	// holds distances 
	float dist;

	// used in generating random values
	int numSections = 10000;

	// thread ID 
	int IDion = threadIdx.x + blockDim.x * blockIdx.x;

	// intermediat values used when generating 
	// random values
	int newVal;
	float newVal2;

	// check if 
	if (IDion <= *d_NUM_ION)
	{
		// position of the current ion
		float3 posCrntIon = d_posIon[IDion];

		// a random offset value
		int offset;
		// if the position of the current ion has been reset
		bool reset = false;
		// if the current ion is out of bounds
		bool outOfBounds = false;
		// temporary distance holders
		float deltaX, deltaY, deltaZ;

		// distance from the center of the simulation
		dist = posCrntIon.x * posCrntIon.x +
			posCrntIon.y * posCrntIon.y +
			posCrntIon.z * posCrntIon.z;
		
		// check if the ion is out of the simulation buble
		if (dist > *d_RAD_SIM_SQRD)
		{
			outOfBounds = true;
		}

		
		// loop over all of the dust particles
		for (int i = 0; i < *d_NUM_DUST; i++)
		{
			// x, y, and z distances between the current
			// ion and dust particle
			deltaX = posCrntIon.x - d_posDust[i].x;
			deltaY = posCrntIon.y - d_posDust[i].y;
			deltaZ = posCrntIon.z - d_posDust[i].z;

			// the squared distance between the current ion
			// and dust particle
			dist = deltaX * deltaX +
				   deltaY * deltaY +
				   deltaZ * deltaZ;

			// check if the dust particle and ion have colided
			if (dist < *d_RAD_DUST_SQRD)
			{
				outOfBounds = true;
			}
		}
		

		// reset the position untill it is in bounds
		if (outOfBounds)
		{
			// random value that is the same for each thread ID within
			// each block
			offset = curand(&statesThread[threadIdx.x]);
			// add the offset to a random value that is the same for each 
			// thread block. Then change the range of the number to 
			// between 0 and numSections.
			newVal = (curand(&statesBlock[blockIdx.x]) + offset) % numSections;
			// change the range of the number to -numsections to +numsections
			newVal2 = (newVal * 2) - numSections;
			// change the range of the number to -d_RAD_SIM to d_RAD_SIM
			posCrntIon.x = *d_RAD_SIM * newVal2 / numSections;

			// random value that is the same for each thread ID within
			// each block
			offset = curand(&statesThread[threadIdx.x]);
			// add the offset to a random value that is the same for each 
			// thread block. Then change the range of the number to 
			// between 0 and numSections.
			newVal = (curand(&statesBlock[blockIdx.x]) + offset) % numSections;
			// change the range of the number to -numsections to +numsections
			newVal2 = (newVal * 2) - numSections;
			// change the range of the number to -d_RAD_SIM to d_RAD_SIM
			posCrntIon.y = *d_RAD_SIM * newVal2 / numSections;

			// random value that is the same for each thread ID within
			// each block
			offset = curand(&statesThread[threadIdx.x]);
			// add the offset to a random value that is the same for each 
			// thread block. Then change the range of the number to 
			// between 0 and numSections.
			newVal = (curand(&statesBlock[blockIdx.x]) + offset) % numSections;
			// change the range of the number to -numsections to +numsections
			newVal2 = (newVal * 2) - numSections;
			// change the range of the number to -d_RAD_SIM to d_RAD_SIM
			posCrntIon.z = *d_RAD_SIM * newVal2 / numSections;

			// set the position reset flag to true
			reset = true;
		}

		// check if the position has been reset
		if(reset)
		{
			// random value that is the same for each thread ID within
			// each block
			offset = curand(&statesThread[threadIdx.x]);
			// add the offset to a random value that is the same for each 
			// thread block. Then change the range of the number to 
			// between 0 and numSections.
			newVal = (curand(&statesBlock[blockIdx.x]) + offset) % numSections;
			// change the range of the number to -d_RAD_SIM/50 to d_RAD_SIM/50
			d_velIon[IDion].x = *d_RAD_SIM * newVal / (numSections * 50);

			// set the x and y velocities to 0
			d_velIon[IDion].y = 0;
			d_velIon[IDion].z = 0;
		}	

		// save the ion position
		d_posIon[IDion] = posCrntIon;
	}
}


// this GPU kernel function is used to initialize the random states 
__global__ void init(unsigned int seed, curandState_t* states) 
{
	// we have to initialize the state 
	curand_init(seed,      // the seed can be the same for each core, here we pass the time in from the CPU 
		blockIdx.x, // the sequence number should be different for each core (unless you want all
					//  cores to get the same sequence of numbers for some reason - use thread id! 
		1,          // the offset is how much extra we advance in the sequence for each call, can be 0 
		&states[blockIdx.x]);
}