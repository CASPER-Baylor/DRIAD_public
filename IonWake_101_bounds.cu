/*
* Project: IonWake
* File Type: function library implemtation
* File Name: IonWake_101_bounds.cu
*
* Created: 6/20/2017
* Last Modified: 8/26/2017
*
* Description:
*	Functions for handeling ions that leave the simulation region
*
* Functions:
*	replaceOutOfBoundsIons()
*	init()
*
*/

// header file
#include "IonWake_101_bounds.h"

/*
* Name: replaceOutOfBoundsIons
* Created: 6/20/2017
* last edit: 8/26/2017
*
* Editors
*	Name: Dustin Sanford
*	Contact: Dustin_Sanford@baylor.edu
*	last edit: 8/26/2017
*
* Description:
*	Checks if an ion has left the simulation radius.
*	If the ion has left the simulation then it is given
*	a new random position on the surface of the upper 
*	hemisphere of the simulation shpere. The ion is then given
*	a new random velocity with the x and y components ranging 
*	from -150 m/s to 150 m/s and the x component ranging from
*	0 m/s to 300 m/s.
*
* Input:
*	d_posIon: ion positions
*	d_velIon: ion velocities
*	statesThread: a set of random states for use with curand.
*		There are as many states as the maximum number of threads
*		per block
*	statesBlock: a set of random states for use with curand.
*		There are as many states as the maximum number of blocks
*	d_RAD_SIM_SQRD: simulation radius squared
*	d_RAD_SIM: simulation radius
*	d_NUM_ION: the number of ions
*	d_NUM_DUST: the number of dust particles
*	d_posDust: the positions of the dust particles
*	d_RAD_DUST_SQRD: dust radius squared
*
* Output (void):
*	d_posIon: out of bounds ions are given a new position
*		on the surface of the upper hemisphere of the 
*		simulation.
*	d_velIon: out of bounds ions are given a new random velocity 
*		with the x and y components ranging from -150 m/s to 150 m/s 
*		and the x component ranging from 0 m/s to 300 m/s.
*
* Asumptions:
*	There are at least as many random states as threads per block
*		number of blocks for statesThread and statesBlock respectivly 
*	All numerical inputs are real numbers 
*
* Includes:
*	cuda_runtime.h
*	device_launch_parameters.h
*	curand_kernel.h
*
*/
__global__ void replaceOutOfBoundsIons
	(float3* d_posIon, float3* d_velIon, curandState_t* statesThread, 
	 curandState_t* statesBlock, float* const d_RAD_SIM_SQRD, 
	 float* const d_RAD_SIM, unsigned int* const d_NUM_ION, 
	 unsigned int* const d_NUM_DUST, float3* d_posDust, 
	 float* const d_RAD_DUST_SQRD)
{
	// distance
	float dist;

	// used in generating random values
	int numSections = 10000;

	// thread ID 
	int IDion = threadIdx.x + blockDim.x * blockIdx.x;

	// intermediat values used when generating 
	// random values
	int newVal;
	float newVal2;
	float theta;
	float phi;

	// check if the thread ID is in the 
	// bounds of the number of ions
	if (IDion <= *d_NUM_ION)
	{
		// position of the current ion
		float3 posCrntIon = d_posIon[IDion];

		// a random offset value
		int offset;
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
		

		// If the ion is out of bounds reset its poition 
		// and velocity 
		if (outOfBounds)
		{

			/*************************
				  Reset Position
			*************************/

			// random value that is the same for each thread ID within
			// each block
			offset = curand(&statesThread[threadIdx.x]);
			// add the offset to a random value that is the same for each 
			// thread block. Then change the range of the number to 
			// between 0 and numSections.
			newVal = (curand(&statesBlock[blockIdx.x]) + offset) % numSections;
			// change the range of the number to -numsections to +numsections
			newVal2 = (newVal * 2) - numSections;
			// change the range of the number to -PI to PI
			theta = (3.14159 * newVal2) / numSections;

			// random value that is the same for each thread ID within
			// each block
			offset = curand(&statesThread[threadIdx.x]);
			// add the offset to a random value that is the same for each 
			// thread block. Then change the range of the number to 
			// between 0 and numSections.
			newVal = (curand(&statesBlock[blockIdx.x]) + offset) % numSections;
			// change the range of the number to -PI/2 to 0
			phi = -(3.14159 * newVal) / numSections;

			// set the ion position to a random position on the 
			// the upper hemisphere of the simulation sphere
			posCrntIon.x = *d_RAD_SIM * sinf(theta) * cosf(phi);
			posCrntIon.y = *d_RAD_SIM * sinf(theta) * sinf(phi);
			posCrntIon.z = *d_RAD_SIM * cosf(theta);
	
			/*************************
				  Reset Velocity
			*************************/

			// random value that is the same for each thread ID within
			// each block
			offset = curand(&statesThread[threadIdx.x]);
			// add the offset to a random value that is the same for each 
			// thread block. Then change the range of the number to 
			// between 0 and numSections.
			newVal = (curand(&statesBlock[blockIdx.x]) + offset) % numSections;
			// change the range of the number to -numsections to +numsections
			newVal2 = (newVal * 2) - numSections;
			// change the range of the number to -300 to 300
			d_velIon[IDion].x = 300 * newVal2 / numSections;

			// random value that is the same for each thread ID within
			// each block
			offset = curand(&statesThread[threadIdx.x]);
			// add the offset to a random value that is the same for each 
			// thread block. Then change the range of the number to 
			// between 0 and numSections.
			newVal = (curand(&statesBlock[blockIdx.x]) + offset) % numSections;
			// change the range of the number to -300 to 0
			d_velIon[IDion].y =  (-300 * newVal2 / numSections);

			// random value that is the same for each thread ID within
			// each block
			offset = curand(&statesThread[threadIdx.x]);
			// add the offset to a random value that is the same for each 
			// thread block. Then change the range of the number to 
			// between 0 and numSections.
			newVal = (curand(&statesBlock[blockIdx.x]) + offset) % numSections;
			// change the range of the number to -numsections to +numsections
			newVal2 = (newVal * 2) - numSections;
			// change the range of the number to -300 to 300
			d_velIon[IDion].z = 300 * newVal2 / numSections;
		}	

		// save the ion position
		d_posIon[IDion] = posCrntIon;
	}
}

/*
* Name: init
* Created: 8/26/2017
* last edit: 8/26/2017
*
* Editors
*	Name: Dustin Sanford
*	Contact: Dustin_Sanford@baylor.edu
*	last edit: 8/26/2017
*
* Description:
*	Initializes an array of random states for use
*	with curand.
*
* Input:
*	seed: a random unsigned int used to seed
*		the random states generator 
*	states: an array to save the random states to
*
* Output (void):
*	states: sates is populated with random states
*
* Asumptions:
*	The seed is diferent for each call
*	Each block has one thread
*	The are many blocks as the length of the 
*		states array
*
* Includes:
*	cuda_runtime.h
*	device_launch_parameters.h
*	curand_kernel.h
*
*/

__global__ void init(unsigned int seed, curandState_t* states) 
{
	// initialize the states
	curand_init(seed, // seed for the random states generator
		blockIdx.x, // number of random states to generate
		1,          // number of random states that is advanced for each
					// subsequent random state in states
		&states[blockIdx.x] // array to save the states to
	);
}