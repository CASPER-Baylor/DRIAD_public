/*
* Project: IonWake
* File Type: function library header
* File Name: IonWake_101_bounds.h
*
* Created: 6/13/2017
* Last Modified: 8/26/2017
*
* Description:
*	Functions for handeling ions that leave the simulation region
*
* Functions:
*	replaceOutOfBoundsIons()
*	init()
*
* Includes:
*	replaceOutOfBoundsIons()
*		cuda_runtime.h
*		device_launch_parameters.h
*		curand_kernel.h
*	init()
*		cuda_runtime.h
*		device_launch_parameters.h
*		curand_kernel.h
*
*/

#ifndef IONWAKE_101_BOUNDS
#define IONWAKE_101_BOUNDS

	/* 
	* Required By:
	*	replaceOutOfBoundsIons()
	*	init()
	* For:
	*	CUDA
	*/
	#include "cuda_runtime.h"

	/*
	* Required By:
	*	replaceOutOfBoundsIons()
	*	init()
	* For:
	*	CUDA
	*/
	#include "device_launch_parameters.h"

	/*
	* Required By:
	*	replaceOutOfBoundsIons()
	*	init()
	* For:
	*	curand
	*/
	#include <curand_kernel.h>

	/*
	* Name: replaceOutOfBoundsIons
	*
	* Editors
	*	Dustin Sanford
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
		curandState_t* statesBlock, float* const d_RAD_SIM_SQRD, float* const d_RAD_SIM, 
		unsigned int* const d_NUM_ION, unsigned int* const d_NUM_DUST, 
		float3* d_posDust, float* const d_RAD_DUST_SQRD);

	/*
	* Name: init

	*
	* Editors
	*	Dustin Sanford
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
	__global__ void init(unsigned int, curandState_t*);

#endif // IONWAKE_101_BOUNDS