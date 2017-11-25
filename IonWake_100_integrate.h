/*
* Project: IonWake
* File Type: function library header
* File Name: IonWake_100_integrate.h
*
* Created: 6/13/2017
* Last Modified: 10/22/2017
*
* Description:
*	Includes time step integrators
*
* Functions:
*	leapfrog_100()
*
* Includes:
*	leapfrog()
*		cuda_runtime.h
*		device_launch_parameters.h
*
*/

#ifndef IONWAKE_100
#define IONWAKE_100

	/*
	* Required By:
	*	leapfrog_100()
	* For:
	*	CUDA
	*/
	#include "cuda_runtime.h"

	/*
	* Required By:
	*	leapfrog_100()
	* For:
	*	CUDA
	*/
	#include "device_launch_parameters.h"

	/*
	* Name: leapfrog_100
	*
	* Editors
	*	Dustin Sanford
	*
	* Description:
	*	Performs a leapfrog integration
	*
	* Input:
	*	pos: positions
	*	vel: velocities
	*	acc: accelerations
	*	d_HALF_TIME_STEP: half of a time step
	*
	* Output (void):
	*	pos: updated position from the integration
	*	vel: updated velocity from the integration
	*	acc: set to zero.
	*
	* Assumptions:
	*	All inputs are real values
	*
	* Includes:
	*	cuda_runtime.h
	*	device_launch_parameters.h
	*/
	__global__ void leapfrog_100
           (float3*, 
            float3*, 
            float3*, 
            const float*);

#endif // IONWAKE_100
