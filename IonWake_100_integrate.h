/*
* Project: IonWake
* File Type: function library header
* File Name: IonWake_100_integrate.h
*
* Created: 6/13/2017
* Last Modified: 8/26/2017
*
* Description:
*	Includes time step integrators
*
* Functions:
*	stepForward()
*
* Includes:
*	stepForward()
*		cuda_runtime.h
*		device_launch_parameters.h
*
*/

#ifndef IONWAKE_100
#define IONWAKE_100

	/*
	* Required By:
	*	stepForward()
	* For:
	*	CUDA
	*/
	#include "cuda_runtime.h"

	/*
	* Required By:
	*	stepForward()
	* For:
	*	CUDA
	*/
	#include "device_launch_parameters.h"

	/*
	* Name: stepForward
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
	*	vel: updated velocitiy from the integration
	*	acc: set to zero.
	*
	* Asumptions:
	*	All inputs are real values
	*
	* Includes:
	*	cuda_runtime.h
	*	device_launch_parameters.h
	*/
	__global__ void stepForward(float3*, float3*, float3*, float* const);

#endif // IONWAKE_100
