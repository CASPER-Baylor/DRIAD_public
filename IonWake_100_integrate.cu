/*
* Project: IonWake
* File Type: function library implementation 
* File Name: IonWake_100_integrate.cu
*
* Created: 6/13/2017
* Last Modified: 10/22/2017
*
* Description:
*	Includes time step integrators
*
* Functions:
*	stepForward()
*
*/

// header file
#include "IonWake_100_integrate.h"

/*
* Name: stepForward
* Created: 6/13/2017
* last edit: 10/22/2017
*
* Editors
*	Name: Dustin Sanford
*	Contact: Dustin_Sanford@baylor.edu
*	last edit: 10/22/2017
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
* Assumptions:
*	All inputs are real values
*
* Includes:
*	cuda_runtime.h
*	device_launch_parameters.h
*
*/
__global__ void leapfrog
       (float3* pos, 
        float3* vel, 
        float3* acc, 
        const float* d_HALF_TIME_STEP)
{
	// thread ID
	int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	/* calculate velocity
	* v = v0 + (1/2) * dT * a
	*******************
	* v  - velocity
	* v0 - initial velocity
	* dT - time step
	* a  - acceleration
	*******************/
	vel[threadID].x += ((*d_HALF_TIME_STEP) * acc[threadID].x);
	vel[threadID].y += ((*d_HALF_TIME_STEP) * acc[threadID].y);
	vel[threadID].z += ((*d_HALF_TIME_STEP) * acc[threadID].z);

	/* calculate position
	* x = x0 + (1/2) * dT * v
	*******************
	* x  - position
	* x0 - initial position
	* dT - time step
	* v  - velocity
	*******************/
	pos[threadID].x += ((*d_HALF_TIME_STEP) * vel[threadID].x);
	pos[threadID].y += ((*d_HALF_TIME_STEP) * vel[threadID].y);
	pos[threadID].z += ((*d_HALF_TIME_STEP) * vel[threadID].z);

	// reset acceleration
	acc[threadID].x = 0;
	acc[threadID].y = 0;
	acc[threadID].z = 0;
}
