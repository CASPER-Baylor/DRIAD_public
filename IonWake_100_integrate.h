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
*	drift_100()
*	kick_100()
*   select_100()
*
* Includes:
*	kick()
*	drift()
*   select()
*		cuda_runtime.h
*		device_launch_parameters.h
*
*/

#ifndef IONWAKE_100
#define IONWAKE_100

	/*
	* Required By:
	*	drift, kick_100()
	* For:
	*	CUDA
	*/
	#include "cuda_runtime.h"

	/*
	* Required By:
	*	drift, kick_100()
	* For:
	*	CUDA
	*/
	#include "device_launch_parameters.h"

        /* Include boundary checking and IonDustAcc calculation */
	#include "IonWake_101_bounds.h"
	#include "IonWake_102_ionAcc.h"

	/*
	*
	* Name:  kick_100
	* Editors
	*	Dustin Sanford, Lorin Matthews
	*
	* Description:
	*	Performs a leapfrog integration: kick for one dt
	*
	* Input:
	*	vel: velocities
	*	acc: accelerations
	*	d_TIME_STEP: half of a time step
	*
	* Output (void):
	*	vel: updated velocity from the integration
	*	acc: reset to zero
	*
	* Assumptions:
	*	All inputs are real values
	*
	* Includes:
	*	cuda_runtime.h
	*	device_launch_parameters.h
	*/
	__global__ void kick_100
           (float3*, 
	    float3*,
            const float*);
	/*
	*
	* Name:  drift_100
	* Editors
	*	Dustin Sanford, Lorin Matthews
	*
	* Description:
	*	Performs a leapfrog integration: dirft for one-half dt
	*
	* Input:
	*	pos:positions
	*	vel: velocities
	*	d_HALF_TIME_STEP: half of a time step
	*
	* Output (void):
	*	pos: updated positions from the integration
	*
	* Assumptions:
	*	All inputs are real values
	*
	* Includes:
	*	cuda_runtime.h
	*	device_launch_parameters.h
	*/
	__global__ void drift_100
           (float3*, 
	    float3*,
            const float*);
			
/*
* Name: select_100
* Created: 3/6/2018
* last edit: 3/6/2018
*
* Editors
*	Name: Lorin Matthews
*	Contact: Lorin_Matthews@baylor.edu
*	last edit: 3/07/2018
*
* Description:
*	Select timestep division for adaptive time step
*
* Input:
*	vel: ion velocities
*   minDistDust: distance to closest dust grain
*   d_RAD_DUST: radius of dust grains
*	d_TIME_STEP: time step
*   d_MAX_DEPTH: maximum divisions of time step
*
* Output (void):
*	m: the number of times timestep is divided by factor of 2
*   timeStepFactor: 2^(m-1)
*
* Assumptions:
*	All inputs are real values.
*
* Includes:
*	cuda_runtime.h
*	device_launch_parameters.h
*
*/
__global__ void select_100
       (float3*, 
	float*,
	const float*,
        const float*,
	const int*,
	int*,
	int*);
	
/*
* Name: KDK_100
* Created: 3/14/2017
* last edit: 3/14/2018
*
* Editors
*	Name: Lorin Matthews
*	Contact: Lorin_Matthews@baylor.edu
*
* Description:
*	Performs KDK 2^m times, using the kick just for the ion-dust 
*   dust acceleration.  'm' is a divisor for the time step so that
*   ions will be accurately captured by dust grains.
*
* Input:
*	pos: positions of ions
*	vel: velocities of ions
*	acc: accelerations of ions
*   d_m: depth for time step divisions
*	d_tsFactor: divisor for time step
*	boundsIon -- flags for boundary crossings
*	d_TIME_STEP: time step
*   d_RAD_SIM_SQRD -or- d_RAD_CYL_SQRT: simulation bounds	
*   (empty) -or- d_HT_CYL: simulation bounds
*	d_GEOMETRY (0=spherical or 1=cylindrical)
*	d_RAD_DUST_SQRD
*	d_NUM_DUST
*   d_posDust
*	d_NUM_ION
*	d_SOFT_RAD_SQRD 
*	d_ION_DUST_ACC_MULT 
*	d_chargeDust
*
* Output (void):
*   pos: updated positions of ions
*	vel: updated velocities of ions
*	boundsIon: updated boundary crossings
*
* Assumptions:
*	All inputs are real values
*
* Includes:
*	cuda_runtime.h
*	device_launch_parameters.h
*
*/
__global__ void KDK_100
    (float3*, 
	 float3*,
	 float3*,
	 const int*,
	 const int*,
	 int*,
	 const float*,
	 const int,
	 const float*,
	 const float*,
	 const float*,
	 const int*,
	 float3*,
	 const int*,
	 const float*,
	 const float*,
	 const float*);
	 
	 /*
* Name: kick_dev
* Created: 3/14/2017
* last edit: 3/14/2018
*
* Editors
*	Name: Lorin Matthews
*	Contact: Lorin_Matthews@baylor.edu
*
* Description:
*	Kick for half timestep in a leapfrog integration
*
* Input:
*	vel: velocity
*	acc: acceleration
*	ts: time step
*
* Output (void):
*	vel: updated velocity from the integration
*
* Assumptions:
*	All inputs are real values
*
* Includes:
*	cuda_runtime.h
*	device_launch_parameters.h
*
*/
__device__ void kick_dev
       (float3*, 
		float3*,
        float);

/*
* Name: drift_dev
* Created: 3/14/2017
* last edit: 3/14/2018
*
* Editors
*	Name: Lorin Matthews
*	Contact: Lorin_Matthews@baylor.edu
*
* Description:
*	Drift for whole  timestep in a leapfrog integration
*
* Input:
*	pos: position
*	vel: velocitie
*	timestep: time step
*
* Output (void):
*	pos: updated position from the integration
*
* Assumptions:
*	All inputs are real values.
*
* Includes:
*	cuda_runtime.h
*	device_launch_parameters.h
*
*/
__device__ void drift_dev
       (float3*, 
        float3*, 
        float)	; 
		
#endif // IONWAKE_100
