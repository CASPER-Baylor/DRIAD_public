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
*	pos: ion positions 
*	posDust: dust positions
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
	float3*,
	float3*,
	float*,
	const float*,
        const float*,
	const int*,
	const float*,
	int* const,
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
*   d_momIonDust
*	d_NUM_ION
*	d_SOFT_RAD_SQRD 
*	d_ION_DUST_ACC_MULT 
*	d_chargeDust
*
* Output (void):
*   pos: updated positions of ions
*	vel: updated velocities of ions
*	boundsIon: updated boundary crossings
*   momIonDust: momentum transfer from ions to dust
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
/*
* Name: checkIonSphereBounds_101_dev
* Created: 3/17/2018
*
* Editors
*	Name: Lorin Matthews
*	Contact: Lorin_Matthews@baylor.edu
*
* Description:
*	Checks if an ion has left the simulation sphere
*
* Input:
*	d_posIion: ion positions
*	d_boundsIon: a flag for if an ion is out of bounds
*	d_RAD_SIM_SQRD: the simulation radius squared
*
* Output (void):
*	d_boundsIon: set to -1 for ions that are outside of the 
*		simulation sphere.
*
* Assumptions:
*	The simulation region is a sphere with (0,0,0) at its center 
*   The number of ions is a multiple of the block size
*   the flag -1 is unique value for the ion bounds flag
*
* Includes:
*	cuda_runtime.h
*	device_launch_parameters.h
*
*/

__device__ void checkIonSphereBounds_101_dev(
      float3* const, 
		int*,
		const float*);
		
/*
* Name: checkIonCylinderBounds_101_dev
*
* Created: 3/17/2018
*
* Editors
*	Name: Lorin Matthews
*	Contact: Lorin_Matthews@baylor.edu
*	last edit: 11/18/2017
*
* Description:
*	Checks if an ion has left the simulation cylinder
*
* Input:
*	d_posIion: ion positions
*	d_boundsIon: a flag for if an ion is out of bounds
*	d_RAD_CYL_SQRD: the simulation radius squared
*	d_HT_CYL: the (half)height of the cylinder
*
* Output (void):
*	d_boundsIon: set to -1 for ions that are outside of the 
*		simulation sphere.
*
* Assumptions:
*	The simulation region is a cylinder with (0,0,0) at its center 
*   The number of ions is a multiple of the block size
*   the flag -1 is unique value for the ion bounds flag
*
* Includes:
*	cuda_runtime.h
*	device_launch_parameters.h
*
*/
__device__ void checkIonCylinderBounds_101_dev(
       float3* const, 
		int*,
		const float*,
		const float*);
			
/*
* Name: checkIonDustBounds_100_dev
* Created: 3/17/2018
* Edited: 4/30/2019 to add ion momentum transfer to dust
*
* Editors
*	Name: Lorin_Matthews
*
* Description:
*	checks if an ion is within  a dust particle 
*
* Input:
*	d_posIon: the ion positions
*	d_velIon: the ion vels
*	d_boundsIon: a flag for if an ion position is out of bounds
*	d_RAD_DUST_SQRD: the radius of the dust particles squared
*	d_NUM_DUST: the number of dust particles 
*	d_posDust: the dust particle positions
*
* Output (void):
*	d_boundsIon: set to the index of the dust particle the ion is
*		in if the ion is in a dust particle.
*	d_momIonDust: momentum (velocity) transferred to dust
*
* Assumptions:
*	All dust particles have the same radius 
*   The dust particles and ions are on the same coordinate axis
*   The number of ions is a multiple of the block size
*
*/
__device__ void checkIonDustBounds_100_dev(
		float3* const, 
		float3* const, 
		int*,
		const float*,
		const int*,
		float3* const,
		float3*);
/*
* Name: calcIonDustAcc_102_dev
* Created: 3/20/2018
*
* Editors
*	Name: Lorin Matthews
*	Contact: Lorin_Matthews@baylor.edu
*	last edit: 3/20/2018
*
* Description:
*	Calculates the ion accelerations due to ion-dust interactions 
*
* Input:
*	d_posIon: the positions of the ions
*	d_accIon: the accelerations of the ions
*	d_posDust: the dust particle positions
*	d_NUM_ION: the number of ions
*	d_NUM_DUST: the number of dust particles
*	d_SOFT_RAD_SQRD: the squared softening radius squared
*	d_ION_DUST_ACC_MULT: a constant multiplier for the ion-dust interaction
*   d_chargeDust: the charge on the dust particles 
*
* Output (void):
*	d_accIon: the acceleration due to all the dust particles
*		is added to the initial ion acceleration
*
* Assumptions:
*	All inputs are real values
*	All ions and dust particles have the parameters specified in the creation 
*		of the d_ION_ION_ACC_MULT value
*   The potential due to the dust particle is a bare coulomb potential
*   The number of ions is a multiple of the block size
*
* Includes:
*	cuda_runtime.h
*	device_launch_parameters.h
*
*/
__device__ void calcIonDustAcc_102_dev(
		float3*, 
		float3*, 
        float3*,
		const int*,
        const int*, 
		const float*, 
		const float*, 
		const float*);
		
#endif // IONWAKE_100
