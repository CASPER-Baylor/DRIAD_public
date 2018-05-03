/*
* Project: IonWake
* File Type: function library implementation 
* File Name: IonWake_100_integrate.cu
*
* Created: 6/13/2017
* Last Modified: 3/20/2018
*
* Description:
*	Includes time step integrators
*
* Functions:
*	kick_100()
*	drift_100()
*	select_100()
*	KDK_100()
* Local Functions
*   drift_dev()
*   kick_dev()
*
*/

// header file
#include "IonWake_100_integrate.h"

/*
* Name: kick_100
* Created: 6/13/2017
* last edit: 01/24/2018
*
* Editors
*	Name: Dustin Sanford
*	Contact: Dustin_Sanford@baylor.edu
*	last edit: 10/22/2017
*	Name: Lorin Matthews
*	Contact: Lorin_Matthews@baylor.edu
*	last edit: 01/24/2018
*
* Description:
*	Kick for one timestep in a leapfrog integration
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
*
*/
__global__ void kick_100
       (float3* vel, 
		float3* acc,
        const float* d_TIME_STEP)
{
	// thread ID
	int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	/* calculate velocity
	* v = v0 + dT * a
	*******************
	* v  - velocity
	* v0 - initial velocity
	* dT - time step
	* a  - acceleration
	*******************/
	vel[threadID].x += ((*d_TIME_STEP) * acc[threadID].x);
	vel[threadID].y += ((*d_TIME_STEP) * acc[threadID].y);
	vel[threadID].z += ((*d_TIME_STEP) * acc[threadID].z);

	// reset acceleration
	acc[threadID].x = 0;
	acc[threadID].y = 0;
	acc[threadID].z = 0;
}


/*
* Name: drift_100
* Created: 6/13/2017
* last edit: 01/25/2018
*
* Editors
*	Name: Dustin Sanford
*	Contact: Dustin_Sanford@baylor.edu
*	last edit: 10/22/2017
*	Name: Lorin Matthews
*	Contact: Lorin_Matthews@baylor.edu
*	last edit: 01/25/2018
*
* Description:
*	Kick for whole  timestep in a leapfrog integration
*
* Input:
*	pos: positions
*	vel: velocities
*	d_HALF_TIME_STEP: half of a time step
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
__global__ void drift_100
       (float3* pos, 
        float3* vel, 
        const float* d_HALF_TIME_STEP)
{
	// thread ID
	int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	/* calculate position
	* x = x0 +  dT/2* v(n+1/2)
	*******************
	* x  - position
	* x0 - initial position
	* dT - time step
	* v  - velocity
	*******************/
	pos[threadID].x += ((*d_HALF_TIME_STEP) * vel[threadID].x);
	pos[threadID].y += ((*d_HALF_TIME_STEP) * vel[threadID].y);
	pos[threadID].z += ((*d_HALF_TIME_STEP) * vel[threadID].z);

}


/*
* Name: select_100
* Created: 3/6/2018
* last edit: 3/6/2018
*
* Editors
*	Name: Lorin Matthews
*	Contact: Lorin_Matthews@baylor.edu
*	last edit: 3/06/2018
*
* Description:
*	Select timestep division for adaptive time step
*
* Input:
*	vel: ion velocities
*   minDistDust: distance to closest dust grain
*   d_RAD_DUST: radius of dust grains
*	d_TIME_STEP: time step
*   d_M_FACTOR: constant used in determining time step division
*   d_MAX_DEPTH: maximum divisions of time step
*
* Output (void):
*	m: the nuspeedber of times timestep is divided by factor of 2
*   tsFactor: 2^(m-1)
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
    (float3* velIon, 
	 float* minDistDust,
	 const float* d_RAD_DUST,
     const float* d_TIME_STEP,
	 const int* d_MAX_DEPTH,
	 const float* d_M_FACTOR,
	 int* m,
	 int* timeStepFactor) {
	
	// thread ID
	int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	
	// initialize variables
	float v2;
	float speed;
	int mtemp;
	int tsf;

	//                Calculate Timestep Depth
	/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 *      m = ceil(ln( M * dT * v / abs(r - R))/ln(2))       *
	 * ------------------------------------------------------- *
	 * m  - timestep depth                                     *
	 * M  - d_M_FACTOR                                         *
	 * dT - time step                                          *
	 * v  - magnitude of velocity                              *
	 * r  - distance to closest dust particle                  *
	 * R  - radius of the dust particles                       *
	 *                                                         *
	 * 30 is a factor which initial tests showed to work well  *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
	// ion speed squared 
	v2 = velIon[threadID].x * velIon[threadID].x + 
		 velIon[threadID].y * velIon[threadID].y + 
		 velIon[threadID].z * velIon[threadID].z; 
	
	// ion speed
	speed = __fsqrt_rn(v2);
	
	// m = ceil(v2)
	// v2 is being used to as an intermediat step in calculating m 
	v2 = __logf(*d_M_FACTOR * *d_TIME_STEP * speed / 
		(minDistDust[threadID] - *d_RAD_DUST)) / __logf(2);

	// timestep depth
	mtemp = ceil(v2);
	
	// check that the timestep depth is within allowed bounds
	if (mtemp < 0){
		mtemp = 0;
	} else if (mtemp > *d_MAX_DEPTH) {
		mtemp = *d_MAX_DEPTH;
	}
	
	// calculate 2^(time step depth)
	tsf = 1;
	for(int i = 0; i < mtemp; i++) {
		tsf = tsf * 2;
	}
	
	// save to global memory
	m[threadID] = mtemp;
	timeStepFactor[threadID] = tsf;
}

/*
* Name: KDK_100
* Created: 3/14/2018
* last edit: 4/30/2018
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
    (float3* posIon, 
	 float3* velIon,
	 float3* ionDustAcc,
	 const int* m,
	 const int* d_tsFactor,
	 int* d_boundsIon,
	 const float* d_TIME_STEP,
	 const int GEOMETRY,
	 const float* d_bndry_sqrd,
	 const float* d_HT_CYL,
	 const float* d_RAD_DUST_SQRD,
	 const int* d_NUM_DUST,
	 float3* d_posDust,
	 const int* d_NUM_ION,
	 const float* d_SOFT_RAD_SQRD,
	 const float* d_ION_DUST_ACC_MULT,
	 const float* d_chargeDust) {

	// thread ID
	int threadID = blockIdx.x * blockDim.x + threadIdx.x;
		
	//local variables
	int timeStepFactor = *d_tsFactor;
	float timeStep = *d_TIME_STEP / timeStepFactor;
	float halfTimeStep = timeStep * 0.5;
	bool stopflag = false;
	 	 
	// Kick for 1/2 a timestep to get started
	kick_dev(velIon+threadID, ionDustAcc+threadID, halfTimeStep); 
		
	// now do Drift, check, calc_accels, Kick, for tsf = 2^(m-1) times
	int depth = 0;
	while(d_boundsIon[threadID] == 0 && depth <= timeStepFactor && !stopflag){
			
		depth++;
		drift_dev(posIon+threadID, velIon+threadID, timeStep);
	
		//Check outside bounds
		if(GEOMETRY == 0) {
            // check if any ions are outside of the simulation sphere
			checkIonSphereBounds_101_dev
            	(posIon+threadID, d_boundsIon+threadID, d_bndry_sqrd);
        } else if(GEOMETRY == 1) {
        	// check if any ions are outside of the simulation cylinder
            checkIonCylinderBounds_101_dev 
            	(posIon+threadID, 
				d_boundsIon+threadID, 
                d_bndry_sqrd, d_HT_CYL);
		}
			
		// check if any ions are inside a dust particle 
		checkIonDustBounds_101_dev
        	(posIon + threadID, 
			d_boundsIon + threadID,
            d_RAD_DUST_SQRD, 
			d_NUM_DUST, 
			d_posDust);
						
		if(d_boundsIon[threadID] ==0){
			// calculate the acceleration due to ion-dust interactions
			calcIonDustAcc_102_dev
            	(posIon + threadID, 
                ionDustAcc + threadID,
                d_posDust,
                d_NUM_ION,
                d_NUM_DUST, 
                d_SOFT_RAD_SQRD, 
                d_ION_DUST_ACC_MULT, 
                d_chargeDust);
    
			// Kick with IonDust accels for deltat/2^(m-1)
			if(depth == timeStepFactor){
				// on last time step, do a half kick
				kick_dev(velIon+threadID, ionDustAcc+threadID, halfTimeStep);
			} else {
				kick_dev(velIon+threadID, ionDustAcc+threadID, timeStep);
			}
		} else {
			stopflag = true;
		}
	}// end for loop over depth			
}
	 
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
       (float3* vel, 
		float3* acc,
        float timestep)
{
	// thread ID
	//int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	// calculate velocity
	// v = v0 + dT * a

	vel->x += timestep  * (acc->x); 
    vel->y += timestep  * (acc->y);
    vel->z += timestep  * (acc->z);


}

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
       (float3* pos, 
        float3* vel, 
        float timestep)
{
	// thread ID
	//int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	// calculate position
	// x = x0 +  dT* v(n+1/2)

	pos->x += timestep  * (vel->x); 
    pos->y += timestep  * (vel->y);
    pos->z += timestep  * (vel->z);
}
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

__device__ void checkIonSphereBounds_101_dev
      (float3* d_posIon, 
		int* d_boundsIon,
		const float* d_RAD_SIM_SQRD){
	
	// distance
	float dist;
	
    // Only check ions which are in bounds
	if (*d_boundsIon==0){

		// distance from the center of the simulation
		dist = d_posIon->x * d_posIon->x +
			d_posIon->y * d_posIon->y +
			d_posIon->z * d_posIon->z;
		
		// check if the ion is out of the simulation sphere
		if (dist > *d_RAD_SIM_SQRD)
		{
			// flag the ion as out of the simulation sphere
			*d_boundsIon = -1;
		}
	}
}


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
__device__ void checkIonCylinderBounds_101_dev
       (float3* d_posIon, 
		int* d_boundsIon,
		const float* d_RAD_CYL_SQRD,
		const float* d_HT_CYL){
	
	// distance
	float dist;
	float distz;


	// Only check ions which are in bounds
	if (*d_boundsIon ==0){

		// radial distance from the center of the cylinder
		dist = d_posIon->x * d_posIon->x +
			d_posIon->y * d_posIon->y ;

		// height from the center of the cylinder
		distz = abs(d_posIon->z);
		
		// check if the ion is out of the simulation cylinder
		if (dist > *d_RAD_CYL_SQRD || distz > *d_HT_CYL)
		{
			// flag the ion as out of the simulation sphere
			*d_boundsIon = -1;
		}
	}
}

/*
* Name: checkIonDustBounds_101_dev
* Created: 3/17/2018
*
* Editors
*	Name: Lorin_Matthews
*
* Description:
*	checks if an ion is within  a dust particle 
*
* Input:
*	d_posIon: the ion positions
*	d_boundsIon: a flag for if an ion position is out of bounds
*	d_RAD_DUST_SQRD: the radius of the dust particles squared
*	d_NUM_DUST: the number of dust particles 
*	d_posDust: the dust particle positions
*
* Output (void):
*	d_boundsIon: set to the index of the dust particle the ion is
*		in if the ion is in a dust particle.
*
* Assumptions:
*	All dust particles have the same radius 
*   The dust particles and ions are on the same coordinate axis
*   The number of ions is a multiple of the block size
*
*/
__device__ void checkIonDustBounds_101_dev(
		float3* d_posIon, 
		int* d_boundsIon,
		const float* d_RAD_DUST_SQRD,
		const int* d_NUM_DUST,
		float3* const d_posDust){
	
	// distance
	float dist;

	// Only check ions which are in bounds
	if (*d_boundsIon == 0){

		// temporary distance holders
		float deltaX, deltaY, deltaZ;
		
		// loop over all of the dust particles
		for (int i = 0; i < *d_NUM_DUST; i++)
		{
			// x, y, and z distances between the current
			// ion and dust particle
			deltaX = d_posIon->x - d_posDust[i].x;
			deltaY = d_posIon->y - d_posDust[i].y;
			deltaZ = d_posIon->z - d_posDust[i].z;

			// the squared distance between the current ion
			// and dust particle
			dist = deltaX * deltaX +
				deltaY * deltaY +
				deltaZ * deltaZ;

			// check if the dust particle and ion have collided
			if (dist < *d_RAD_DUST_SQRD)
			{
				// flag which dust particle the ion is in
				*d_boundsIon = (i + 1);
			}
		}
	}
}


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
		float3* d_posIon, 
		float3* d_accIon, 
        float3* d_posDust,
		const int* d_NUM_ION,
        const int* d_NUM_DUST, 
		const float* d_SOFT_RAD_SQRD, 
		const float* d_ION_DUST_ACC_MULT, 
		const float* d_chargeDust)
{

	// allocate variables
	float3 dist;
	float distSquared;
	float hardDist;
	float linForce;
        float softDistSqrd= 2.5e-13;

	//reset the acceleration
	d_accIon->x = 0;
	d_accIon->y = 0;
	d_accIon->z = 0;
	
	// loop over all of the dust particles
	for (int h = 0; h < *d_NUM_DUST; h++) {

			// calculate the distance between the ion in shared
			// memory and the current thread's ion
			dist.x = d_posIon->x - (d_posDust+h)->x;
			dist.y = d_posIon->y - (d_posDust+h)->y;
			dist.z = d_posIon->z - (d_posDust+h)->z;

			// calculate the distance squared
			distSquared = dist.x*dist.x + dist.y*dist.y + dist.z*dist.z;

			// calculate the hard distance
			hardDist = __fsqrt_rn(distSquared + softDistSqrd);

			// calculate a scaler intermediate
			linForce = *d_ION_DUST_ACC_MULT * d_chargeDust[h] / 
                        (hardDist*hardDist*hardDist);

			// add the acceleration to the current ion's acceleration
			d_accIon->x += linForce * dist.x;
			d_accIon->y += linForce * dist.y;
			d_accIon->z += linForce * dist.z;	

	} // end loop over dust

}
