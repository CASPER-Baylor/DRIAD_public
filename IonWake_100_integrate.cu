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
*   d_MAX_DEPTH: maximum divisions of time step
*
* Output (void):
*	m: the number of times timestep is divided by factor of 2
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
       (float3* vel, 
	float* minDistDust,
	const float* d_RAD_DUST,
        const float* d_TIME_STEP,
	const int* d_MAX_DEPTH,
	int* m,
	int* tsFactor)
{
	// thread ID
	int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	
	// initialize variables
	float v2;
	float v;
	int mtemp;
	int tsf;

	/* calculate timestep depth
	* m = ceil(ln(30 * dT * v / (dist - dustRadius))/ln(2))
	*******************
	* dT - time step
	* v  - magnitude of velocity
	* dist - distance to closest dust particle
	* dustRadius  - radius of dust particles
	* 30 is a factor which initial tests showed to work well
	*******************/
	
	v2 = vel[threadID].x * vel[threadID].x + 
		vel[threadID].y * vel[threadID].y + 
		vel[threadID].z * vel[threadID].z; 
		
	v =__fsqrt_rn(v2);
	
	v2 = __logf(30 * *d_TIME_STEP * v /(minDistDust[threadID] - *d_RAD_DUST))
		/ __logf(2);
	mtemp = ceil(v2);
	
	if (mtemp < 0){
		mtemp = 0;
	}
	else if (mtemp > *d_MAX_DEPTH) {
		mtemp = *d_MAX_DEPTH;
	}
	
	tsf = 1;
	for(int i = 0; i < mtemp; i++)
	{
		tsf = tsf * 2;
	}
	
	m[threadID] = mtemp;
	tsFactor[threadID] = tsf;
}

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
    (float3* pos, 
	 float3* vel,
	 float3* acc,
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
	 const float* d_chargeDust)
	 {
		// thread ID
		int threadID = blockIdx.x * blockDim.x + threadIdx.x;
		
		//local variables
		int tsf = *d_tsFactor;
		float ts = *d_TIME_STEP * tsf;
		float half_ts = ts * 0.5;
	 	 
		// Kick for 1/2 a timestep to get started
		kick_dev(vel+threadID, acc+threadID, half_ts); 
	
	// now do Drift, check, calc_accels, Kick, for tsf = 2^(m-1) times
		while(d_boundsIon[threadID] ==0){
			for (int depth = 1; depth <= tsf; depth++){

			drift_dev(pos+threadID,vel+threadID,ts);
	
			//Check outside bounds
			if(GEOMETRY == 0) {
                    // check if any ions are outside of the simulation sphere
                    checkIonSphereBounds_101_dev 
                          (pos+threadID, d_boundsIon+threadID, d_bndry_sqrd);
                   }

            if(GEOMETRY == 1) {
                    // check if any ions are outside of the simulation cylinder
                    checkIonCylinderBounds_101_dev 
                          (pos+threadID, d_boundsIon+threadID, 
                           d_bndry_sqrd, d_HT_CYL);
					}
			
			// check if any ions are inside a dust particle 
			checkIonDustBounds_101_dev
                       (pos+threadID, d_boundsIon+threadID,
                        d_RAD_DUST_SQRD, d_NUM_DUST, d_posDust);
		// Calc IonDust accels
		// calculate the acceleration due to ion-dust interactions
				calcIonDustAcc_102_dev
                       (&pos[threadID], 
                        &acc[threadID],
                        *d_posDust,
                        *d_NUM_ION,
                        *d_NUM_DUST, 
                        *d_SOFT_RAD_SQRD, 
                        *d_ION_DUST_ACC_MULT, 
                        *d_chargeDust);
    
		// Kick with IonDust accels for deltat/2^(m-1)
			if(depth == tsf){
				// on last time step, do a half kick
				kick_dev(vel+threadID, acc+threadID, half_ts);
			}
			else {
				kick_dev(vel+threadID, acc+threadID, ts);
			}

		} //end for loop over depth
	 }// end while dust in bounds
		
		
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
