/*
* Project: IonWake
* File Type: function library implementation
* File Name: IonWake_101_bounds.cu
*
* Created: 6/13/2017
* Last Modified: 11/18/2017
*
* Description:
*	Functions for handling the boundary conditions for the ion positions.
*   This includes both determining when an ion is out of bounds as well
*   as reinserting out of bounds ions back into the simulation.
*
* Functions:
*	checkIonSphereBounds_101()
*	checkIonCylinderBounds_101()
*	checkIonDustBounds_101()
*	checkIonSphereBounds_101_dev()
*	checkIonCylinderBounds_101_dev()
*	checkIonDustBounds_101_dev()
*	injectIonSphere_101()
*	injectIonCylinder_101
*	resetIonBounds_101()
*	initInjectIonSphere_101()
*	initInjectIonCylinder_101()
*   	invertFind_101()
*	init_101()
*
*/

// header file
#include "IonWake_101_bounds.h"

/*
* Name: checkIonSphereBounds_101
* Created: 10/2/2017
* last edit: 11/14/2017
*
* Editors
*	Name: Dustin Sanford
*	Contact: Dustin_Sanford@baylor.edu
*	last edit: 11/14/2017
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

__global__ void checkIonSphereBounds_101
      (float3* const d_posIon, 
		int* d_boundsIon,
		float* const d_RAD_SIM_SQRD){
	
	// distance
	float dist;

	// thread ID 
	int IDion = threadIdx.x + blockDim.x * blockIdx.x;
	
    // Only check ions which are in bounds
	if (d_boundsIon[IDion] ==0){
		// position of the current ion
		float3 posCrntIon = d_posIon[IDion];

		// distance from the center of the simulation
		dist = posCrntIon.x * posCrntIon.x +
			posCrntIon.y * posCrntIon.y +
			posCrntIon.z * posCrntIon.z;
		
		// check if the ion is out of the simulation sphere
		if (dist > *d_RAD_SIM_SQRD)
		{
			// flag the ion as out of the simulation sphere
			d_boundsIon[IDion] = -1;
		}
	}
}



/*
* Name: checkIonCylinderBounds_101
* This is really checkIonCylinderBounds_101
*
* Created: 11/18/2017
* last edit: 11/18/2017
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
__global__ void checkIonCylinderBounds_101
       (float3* const d_posIon, 
		int* d_boundsIon,
		float* const d_RAD_CYL_SQRD,
		float* const d_HT_CYL){
	
	// distance
	float dist;
	float distz;

	// thread ID 
	int IDion = threadIdx.x + blockDim.x * blockIdx.x;

	// Only check ions which are in bounds
	if (d_boundsIon[IDion] ==0){
		// position of the current ion
		float3 posCrntIon = d_posIon[IDion];

		// radial distance from the center of the cylinder
		dist = posCrntIon.x * posCrntIon.x +
			posCrntIon.y * posCrntIon.y ;

		// height from the center of the cylinder
		distz = abs(posCrntIon.z);
		
		// check if the ion is out of the simulation cylinder
		if (dist > *d_RAD_CYL_SQRD || distz > *d_HT_CYL)
		{
			// flag the ion as out of the simulation sphere
			d_boundsIon[IDion] = -1;
		}
	}
}

/*
* Name: checkIonDustBounds_101
* Created: 10/2/2017
* last edit: 11/14/2017
*
* Editors
*	Name: Dustin Sanford
*	Contact: Dustin_Sanford@baylor.edu
*	last edit: 11/14/2017
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
* Includes:
*	cuda_runtime.h
*	device_launch_parameters.h
*
*/
__global__ void checkIonDustBounds_101(
	float3* const d_posIon, 
	int* d_boundsIon,
	float* const d_RAD_DUST_SQRD,
	int* const d_NUM_DUST,
	float3* const d_posDust) {
	
	// distance
	float dist;

	// thread ID 
	int IDion = threadIdx.x + blockDim.x * blockIdx.x;

	// Only check ions which are in bounds
	if (d_boundsIon[IDion] == 0) {
		// position of the current ion
		float3 posCrntIon = d_posIon[IDion];

		// temporary distance holders
		float deltaX, deltaY, deltaZ;
		
		// loop over all of the dust particles
		for (int i = 0; i < *d_NUM_DUST; i++) {
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

			// check if the dust particle and ion have collided
			if (dist <= *d_RAD_DUST_SQRD) {
				// flag which dust particle the ion is in
				d_boundsIon[IDion] = (i + 1);
			}
		}
	}
}


/*
* Name: injectIonSphere_101
* Created: 10/2/2017
* last edit: 11/14/2017
*
* Editors
*	Name: Dustin Sanford
*	Contact: Dustin_Sanford@baylor.edu
*	last edit: 11/14/2017
*
*   Name: Lorin Matthews
*   Contact: Lorin_Matthews@baylor.edu
*   last edit: 10/16/2017
*
* Description:
*	Injects ions into the simulation sphere 
*	as described in Piel 2017 
*
* Input:
*	d_posIon: ion positions
*	d_velIon: ion velocities
*   d_accIon: ion accelerations
*	randStates: a set of random states with at least as many
*		random states as there are threads
*	d_RAD_SIM: radius of the simulation sphere
*	d_boundsIon: a flag for ions that are out of bounds
*	d_GCOM: a matrix used to insert ions 
*	d_QCOM: a matrix used to insert ions
*   d_VCOM: a matrix used to insert ions 
*	d_NUM_DIV_QTH: the length of d_QCOM
*	d_NUM_DIV_VEL: the number of division in d_GCOM allong one axis
*   d_SOUND_SPEED: the sound speed of the plasma
*   d_TEMP_ION: the ion temperature 
*   d_PI: pi
*   d_TEMP_ELC: the electron temperature 
*   d_MACH: the mach number 
*   d_MASS_SINGLE_ION: the mass of a single ion
*	d_BOLTZMANN: the boltzmann constant 
*
* Output (void):
*	d_posIon: each ion that is out of bounds is given a new 
*		position according to Piel 2017
*	d_velIon: each ion that is out of bounds is given a new 
*		velocity according to Piel 2017
*   d_accIon: is reset to 0
*
* Assumptions:
*	The assumptions are the same as in Piel 2017
*   The number of ions is a multiple of the block size 
*
* Includes:
*	cuda_runtime.h
*	device_launch_parameters.h
*	curand_kernel.h
*
*/
__global__ void injectIonSphere_101(
		float3* d_posIon, 
		float3* d_velIon,
		float3* d_accIon,		
		curandState_t* const randStates, 
		float* const d_RAD_SIM, 
		int* const d_boundsIon,
		float* const d_GCOM,
		float* const d_QCOM,
		float* const d_VCOM,
		int* const d_NUM_DIV_QTH,
		int* const d_NUM_DIV_VEL,
		float* const d_SOUND_SPEED,
		float* const d_TEMP_ION,
		float* const d_PI,
		float* const d_TEMP_ELC,
		float* const d_MACH,
		float* const d_MASS_SINGLE_ION,
		float* const d_BOLTZMANN,
		int xac){
	
	// thread ID 
	int IDion = threadIdx.x + blockDim.x * blockIdx.x;
		
	// check if the ion is out of bounds 
	if (d_boundsIon[IDion] != 0){
	
		float randNum,
                floatQIndex,
                partQIndex,
                upperFloatGIndex,
                lowerFloatGIndex,
                radVel,
                part_radVel,
                phi,
                cosPhi,
                sinPhi,
                sinTheta,
                thetaVel,
                phiVel,
                cosTheta;
			  
		int tempIndex,
			tempIndex1;
	
		float velScale = __fsqrt_rn( 3.0 * (*d_BOLTZMANN) * (*d_TEMP_ION) 
                            / *d_MASS_SINGLE_ION);
                        
		float driftVelIon = (*d_SOUND_SPEED) * (*d_MACH); 
        
		float normDriftVel = driftVelIon / velScale;
		
		// get a random number from 0 to 1
		randNum = curand_uniform(&randStates[IDion]);
		
		// find the floating point index of randNum in d_Qcom 
		floatQIndex = invertFind_101(d_QCOM, *d_NUM_DIV_QTH, randNum);
		
		// get the floating point part of floatQIndex
		partQIndex = floatQIndex - static_cast<int>(floatQIndex);
		
		// get a random number from 0 to 1
		randNum = curand_uniform(&randStates[IDion]);
		
		// Pick normal velocity from cumulative G.
		tempIndex = static_cast<int>(floatQIndex) * *d_NUM_DIV_VEL;
		lowerFloatGIndex = invertFind_101(
				&d_GCOM[tempIndex],
				*d_NUM_DIV_VEL,
				randNum);
				
		tempIndex = static_cast<int>(floatQIndex + 1) * *d_NUM_DIV_VEL;
		upperFloatGIndex = invertFind_101(
				&d_GCOM[tempIndex],
				*d_NUM_DIV_VEL,
				randNum);
		
		// interpolate between upperFloatGIndex and lowerFloatGIndex to get 
        // a normalized radial velocity that ranges from 0 to d_NUM_DIV_VEL
		radVel = (partQIndex * upperFloatGIndex) + 
                 ( 1.0 - partQIndex ) * lowerFloatGIndex;
		
        // integer part of radVel 
		tempIndex = static_cast<int>(radVel); 
        // fractional part of radvel
		part_radVel = radVel - tempIndex; 
        
        tempIndex1 = tempIndex + 1;
		
        // interpolate the value of radVel from Vcom 
        radVel = part_radVel * d_VCOM[tempIndex1] + 
                 (1.0-part_radVel) * d_VCOM[tempIndex];
		
		
		// cos(theta), where theta is the angle of the velocity 
		// opposite to the opposite to the radius
		cosTheta = 1.0 - (2.0 * floatQIndex)/(*d_NUM_DIV_QTH - 1);
		
		// sin(theta)
		sinTheta = __fsqrt_rn( 1.0 - cosTheta * cosTheta );
		
		// get a random number from a normal distribution
		randNum = curand_normal(&randStates[IDion]);

		thetaVel = randNum - sinTheta * normDriftVel;
		
		// get a random number from a normal distribution
		randNum = curand_normal(&randStates[IDion]);
		
		phiVel = randNum;
		
		// get a random number from 0 to 1
		randNum = curand_uniform(&randStates[IDion]);

		// get phi
		phi = randNum * 2.0 * *d_PI;
		cosPhi = cosf(phi);
		sinPhi = sinf(phi);
		
		// convert the velocity from spherical to cartesian 
      	d_velIon[IDion].z = -(radVel*cosTheta - thetaVel*sinTheta)*velScale;
      	d_velIon[IDion].y = -((radVel*sinTheta + thetaVel*cosTheta)*sinPhi + 
                            phiVel*cosPhi)* velScale;
      	d_velIon[IDion].x = -((radVel*sinTheta + thetaVel*cosTheta)*cosPhi - 
                            phiVel*sinPhi)* velScale;
		
		// convert the position from spherical to cartesian and multiply by 
        // rfrac so that the position is within the simulation boundary
      	float rfrac = 0.9999;
      	d_posIon[IDion].z = *d_RAD_SIM * rfrac * cosTheta;
      	d_posIon[IDion].y = *d_RAD_SIM * rfrac * sinTheta * sinPhi;
      	d_posIon[IDion].x = *d_RAD_SIM * rfrac * sinTheta * cosPhi;
		
		// polarity switching
		if(xac ==1) {
			d_posIon[IDion].z *= -1.0;
			d_velIon[IDion].z *= -1.0;
		}

		// reset the acceleration
		d_accIon[IDion].x = 0;
		d_accIon[IDion].y = 0;
		d_accIon[IDion].z = 0;
	}
}

/*
* Name: injectIonCylinder_101
* Created: 11/19/2017
* last edit: 11/19/2017
*
* Editors
*   Name: Lorin Matthews
*   Contact: Lorin_Matthews@baylor.edu
*   last edit: 11/19/2017
*
* Description:
*	Injects ions into the simulation cylinder
*	based on the three surfaces (top, side, bottom) 
*	using the angles at theta = 0, 90, 180
* 	from the Hutchinson algorithm, described in Piel 2017 
*
* Input:
*	d_posIon: ion positions
*	d_velIon: ion velocities
*   	d_accIon: ion accelerations
*	randStates: a set of random states with at least as many
*		random states as there are threads
*	d_RAD_CYL: radius of the simulation cylinder
*	d_HT_CYL: (half) height of the cylinder
*	d_boundsIon: a flag for ions that are out of bounds
*	d_GCOM: a matrix used to insert ions 
*	d_QCOM: a matrix used to insert ions
*   	d_VCOM: a matrix used to insert ions 
*	d_NUM_DIV_QTH: the length of d_QCOM
*	d_NUM_DIV_VEL: the number of division in d_GCOM allong one axis
*   d_SOUND_SPEED: the sound speed of the plasma
*   d_TEMP_ION: the ion temperature 
*   d_PI: pi
*   d_TEMP_ELC: the electron temperature 
*   d_MACH: the mach number 
*   d_MASS_SINGLE_ION: the mass of a single ion
*	d_BOLTZMANN: the boltzmann constant 
*
* Output (void):
*	d_posIon: each ion that is out of bounds is given a new 
*		position assuming flowing ions
*	d_velIon: each ion that is out of bounds is given a new 
*		velocity from a shifted Maxwellian
*   	d_accIon: is reset to 0
*
* Assumptions:
*	The assumptions based on those developed in Hutchinson
*	(Sphere in a flowing plasma).  The cylinder only has sides
*	that face thatea = 0, 90, or 180 degrees.
*   The number of ions is a multiple of the block size 
*
* Includes:
*	cuda_runtime.h
*	device_launch_parameters.h
*	curand_kernel.h
*
*/
__global__ void injectIonCylinder_101(
	float3* d_posIon, 
	float3* d_velIon,
	float3* d_accIon,		
	curandState_t* const randStates, 
	float* const d_RAD_CYL, 
	float* const d_HT_CYL, 
	int* const d_boundsIon,
	float* const d_GCOM,
	float* const d_QCOM,
	float* const d_VCOM,
	int* const d_NUM_DIV_QTH,
	int* const d_NUM_DIV_VEL,
	float* const d_SOUND_SPEED,
	float* const d_TEMP_ION,
	float* const d_PI,
	float* const d_TEMP_ELC,
	float* const d_MACH,
	float* const d_MASS_SINGLE_ION,
	float* const d_BOLTZMANN,
	int xac){
	
	// thread ID 
	int IDion = threadIdx.x + blockDim.x * blockIdx.x;
		
	// check if the ion is out of bounds 
	if (d_boundsIon[IDion] != 0) {
	
		float randNum,
        	QIndex,
            lowerFloatGIndex,
            radVel,
            part_radVel,
            phi,
            cosPhi,
            sinPhi,
            sinTheta,
            thetaVel,
            phiVel,
            cosTheta;
			  
		int tempIndex,
			tempIndex1;
	
		float velScale = __fsqrt_rn( 3.0 * (*d_BOLTZMANN) * (*d_TEMP_ION) 
                            / *d_MASS_SINGLE_ION);
                        
		float driftVelIon = (*d_SOUND_SPEED) * (*d_MACH); 
        
		float normDriftVel = driftVelIon / velScale;
		
		// get a random number from 0 to 1
		randNum = curand_uniform(&randStates[IDion]);
		
		// find the index of randNum in d_Qcom 
		if (randNum < d_QCOM[0]) {
			QIndex = 0;
		} else if(randNum < d_QCOM[1]) {
			QIndex = 1;
		} else {
			QIndex = 2;
		}
		
		// get a random number from 0 to 1
		randNum = curand_uniform(&randStates[IDion]);
		
		// Pick normal velocity from cumulative G.
		tempIndex = static_cast<int>(QIndex) * *d_NUM_DIV_VEL;
		lowerFloatGIndex = invertFind_101(
			&d_GCOM[tempIndex],
			*d_NUM_DIV_VEL,
			randNum);
				
		/*
		* tempIndex = static_cast<int>(QIndex + 1) * *d_NUM_DIV_VEL;
		* upperFloatGIndex = invertFind_101(
		* &d_GCOM[tempIndex],
		* *d_NUM_DIV_VEL,
		* randNum);
		*		
		* // interpolate between upperFloatGIndex and lowerFloatGIndex to get 
		* // a normalized radial velocity that ranges from 0 to d_NUM_DIV_VEL
		* radVel = (partQIndex * upperFloatGIndex) + 
		* ( 1 - partQIndex ) * lowerFloatGIndex;
		*/

		radVel = lowerFloatGIndex; //This may not be right
		
        // integer part of radVel 
		tempIndex = static_cast<int>(radVel); 
        
		// fractional part of radvel
		part_radVel = radVel - tempIndex; 
        
        tempIndex1 = tempIndex + 1;
		
        // interpolate the value of radVel from Vcom 
        radVel = part_radVel * d_VCOM[tempIndex1] + 
        	(1.0-part_radVel) * d_VCOM[tempIndex];
		
		
		// cos(theta), where theta is the angle of the velocity 
		// from the z-axis
		cosTheta = 1.0 - (2.0 * QIndex)/(*d_NUM_DIV_QTH - 1);
		
		// sin(theta)
		sinTheta = __fsqrt_rn( 1.0 - cosTheta * cosTheta );

		
		// get a random number from a normal distribution
		randNum = curand_normal(&randStates[IDion]);

		thetaVel = randNum - sinTheta * normDriftVel;
		
		// get a random number from a normal distribution
		randNum = curand_normal(&randStates[IDion]);
		
		phiVel = randNum;
		
		// get a random number from 0 to 1
		randNum = curand_uniform(&randStates[IDion]);

		// get phi
		phi = randNum * 2.0 * *d_PI;
		cosPhi = cosf(phi);
		sinPhi = sinf(phi);
		
		// convert the velocity from spherical to cartesian 
      	d_velIon[IDion].z = -(radVel*cosTheta - thetaVel*sinTheta)*velScale;
      	d_velIon[IDion].y = -((radVel*sinTheta + thetaVel*cosTheta)*sinPhi + 
        	phiVel*cosPhi)* velScale;
      	d_velIon[IDion].x = -((radVel*sinTheta + thetaVel*cosTheta)*cosPhi - 
        	phiVel*sinPhi)* velScale;
		
		// Select a random location on the top or side 
        // Multiply by rfrac so that the position is within the simulation boundary
      	float rfrac = 0.9999;
	
		if(QIndex == 1) {
			//location is on the side, so choose a random z 
			// get a random number from 0 to 1
			randNum = curand_uniform(&randStates[IDion]);
      			d_posIon[IDion].z =  rfrac * (randNum * 2.0 - 1.0) * *d_HT_CYL * rfrac;
      			d_posIon[IDion].y = *d_RAD_CYL * rfrac * sinTheta * sinPhi;
      			d_posIon[IDion].x = *d_RAD_CYL * rfrac * sinTheta * cosPhi;
		} else {
			//location is on the top or bottom, so choose random x and y
			d_posIon[IDion].z = rfrac * cosTheta * *d_HT_CYL * rfrac;

			float dist = 1.1  * *d_RAD_CYL * *d_RAD_CYL;
			while (dist > *d_RAD_CYL * *d_RAD_CYL) {
				randNum = curand_uniform(&randStates[IDion]);
				d_posIon[IDion].x = (randNum*2.0-1.0) * *d_RAD_CYL;
				randNum = curand_uniform(&randStates[IDion]);
				d_posIon[IDion].y = (randNum*2.0-1.0) * *d_RAD_CYL;

				// See if this position is inside the cylinder
				dist = d_posIon[IDion].x * d_posIon[IDion].x + 
				   d_posIon[IDion].y * d_posIon[IDion].y;
			}

		}
		
		// polarity switching
		if(xac == 1) {
			d_posIon[IDion].z *= -1.0;
			d_velIon[IDion].z *= -1.0;
		}	
		
		// reset the acceleration
		d_accIon[IDion].x = 0;
		d_accIon[IDion].y = 0;
		d_accIon[IDion].z = 0;
	}
}

/*
* Name: resetIonBounds_101
* Created: 10/2/2017
* last edit: 11/14/2017
*
* Editors
*	Name: Dustin Sanford
*	Contact: Dustin_Sanford@baylor.edu
*	last edit: 11/14/2017
*
* Description:
*	Resets d_boundsIon to all 0
*
* Input:
*	d_boundsIon: a flag for out of bounds ions
*
* Output (void):
*	d_boundsIon: all indices are reset to 0
*
* Assumptions:
*	0 is not used as an out of bounds flag
*
* Includes:
*	cuda_runtime.h
*	device_launch_parameters.h
*
*/
__global__ void resetIonBounds_101(int* d_boundsIon){
	
	// thread ID 
	int IDion = threadIdx.x + blockDim.x * blockIdx.x;
	
	// reset d_boundsIon
	d_boundsIon[IDion] = 0;
}

/*
* Name: init_101
* Created: 8/26/2017
* last edit: 11/14/2017
*
* Editors
*	Name: Dustin Sanford
*	Contact: Dustin_Sanford@baylor.edu
*	last edit: 11/14/2017
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
* Assumptions:
*	The seed is different for each call
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

__global__ void init_101(unsigned int seed, curandState_t* states) 
{
	// initialize the states
	curand_init(seed, // seed for the random states generator
		blockIdx.x, // number of random states to generate
		1,          // number of random states that is advanced for each
					// subsequent random state in states
		&states[blockIdx.x] // array to save the states to
	);
}

/*
* Name: initInjectIonSphere_101
* Created: 9/21/2017
* last edit: 11/14/2017
*
* Editors
*	Name: Lorin Matthews
*	Contact: Lorin_Matthews@baylor.edu
*	last edit: 11/16/2017
*
*	Name: Dustin Sanford
*	Contact: Dustin_Sanford@baylor.edu
*	last edit: 11/14/2017
*
* Description:
*	Initializes matrices for randomly selecting location on spherical 
*	boundary given shifted Maxwellian distribution in the velocities.
*	Code has been translated from Hutchinson's SCEPTIC code written 
*	in fortran
* 
* Input:
*	d_GCOM: a matrix used to insert ions 
*	d_QCOM: a matrix used to insert ions
*       d_VCOM: a matrix used to insert ions 
*	NUM_DIV_QTH: the length of d_QCOM
*	NUM_DIV_VEL: the number of division in d_GCOM allong one axis
*   TEMP_ION: the ion temperature 
*   PI: pi
*   TEMP_ELC: the electron temperature 
*   MACH: the mach number 
*   MASS_SINGLE_ION: the mass of a single ion
*	BOLTZMANN: the boltzmann constant 
*   DRIFT_VEL_ION: the drift velocity of the ions 
*   fileName: an output file for debugging 
*
* Output (void):
*	d_QCOM: cumulative distribution in cosine angles Qth
*	d_GCOM: cumulative distribution of velocities 
*	d_VCOM: spread of radial velocities 
*
* Assumptions:
*	All inputs are real values 
*	Ions are flowing in z-direction 
*
* Includes:
*	cuda_runtime.h
*	device_launch_parameters.h
*	<cstdio>
*	<fstream>
*	<string>
*
*/

void initInjectIonSphere_101(
		const int NUM_DIV_QTH,
		const int NUM_DIV_VEL,
		const float TEMP_ELC,
		const float TEMP_ION,
		const float DRIFT_VEL_ION,
		const float MACH,
		const float MASS_SINGLE_ION,
		const float BOLTZMANN,
		const float PI,
		float* d_QCOM,
		float* d_VCOM,
		float* d_GCOM,
		const bool debugMode,
		std::ostream& fileName){ 
	
	// allocate memory to create the matrices on the host
	float* Qcom = new float[NUM_DIV_QTH];
	float* Vcom = new float[NUM_DIV_VEL];
	float* Gcom = new float[NUM_DIV_QTH * NUM_DIV_VEL];
	
	// useful variables 
	float const1 = 1/sqrt(2.0 * PI);
	float const2 = 1/sqrt(2.0);
	
	// normalized ion temperature 
	float normTempIon = 3.0 * BOLTZMANN * TEMP_ION / MASS_SINGLE_ION;
	
	// range of velocities (times (Ti/m_i)^1/2) permitted for injection
	float vspread = 5.0 +  (DRIFT_VEL_ION/ sqrt(normTempIon));
	
	// first term of Qcom
	Qcom[0] = 0;
	
	// variables used in the loop over the angles
	int dqp = 0;
	float Qth,
		  vdr,
		  dqn,
		  temp;

	// loop over the angles
	for (int i = 0; i < NUM_DIV_QTH; i++)
	{
		// Qth is the cosine angle of the ith interpolation position
		Qth = 1.0 - (2.0 * i / static_cast<float>(NUM_DIV_QTH - 1));

		// scale drift velocity to the ion temperature
		vdr = DRIFT_VEL_ION * Qth /sqrt(normTempIon);
		
		dqn = (const1 * exp((-0.5) * vdr * vdr) )
				+ ((0.5) * vdr * erfcf( (-1.0) * const2 * vdr));

		
		if (i > 0){
			Qcom[i] = Qcom[i-1] + dqp + dqn;
		}
		dqp = dqn;

		
		// at this angle, now do
		for (int j = 0; j < NUM_DIV_VEL; j++) {
			
		//construct cumulative distribution of radial velocity on mesh Vcom
			Vcom[j] = vspread  * j / (NUM_DIV_VEL - 1); 

			// temporary value
			temp = Vcom[j]-vdr;
			
			//cumulative distribution is Gcom
			Gcom[i * NUM_DIV_VEL + j] = 
				(
				dqn - 
				const1 * exp(-0.5 * temp * temp) - 
				 0.5 * vdr * erfcf( const2 * temp ) 
				) / dqn;
		}

		Gcom[i * NUM_DIV_VEL] = 0; 
		Gcom[i * NUM_DIV_VEL + (NUM_DIV_VEL - 1)] = 1;
	}
	
	
	for (int i = 1; i < NUM_DIV_QTH; i++){
		Qcom[i] = Qcom[i] / Qcom[NUM_DIV_QTH - 1];
	}	
	
	// variable to hold cuda status 
	cudaError_t cudaStatus;
	
	// copy Gcom to the device
	cudaStatus = cudaMemcpy(d_GCOM, Gcom,
		sizeof(float) * NUM_DIV_QTH * NUM_DIV_VEL, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed: d_GCOM\n");
	}
	
	// copy Vcom to the device
	cudaStatus = cudaMemcpy(d_VCOM, Vcom,
		sizeof(float) * NUM_DIV_VEL, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed: d_VCOM\n");
	}
	
	// copy Qcom to the device
	cudaStatus = cudaMemcpy(d_QCOM, Qcom,
		sizeof(float) * NUM_DIV_QTH, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed: d_QCOM\n");
	}
	
	if (debugMode){
		
		fileName << "--- Qcom ---" << std::endl;
		for (int i = 0; i < NUM_DIV_QTH; i++){
			fileName << Qcom[i] << ";..." << std::endl;		
		}
		fileName << std::endl;
		
		fileName << "--- Vcom ---" << std::endl;
		for (int i = 0; i < NUM_DIV_VEL; i++){
			fileName << Vcom[i] << std::endl;
		}
		fileName << std::endl;
		
		fileName << "--- Gcom ---" << std::endl;
		for (int i = 0; i < NUM_DIV_QTH; i++){
			for (int j = 0; j < NUM_DIV_VEL; j++){
				
				fileName << Gcom[i * NUM_DIV_VEL + j];
				
				if ((j+1) < NUM_DIV_VEL){
					fileName << ",";
				}
				
				fileName << "  ";
			}
			fileName << std::endl;
		}
	} 
	
	// free host memory
	delete[] Qcom;
	delete[] Vcom;
	delete[] Gcom;
	
}

/*
* Name: initInjectIonCylinder_101
* Created: 11/19/2017
* last edit: 11/19/2017
*
* Editors
*	Name: Lorin Matthews
*	Contact: Lorin_Matthews@baylor.edu
*	last edit: 11/19/2017
*
*
* Description:
*	Initializes matrices for randomly selecting location on cylindrical 
*	boundary given shifted Maxwellian distribution in the velocities.
*	Code based on Hutchinson's SCEPTIC code  for spherical boundary written 
*	in fortran
* 
* Input:
*	d_GCOM: a matrix used to insert ions 
*	d_QCOM: a matrix used to insert ions
*       d_VCOM: a matrix used to insert ions 
*	NUM_DIV_QTH: the length of d_QCOM
*	NUM_DIV_VEL: the number of division in d_GCOM allong one axis
*	RAD_CYL: the radius of the top
*	HT_CYL: (half) height of the cylinder
*   TEMP_ION: the ion temperature 
*   PI: pi
*   TEMP_ELC: the electron temperature 
*   MACH: the mach number 
*   MASS_SINGLE_ION: the mass of a single ion
*	BOLTZMANN: the boltzmann constant 
*   DRIFT_VEL_ION: the drift velocity of the ions 
*   fileName: an output file for debugging 
*
* Output (void):
*	d_QCOM: cumulative distribution in cosine angles Qth
*	d_GCOM: cumulative distribution of velocities 
*	d_VCOM: spread of radial velocities 
*
* Assumptions:
*	All inputs are real values 
*	Ions are flowing in z-direction 
*	Cylinder is along the z-axis
*	NUM_DIV_QTH = 3 for top, side, bottom
*
* Includes:
*	cuda_runtime.h
*	device_launch_parameters.h
*	<cstdio>
*	<fstream>
*	<string>
*
*/

void initInjectIonCylinder_101(
		const int NUM_DIV_QTH,
		const int NUM_DIV_VEL,
		const float RAD_CYL,
		const float HT_CYL,
		const float TEMP_ELC,
		const float TEMP_ION,
		const float DRIFT_VEL_ION,
		const float MACH,
		const float MASS_SINGLE_ION,
		const float BOLTZMANN,
		const float PI,
		float* d_QCOM,
		float* d_VCOM,
		float* d_GCOM,
		const bool debugMode,
		std::ostream& fileName){ 
	
	// allocate memory to create the matrices on the host
	float* Qcom = new float[NUM_DIV_QTH];
	float* Vcom = new float[NUM_DIV_VEL];
	float* Gcom = new float[NUM_DIV_QTH * NUM_DIV_VEL];
	
	// useful variables 
	float const1 = 1/sqrt(2.0 * PI);
	float const2 = 1/sqrt(2.0);
	
	// normalized ion temperature 
	float normTempIon = 3.0 * BOLTZMANN * TEMP_ION / MASS_SINGLE_ION;
	
	// range of velocities (times (Ti/m_i)^1/2) permitted for injection
	float vspread = 5.0 +  (DRIFT_VEL_ION/ sqrt(normTempIon));
	
	
	// variables used in the loop over the angles
	float Qth,
		  vdr,
		  dqn,
		  temp;
	float costheta [NUM_DIV_QTH];
	float area [NUM_DIV_QTH];

	//Three angles for top, side, bottom
	costheta[0] = 1.0;
	costheta[1] = 0.0;
	costheta[2] = -1.0;

	//Area of each side to weight the fluxes
	area[0] = PI * RAD_CYL * RAD_CYL;
	area[1] = 4.0 * PI * RAD_CYL * HT_CYL; //Since we are using half-height
	area[2] = PI * RAD_CYL * RAD_CYL;

	// loop over the angles
	for (int i = 0; i < NUM_DIV_QTH; i++)
	{
		// Qth is the cosine angle of the ith interpolation position
		Qth = costheta[i];

		// scale drift velocity to the ion temperature
		vdr = DRIFT_VEL_ION * Qth /sqrt(normTempIon);
		
		// dqn is really Gamma(infinity)
		dqn = (const1 * exp((-0.5) * vdr * vdr) )
				+ ((0.5) * vdr * erfcf( (-1.0) * const2 * vdr));

		
		if (i == 0){
		     Qcom[i] = dqn * area[i];
		     }
		else {
			Qcom[i] = Qcom[i-1] + dqn * area[i];
		}
	
		
		// at this angle, now do
		for (int j = 0; j < NUM_DIV_VEL; j++) {
			
		//construct cumulative distribution of radial velocity on mesh Vcom
			Vcom[j] = vspread  * j / (NUM_DIV_VEL - 1); 

			// temporary value
			temp = Vcom[j]-vdr;
			
			//cumulative distribution is Gcom
			Gcom[i * NUM_DIV_VEL + j] = 
				(
				dqn - 
				const1 * exp(-0.5 * temp * temp) - 
				 0.5 * vdr * erfcf( const2 * temp ) 
				) / dqn;
		}

		Gcom[i * NUM_DIV_VEL] = 0; 
		Gcom[i * NUM_DIV_VEL + (NUM_DIV_VEL - 1)] = 1;
	}
	
	
	for (int i = 0; i < NUM_DIV_QTH; i++){
		Qcom[i] = Qcom[i] / Qcom[NUM_DIV_QTH - 1];
	}	
	
	// variable to hold cuda status 
	cudaError_t cudaStatus;
	
	// copy Gcom to the device
	cudaStatus = cudaMemcpy(d_GCOM, Gcom,
		sizeof(float) * NUM_DIV_QTH * NUM_DIV_VEL, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed: d_GCOM\n");
	}
	
	// copy Vcom to the device
	cudaStatus = cudaMemcpy(d_VCOM, Vcom,
		sizeof(float) * NUM_DIV_VEL, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed: d_VCOM\n");
	}
	
	// copy Qcom to the device
	cudaStatus = cudaMemcpy(d_QCOM, Qcom,
		sizeof(float) * NUM_DIV_QTH, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed: d_QCOM\n");
	}
	
	if (debugMode){
		
		fileName << "--- Qcom ---" << std::endl;
		for (int i = 0; i < NUM_DIV_QTH; i++){
			fileName << Qcom[i] << ";..." << std::endl;		
		}
		fileName << std::endl;
		
		fileName << "--- Vcom ---" << std::endl;
		for (int i = 0; i < NUM_DIV_VEL; i++){
			fileName << Vcom[i] << std::endl;
		}
		fileName << std::endl;
		
		fileName << "--- Gcom ---" << std::endl;
		for (int i = 0; i < NUM_DIV_QTH; i++){
			for (int j = 0; j < NUM_DIV_VEL; j++){
				
				fileName << Gcom[i * NUM_DIV_VEL + j];
				
				if ((j+1) < NUM_DIV_VEL){
					fileName << ",";
				}
				
				fileName << "  ";
			}
			fileName << std::endl;
		}
	} 
	
	// free host memory
	delete[] Qcom;
	delete[] Vcom;
	delete[] Gcom;
	
}

/*
* Name: invertFind_101
* Created: 6/22/2017
* last edit: 11/14/2017
*
* Editors
*	Name: Dustin Sanford
*	Contact: Dustin_Sanford@baylor.edu
*	last edit: 11/14/2017
*
*	Name: Lorin Matthews
*	Contact: Lorin_Matthews@baylor.edu
*	last edit:6/22/2017
*
* Description:
*	finds the floating point "index" of the value y 
*	in the passed in matrix
* 
* Input:
*	mat: that matrix to search through
*	sizeMat: the number of entries in mat
*	y: the value to search for in mat
*
* Output (float):
*	floatIndex: the floating point "index" of the value y
*		in the passed in matrix 
*
* Assumptions:
*	mat[n] < mat[n + 1] 
*	mat[0] <= y <= mat[sizeMat]
*
* Includes:
*	cuda_runtime.h
*	device_launch_parameters.h
*
*/
__device__ float invertFind_101(float* const mat, int sizeMat, float y){

	int intIndex = 0;
	
	// find the index of the last value in mat that is 
	// less than y
	while (mat[intIndex + 1] < y && intIndex < sizeMat){
		intIndex++;
	}

	// get a floating point "index" by assuming the function 
	// represented by the entries in mat are locally linear between
	// indices  and interpolating between the values in mat
	// that y fell between
	float floatIndex = intIndex + 
		(y - mat[intIndex])/(mat[intIndex+1] - mat[intIndex]);

	return floatIndex;
}	
