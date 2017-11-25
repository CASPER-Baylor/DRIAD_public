/*
* Project: IonWake
* File Type: function library header
* File Name: IonWake_101_bounds.h
*
* Created: 6/13/2017
* Last Modified: 10/22/2017
*
* Description:
*	Functions for handling ions that ions have an illegal position.
*	Such as outside of the simulation region or inside a dust particle.
*
* Functions:
*	checkIonSphereBounds()
*	checkIonDustBounds()
*	injectIonSpherePiel()
*	resetIonBounds()
*	initInjectIonPiel()
*	init()
*
* Includes:
*	checkIonSphereBounds()
*		device_launch_parameters.h
*		curand_kernel.h
*	checkIonDustBounds()
*		device_launch_parameters.h
*		curand_kernel.h
*	injectIonSpherePiel()
*		cuda_runtime.h
*		device_launch_parameters.h
*		curand_kernel.h
*	resetIonBounds()
*		device_launch_parameters.h
*		curand_kernel.h
*	init()
*		cuda_runtime.h
*		device_launch_parameters.h
*		curand_kernel.h
*	initInjectIonPiel()
*		device_launch_parameters.h
*		curand_kernel.h
*		<cstdio>
*		<fstream>
*		<string>
* 	invertFind()
*		device_launch_parameters.h
*		curand_kernel.h		
*
*/

#ifndef IONWAKE_101_BOUNDS
#define IONWAKE_101_BOUNDS

	/* 
	* Required By:
	*	checkIonSphereBounds()
	*	checkIonDustBounds()
	*	injectIonSpherePiel()
	*	resetIonBounds()
	*	initInjectIonPiel()
	* 	invertFind()
	*	init()
	* For:
	*	CUDA
	*/
	#include "cuda_runtime.h"

	/*
	* Required By:
	*	checkIonSphereBounds()
	*	checkIonDustBounds()
	*	injectIonSpherePiel()
	*	resetIonBounds()
	*	initInjectIonPiel()
	* 	invertFind()
	*	init()
	* For:
	*	CUDA
	*/
	#include "device_launch_parameters.h"

	/*
	* Required By:
	*	injectIonSpherePiel()
	*	init()
	* For:
	*	curand
	*/
	#include <curand_kernel.h>
		
	/*
	* Required By:
	*	initInjectIonPiel()
	* 
	* For:
	* 	sqrt()
	*/
	#include <cmath>

	/*
	* Required By:
	*	initInjectIonPiel()
	* 
	* For:
	* 	fprintf()
	*	stderr
	*/
	#include <cstdio>
	
	/*
	* Required By:
	*	initInjectIonPiel()
	* 
	* For:
	* 	fstream
	*/
	#include <fstream>
	
	/*
	* Required By:
	*	initInjectIonPiel()
	* 
	* For:
	* 	std::string
	*/
	#include <string>
	
	/*
	* Name: checkIonSphereBounds
	*
	* Editors
	*	Dustin Sanford
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
	* Includes:
	*	cuda_runtime.h
	*	device_launch_parameters.h
	*
	*/
	__global__ void checkIonSphereBounds(
			float3* const, 
			int*, 
			float* const);
	
    /*
    * checkIonDustBounds
    *
    * Editors
    *	Dustin Sanford
    *
    * Description:
    *	checks if an ion is within a dust particle 
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
	__global__ void checkIonDustBounds(		
		float3* const, 
		int*,
		float* const,
		int* const,
		float3* const);
	
	/*
    * injectIonSpherePiel
    *
    * Editors
    *	Dustin Sanford
    *   Lorin Matthews
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
    *		position according  to Piel 2017
    *	d_velIon: each ion that is out of bounds is given a new 
    *		position according to Piel 2017
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
	__global__ void injectIonPiel(
			float3*, 
			float3*, 
			float3*,
			curandState_t* const, 
			float* const, 
			int* const,
			float* const,
			float* const,
			float* const,
			int* const,
			int* const,
			float* const,
			float* const,
			float* const,
			float* const,
			float* const,
			float* const,
			float* const); 
			
	/*
	* Name: resetIonBounds
	*
	* Editors
	*	Dustin Sanford
	*
	* Description:
	*	Resets d_boundsIon to all 0
	*
	* Input:
	*	d_boundsIon: a flag for out of bounds ions
	*
	* Output (void):
	*	d_boundsIon: all indecies reset to 0
	*
	* Asumptions:
	*
	* Includes:
	*	cuda_runtime.h
	*	device_launch_parameters.h
	*
	*/
	__global__ void resetIonBounds(int*);

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
	__global__ void init(unsigned int, curandState_t*);
	

	/*
	* Name: initInjectIonPiel
	*
	* Editors
	*	Lorin Matthews
	*	Dustin Sanford
	*
	* Description:
	*	Initializes matrices for randomly selecting location on spherical 
	*	boundary given shifted Maxwellian distribution in the velocities.
	*
	* Input:
	*	d_posIon: ion positions
	*	d_velIon: ion velocities
	*	statesThread: a set of random states with at least as many
	*		random states as there are threads per block
	*	statesBock: a set of random states with at least as many 
	*		random states as there are blocks
	*	d_RAD_SIM: radius of the simulation sphere
	*	d_boundsIon: a flag for ions that are out of bounds
	*	d_GCOM: a matrix used to insert ions 
	*	d_QCOM: a matrix used to insert ions
	*	d_NUM_DIV_QTH: the length of d_QCOM
	*	d_NUM_DIV_VEL: the number of division in d_GCOM allong one axis
	*	d_DRIFT_VEL_ION: the drift velocity of the ions (MACH * SOUND_SPEED)
	*	d_TEMP_ION: the temperature of the ions
	*	d_PI: PI
	*
	* Output (void):
	*	Qcom: cumulative distribution in cosine angles Qth
	*	Gcom: cumulative distribution of velocities 
	*	Vcom: spread of radial velocities 
	*
	* Assumptions:
	*	All inputs are real values 
	*	Ions are flowing in z-direction 
	*
	* Includes:
	*	cuda_runtime.h
	*	device_launch_parameters.h
	*	<cstio>
	*	<fstream>
	*	<string>
	*
	*/
	void initInjectIonPiel(
		const int,
		const int,
		const float,
		const float,
		const float,
		const float,
		const float,
		const float,
		const float,
		float*,
		float*,
		float*,
		const bool,
		std::ostream&);
		
	/*
	* Name: invertFind
	* Created: 6/22/2017
	* last edit: 10/22/2017
	*
	* Editors
	*	Name: Dustin Sanford
	*	Contact: Dustin_Sanford@baylor.edu
	*	last edit: 10/22/2017
	*
	*	Name: Lorin Matthews
	*	Contact: Lorin_Matthews@baylor.edu
	*	last edit:6/22/2017
	*
	* Description:
	*	finds the floating point "index" of the value y in
	*	in the passed in matrix
	* 
	* Input:
	*	mat: that matrix to search through
	*	sizeMat: the number of entries in mat
	*	y: the value to search for in mat
	*
	* Output (float):
	*	The floating point "index" of the value y in the passed in matrix 
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
	__device__ float invertFind(float* const, int, float);
	
#endif // IONWAKE_101_BOUNDS