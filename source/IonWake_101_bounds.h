/*
* Project: IonWake
* File Type: function library header
* File Name: IonWake_101_bounds.h
*
* Created: 6/13/2017
* Last Modified: 09/10/2020
*
* Description:
*	Functions for handling ions that have an illegal position.
*	Such as outside of the simulation region or inside a dust particle.
*
* Functions:
*	checkIonDustBounds_101()
*	injectIonSphere_101()
*	injectIonCylinder_101()
*	resetIonBounds_101()
*	initInjectIonSphere_101()
*	initInjectIonCylinder_101()
*   invertFind_101()
*	init_101()
*	boundaryEField_101()
*	tile_calculation()
*	pointPointPotential()
*
* Includes:
*	checkIonDustBounds_101()
*		device_launch_parameters.h
*		cuda_runtime.h
*	injectIonSphere_101()
*		cuda_runtime.h
*		device_launch_parameters.h
*		curand_kernel.h
*	injectIonCylinder_101()
*		cuda_runtime.h
*		device_launch_parameters.h
*		curand_kernel.h
*	resetIonBounds_101()
*       cuda_runtime.h
*		device_launch_parameters.h
*	init_101()
*		cuda_runtime.h
*		device_launch_parameters.h
*		curand_kernel.h
*	initInjectIonSphere_101()
*       cuda_runtime.h
*		device_launch_parameters.h
*		curand_kernel.h
*		<cstdio>
*		<fstream>
*		<string>
*	initInjectIonCylinder_101()
*       cuda_runtime.h
*		device_launch_parameters.h
*		curand_kernel.h
*		<cstdio>
*		<fstream>
*		<string>
* 	invertFind_101()
*		device_launch_parameters.h
*		curand_kernel.h		
*
*/

#ifndef IONWAKE_101_BOUNDS
#define IONWAKE_101_BOUNDS

	/* 
	* Required By:
	*	checkIonDustBounds_101()
	*	injectIonSphere_101()
	*	injectIonCylinder_101()
	*	resetIonBounds_101()
	* 	invertFind_101()
	*	init_101()
	* For:
	*	CUDA
	*/
	#include "cuda_runtime.h"

	/*
	* Required By:
	*	checkIonDustBounds_101()
	*	injectIonSphere_101()
	*	injectIonSphereCylinder_101()
	*	resetIonBounds_101()
	* 	invertFind_101()
	*	init_101()
	* For:
	*	CUDA
	*/
	#include "device_launch_parameters.h"

	/*
	* Required By:
	*	injectIonSphere_101()
	*	injectIonCylinder_101()
	*	init_101()
	* For:
	*	curand
	*/
	#include <curand_kernel.h>
		
	/*
	* Required By:
	*	initInjectIonSphere_101()
	*	initInjectIonCylinder_101()
	* For:
	* 	sqrt()
	*/
	#include <cmath>

	/*
	* Required By:
	*	initInjectIonSphere_101()
	*	initInjectIonCylinder_101()
	* For:
	* 	fprintf()
	*	stderr
	*/
	#include <cstdio>
	
	/*
	* Required By:
	*	initInjectIonSphere_101()
	*	initInjectIonCylinder_101()
	* For:
	* 	fstream
	*/
	#include <fstream>
	
	/*
	* Required By:
	*	initInjectIonSphere_101()
	*	initInjectIonCylinder_101()
	* For:
	* 	std::string
	*/
	#include <string>
	
    /*
    * checkIonDustBounds_101
    *
    * Editors
    *	Dustin Sanford
    *
    * Description:
    *	checks if an ion is within a dust particle 
    *
    * Input:
    *	d_posIon: the ion positions and charges
    *	d_boundsIon: a flag for if an ion position is out of bounds
    *	d_RAD_DUST_SQRD: the radius of the dust particles squared
    *	d_NUM_DUST: the number of dust particles 
    *	d_posDust: the dust particle positions and charges
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
		float4* const, 
		int*,
		float* const,
		int* const,
		float4* const);

			
/*
    * injectIonSphere_101
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
    *	d_posIon: ion positions and charges
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
    *	d_CHARGE_ION; charge on the superion
	*   xac: 0 or 1 for polarity switching of E field
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
	__global__ void injectIonSphere_101(
			float4*, 
			float4*, 
			float4*,
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
			float* const,
			float* const,
			int); 

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
*       Injects ions into the simulation cylinder
*       based on the three surfaces (top, side, bottom)
*       using the angles at theta = 0, 90, 180
*       from the Hutchinson algorithm, described in Piel 2017
*
* Input:
*       d_posIon: ion positions
*       d_velIon: ion velocities
*       d_accIon: ion accelerations
*       randStates: a set of random states with at least as many
*               random states as there are threads
*       d_RAD_CYL: radius of the simulation cylinder
*       d_HT_CYL: (half) height of the cylinder
*       d_boundsIon: a flag for ions that are out of bounds
*       d_GCOM: a matrix used to insert ions
*       d_QCOM: a matrix used to insert ions
*       d_VCOM: a matrix used to insert ions
*       d_NUM_DIV_QTH: the length of d_QCOM
*       d_NUM_DIV_VEL: the number of division in d_GCOM allong one axis
*   d_SOUND_SPEED: the sound speed of the plasma
*   d_TEMP_ION: the ion temperature
*   d_PI: pi
*   d_TEMP_ELC: the electron temperature
*   d_MACH: the mach number
*   d_MASS_SINGLE_ION: the mass of a single ion
*   d_BOLTZMANN: the boltzmann constant
*	d_CHARGE_ION: charge on the superion
*	plasma_counter: index of evolving plasma parameters
*	counter_part: fraction part of plasma timestep
*   xac: 0 or 1 for polarity switching of E field
*
* Output (void):
*       d_posIon: each ion that is out of bounds is given a new
*               position, assuming flowing ions, and charge
*       d_velIon: each ion that is out of bounds is given a new
*               velocity from a shifted Maxwellian
*       d_accIon: is reset to 0
*
* Assumptions:
*       The assumptions based on those developed in Hutchinson
*       (Sphere in a flowing plasma).  The cylinder only has sides
*       that face thatea = 0, 90, or 180 degrees.
*   The number of ions is a multiple of the block size
*
* Includes:
*       cuda_runtime.h
*       device_launch_parameters.h
*       curand_kernel.h
*
*/
__global__ void injectIonCylinder_101(
                float4* ,
                float4* ,
                float4* ,
                curandState_t* const ,
                float* const ,
                float* const ,
                int* const ,
                float* const ,
                float* const ,
                float* const ,
                int* const ,
                int* const ,
                float* const ,
                float* const ,
                float* const ,
                float* const ,
                float* const ,
                float* const ,
                float* const ,
                float* const ,
				int, float,
				int );	
		
	/*
	* Name: resetIonBounds_101
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
	__global__ void resetIonBounds_101(int*);

	/*
	* Name: init_101
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
	__global__ void init_101(unsigned int, curandState_t*);
	

	/*
	* Name: initInjectIonSphere_101
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
	void initInjectIonSphere_101(
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
* Name: initInjectIonCylinder_101
* Created: 11/19/2017
* last edit: 11/19/2017
*
* Editors
*       Name: Lorin Matthews
*       Contact: Lorin_Matthews@baylor.edu
*       last edit: 11/19/2017
*
*
* Description:
*       Initializes matrices for randomly selecting location on cylindrical
*       boundary given shifted Maxwellian distribution in the velocities.
*       Code based on Hutchinson's SCEPTIC code  for spherical boundary written
*       in fortran
*
* Input:
*       NUM_DIV_QTH: the length of d_QCOM
*       NUM_DIV_VEL: the number of division in d_GCOM allong one axis
*       TIME_EVOL: 0 if static plasma, otherwise number of conditions
*       RAD_CYL: the radius of the top
*       HT_CYL: (half) height of the cylinder
*   	evolTe: electron temperature
*   	evolTi: ion temperature
*   	evolVz: ion drift speed
*   	evolMach: ion Mach number
*   	MASS_SINGLE_ION: the mass of a single ion
*       BOLTZMANN: the boltzmann constant
*   	PI: pi
*       d_GCOM: a matrix used to insert ions
*       d_QCOM: a matrix used to insert ions
*       d_VCOM: a matrix used to insert ions
*       debugMode: 0 or 1 for debug output
*   	fileName: an output file for debugging
*
* Output (void):
*       d_QCOM: cumulative distribution in cosine angles Qth
*       d_GCOM: cumulative distribution of velocities
*       d_VCOM: spread of radial velocities
*
* Assumptions:
*       All inputs are real values
*       Ions are flowing in z-direction
*       Cylinder is along the z-axis
*       NUM_DIV_QTH = 3 for top, side, bottom
*
* Includes:
*       cuda_runtime.h
*       device_launch_parameters.h
*       <cstdio>
*       <fstream>
*       <string>
*
*/

void initInjectIonCylinder_101(
                const int ,
                const int ,
                const int ,
                const float ,
                const float ,
                float* ,
                float* ,
                float* ,
                float* ,
                const float ,
                const float ,
                const float ,
                float* ,
                float* ,
                float* ,
                const bool ,
                std::ostream&);

		
	/*
	* Name: invertFind_101
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
	__device__ float invertFind_101(float* const, int, float);


/*
 * Name: boundaryEField_101
 * Created: 10/10/20
 *
 * Editors
 *   Name: Lorin Matthews
 *   Contact: Lorin_Matthews@baylor.edu
 *
 * Description:
 *   Calculates the radial electric potential from ions outside the simulation boundary
 *
 * Input:
 *   GRID_POS:positions in the r-z plane
 *   GCYL_POS: positions within the cylindrical simulation region
 *   INV_DEBYE: 1/DEBYE, where DEBYE is electron Debye length
 *   DEN_ION: the number density of the ions
 * Output (void):
 *  ionOutPotential: potential calculated at each point in GRID_POS due to ions
 *                       contained within a box centered about GCYL_POS, summed
 *                       over all the GCYL_POS
 *
 */

__global__ void boundaryEField_101
    (float2*,
    float4*,
	int* const,
	int* const,
    float* const,
    float* const,
    float*,
    int);

/*
 * Name:tile_calculation_101
 * Created: 10/10/20
 *
 * Editors
 *   Name: Lorin Matthews
 *   Contact: Lorin_Matthews@baylor.edu
 *
 * Description:Evaluates potential from p GCYL_POS at p GRID_POS using
 *       a p x p tile.
 *       Based on Fast N_body calculation, tile_calculation Ch 31 GPU Gems.
 */
__device__ float tile_calculation_101(
    float2 myPosition,
    float invdebye,
    float ni,
    float V); 


/*
 * Name:pointPointPotential_101
 * Created: 10/10/20
 *
 * Editors
 *   Name: Lorin Matthews
 *   Contact: Lorin_Matthews@baylor.edu
 *
 * Description:Evaluates Yukawa potential at point bi from ions centered at bj
 *       Based on Fast N_body calculation, bodyBodyInteraction Ch 31 GPU Gems.
 */
 __device__ float pointPointPotential_101(
    float2,
    float4,
    float,
    float,
    float);
 
#endif // IONWAKE_101_BOUNDS
