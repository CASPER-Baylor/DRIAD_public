/*
* Project: IonWake
* File Type: function library implementation
* File Name: IonWake_105_ionCollisions.cu
*
* Created: 6/15/2018
*
* Description:
*	Includes functions for calculating ion-neutral collisions.  
*	All routines are based on:
*	Particle-In-Cell + Monte-Carlo-Collision for Helium gas  (Peter Hartmann) 
* 	translated from he_pic_2009_f.cc Donko Zoltan   
*
* Functions:
*	setIonCrossSection_105()
*	ionCollisions_105()
*	setCollisionList_105()
* Local functions:
*   collisionIonNeutral
*	random_maxwell_velocity
*   errorFn_Inv
*
* Includes:
*   ionCollisions_105()
*   	device_launch_parameters.h
*   	cuda_runtime.h
*   	curand_kernel.h
*
*/

// header file
#ifndef IONWAKE_105_IONCOLLISIONS
#define IONWAKE_105_IONCOLLISIONS

//for sqrt and rand
#include <cmath>

// for fprintf(), stderr
#include <cstdio>

//for fstream
#include <fstream>

//for std::string
#include <string>

// srand
#include <stdlib.h>
//time
#include <time.h>

//for std::vector
//#include <vector>

//for std::copy
//#include <algorithm>

//for CUDA
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include <curand_kernel.h>

/*
* Name: setIonCrossSections_105
* Created: 6/15/2018
*
* Editors
*	Name: Lorin Matthews
*	Contact: Lorin_Matthews@baylor.edu
*
* Description:
*	Reads in ion neutral collision cross sections and determines
*	total cross section and total ion collision frequency.
*
* Input:
*	d_gasType: type of gas (Argon or Neon)
*	i_cs_ranges: number of entries in range of ion cross sections
*
* Output (void):
*	d_sigma_i: backscattering and elastic cross sections as a function of energy
*	d_sigma_i_tot: total collision cross sections
*	d_tot_ion_coll_freq: summed ion collision frequency
*
*/
void setIonCrossSection_105
       (const int, 
		const int,
		const float,
		const float,
        float*, 
        float*, 
        float*, 
        float&,
		const bool, 
		std::ostream&) ;
		 

/*
* Name: ionCollisions_105
* Created: 6/15/2018
*
* Editors
*	Name: Lorin Matthews
*	Contact: Lorin_Matthews@baylor.edu
*
* Description:
*	Performs collisions between ions and neutrals.  
*	Particle-In-Cell + Monte-Carlo-Collision for Helium gas  (Peter Hartmann) 
* 	translated from he_pic_2009_f.cc Donko Zoltan             
*
* Input:
*	NUM_ION
* 	tot_ion_coll_freq
*	TIME_STEP = dt_i: ion time step
*	velIon
*	TEMP_ION
*	MASS_SINGLE_ION
*	sigma_i1
*	sigma_i2
*	sigma_i_tot
*
* Output (void):
*	velIon
*
*/

//__global__ void ionCollisions_105 (
//		int*,
//		float* const,
__global__ void ionCollisions_105 (
		int*,
		float* const,
		float* const,
		float* const,
		int* const,
		float* const,
		float*,
		float*,
		float*,
		float3*,
		curandState_t* const, 
		int*);

/*
* Name: setCollisionList_105() 
*
* Description:
*	Zeros the collision list 
*
* Inputs:
*	d_collList: list of ions to undergo collision 
*
* Output (void):
*	d_collList: is set to zero or one 
*	
* Assumptions:
*	The number entries = # of ions is a multiple of the block size
*
* Includes:
*	cuda_runtime.h
* 	device_launch_parameters.h
*
*/

__global__ void setCollisionList_105
	(int*, int); 

/*
* Name: collision_ion_neutral_105
* Created: 6/15/2018
*
* Editors
*	Name: Lorin Matthews
*	Contact: Lorin_Matthews@baylor.edu
*
* Description:
*	Models a collision between ion and neutral atom, setting the new
*	velocity for the ion.
*
* Input:
*	index: the index of the ion which was chosen for collision
*	ionPos: position of the ion before the collision
*	ionVel: velocity of ion before collision
*	vx_2: velocity of neutral atom (squared)
*	vy_2:
*	vz_2:
*	sigma_i: collision cross section
*	ee: index for energy of the collision
*
* Output (void):
*	ionVel: velocity of ion after collision
*
*/
//void collisionIonNeutral(
//	int, 
//	float3*,  
//	double, 
//	double, 
//	double, 
//	double,
//	double,
//	double);


//--------------------------------------------------------------------------    
// Maxwellian target sampling 
//--------------------------------------------------------------------------    
/*
* Name: errorFn_inv_105
* Created: 6/15/2018
*
* Editors
*	Name: Lorin Matthews
*	Contact: Lorin_Matthews@baylor.edu
*
* Description:
*	Inverts the error function -- used to create a random number with
*	a Gaussian probability distribution.  Algorithm adapted from
* 	Particle-In-Cell + Monte-Carlo-Collision for Helium gas (Peter Hartmann)  
* 	translated from he_pic_2009_f.cc Donko Zoltan             
*
* Input:
*	y: a random number -1 <   y    < 1
*
* Output (void):
*	x: -inf < erf^-1 < inf
*
*/

__device__ float errorFn_inv(float y); 

//--------------------------------------------------------------------
// sampling of Maxwellian distributions :

//double random_maxwell_velocity(void);

#endif // IONWAKE_105_IONCOLLISIONS
