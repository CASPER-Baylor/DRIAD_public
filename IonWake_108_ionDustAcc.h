/*
* Project: IonWake
* File Type: function library header
* File Name: IonWake_108_ionDustAcc.h
*
* Created: 6/13/2017
* Last Modified: 8/29/2017
*
* Description:
*	Includes functions for handeling ion-dust accelerations
*
* Functions:
*	calcIonDustForces()
*
* Includes:
*	calcIonDustForces()
*		cuda_runtime.h
*		device_launch_parameters.h
*/

#ifndef IONWAKE_108
#define IONWAKE_108

	/*
	* Required By:
	*	calcIonDustForces()
	* For:
	*	CUDA
	*/
	#include "cuda_runtime.h"

	/*
	* Required By:
	*	calcIonDustForces()
	* For:
	*	CUDA
	*/
	#include "device_launch_parameters.h"

	/*
	* Name: calcIonDustForces
	*
	* Editors
	*	Dustin Sanford
	*
	* Description:
	*	Calculates the ion accelerations due to ion-dust
	*	interactions
	*
	* Input:
	*	d_posIon: the positions of the ions
	*	d_accIon: the accelerations of the ions
	*	d_NUM_ION: the number of ions
	*	d_SOFT_RAD_SQRD: the squared softening radius
	*	d_ION_DUST_ACC_MULT: a constant multiplier for the ion-dust interaction
	*	d_INV_DEBYE: the inverse of the debye
	*	d_NUM_DUST: the number of dust particles
	*	d_posDust: the dust particle poisitions
	*
	* Output (void):
	*	d_accIon: the acceleration due to all the dust particles
	*		is added to the initial ion acceleration
	*
	* Asumptions:
	*	All inputs are real values
	*	All ions and dust particles have the parameters specified in the creation
	*		of the d_ION_ION_ACC_MULT value
	*
	* Includes:
	*	cuda_runtime.h
	*	device_launch_parameters.h
	*
	*/
	__global__ void calcIonDustForces(float3*, float3* , unsigned int*, 
		float*, float*, float*, unsigned int*, float3*);

#endif