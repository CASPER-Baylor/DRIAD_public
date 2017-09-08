/*
* Project: IonWake
* File Type: function library header
* File Name: IonWake_102_ionIonAcc.h
*
* Created: 6/13/2017
* Last Modified: 8/28/2017
*
* Description:
*	Includes functions for handeling ion-ion accelerations
*
* Functions:
*	calcIonIonForces()
*
* Includes:
*	calcIonIonForces()
*		cuda_runtime.h
*		device_launch_parameters.h
*	
*/

#ifndef IONWAKE_102
#define IONWAKE_102

	/*
	* Required By:
	*	calcIonIonForces()
	* For:
	*	CUDA
	*/
	#include "cuda_runtime.h"

	/*
	* Required By:
	*	calcIonIonForces()
	* For:
	*	CUDA
	*/
	#include "device_launch_parameters.h"

	/*
	* Name: calcIonIonForces
	*
	* Editors
	*	Dustin Sanford
	*
	* Description:
	*	Calculates the accelerations due to ion-ion
	*	interactions modled as Yakawa particles
	*
	* Input:
	*	d_posIon: the positions of the ions
	*	d_accIon: the accelerations of the ions
	*	d_NUM_ION: the number of ions
	*	d_SOFT_RAD_SQRD: the squared softening radius
	*	d_ION_ION_ACC_MULT: a constant multiplier for the yakawa interaction
	*	d_INV_DEBYE: the inverse of the debye
	*
	* Output (void):
	*	d_accIon: the acceleration due to all of the other ions
	*		is added to the initial ion acceleration
	*
	* Asumptions:
	*	All inputs are real values
	*	All ions have the parameters specified in the creation of the
	*		d_ION_ION_ACC_MULT value
	*
	* Includes:
	*	cuda_runtime.h
	*	device_launch_parameters.h
	*
	*/
	__global__ void calcIonIonForces(float3*, float3*, unsigned int * const,
		float * const, float * const, float * const);

#endif // IONWAKE_102
