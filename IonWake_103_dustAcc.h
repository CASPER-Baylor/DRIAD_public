/*
* Project: IonWake
* File Type: function library header
* File Name: IonWake_103_dustAcc.h
*
* Description:
*	Includes functions for calculating dust accelerations 
*
* Functions:
*	calcDustIonAcc_103()
*
* Includes:
*	calcDustIonAcc_103()
*		cuda_runtime.h
*		device_launch_parameters.h
*/

#ifndef IONWAKE_103_DUSTACC
#define IONWAKE_103_DUSTACC

	/*
	* Required By:
	* calcDustIonAcc_103()
	* For:
	*	CUDA
	*/
	#include "cuda_runtime.h"

	/*
	* Required By:
	* calcDustIonAcc_103()
	* For:
	*	CUDA
	*/
	#include "device_launch_parameters.h"

    /*
    * calcIonIonAcc_102
    *
    * Editors
    *	Dustin Sanford
    *
    * Description:
    *	Calculates the accelerations due to ion-ion 
    *	interactions modeled as Yakawa particles.
    *
    * Input:
    *	d_posIon: the positions of the ions
    *	d_accIon: the accelerations of the ions
    *	d_NUM_ION: the number of ions
    *	d_SOFT_RAD_SQRD: the squared softening radius squared
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
    *   The ion potential is a Yukawa potential
    *   The number of ions is a multiple of the block size 
    *
    * Includes:
    *	cuda_runtime.h
    *	device_launch_parameters.h
    *
    */
	__global__ void calcDustIonAcc_103
           (float3 *, 
            float3 *,
			float3 *, 
            int * const,
			int * const,
			int * const, 
            float * const);
#endif
