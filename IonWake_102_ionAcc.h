/*
* Project: IonWake
* File Type: function library implementation
* File Name: IonWake_102_ionAcc.cu
*
* Created: 6/13/2017
* Last Modified: 10/21/2017
*
* Description:
*	Includes functions for calculating ion-ion accelerations 
*
* Functions:
*	calcIonIonAcc()
*   calcIonDustAcc()
*   calcExtrnElcAcc()
*
* Includes:
*	calcIonIonAcc()
*		cuda_runtime.h
*		device_launch_parameters.h
*	calcIonDustAcc()
*		cuda_runtime.h
*		device_launch_parameters.h
*	calcExtrnElcAcc()
*		cuda_runtime.h
*		device_launch_parameters.h
*/

#ifndef IONWAKE_102_IONACC
#define IONWAKE_102_IONACC

	/*
	* Required By:
	*	calcIonIonAcc()
    *	calcIonDustAcc()
    *	calcExtrnElcAcc()
	* For:
	*	CUDA
	*/
	#include "cuda_runtime.h"

	/*
	* Required By:
	*	calcIonIonAcc()
    *	calcIonDustAcc()
    *	calcExtrnElcAcc()
	* For:
	*	CUDA
	*/
	#include "device_launch_parameters.h"

    /*
    * calcIonIonAcc
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
	__global__ void calcIonIonAcc
           (float3*, 
            float3*, 
            int * const,
            float * const, 
            float * const, 
            float * const);

	/*
    * calcIonDustAcc
    *
    * Editors
    *	Dustin Sanford
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
    *	d_INV_DEBYE: the inverse of the debye
    *   d_chargeDust: the charge on the dust particles 
    *
    * Output (void):
    *	d_accIon: the acceleration due to all the dust particles
    *		is added to the initial ion acceleration
    *
    * Assumptions:
    *	All inputs are real values
    *	All ions and dust particles have the parameters specified in the 
    *       creation of the d_ION_ION_ACC_MULT value
    *   The potential due to the dust particle is a bare coulomb potential
    *   The number of ions is a multiple of the block size
    *
    * Includes:
    *	cuda_runtime.h
    *	device_launch_parameters.h
    *
    */
	__global__ void calcIonDustAcc(
			float3*, 
			float3*,
            float3*,
			int* const,
            int* const,
			float* const, 
			float* const, 
			float* const, 
			float* const);

    /*
    * calcExtrnElcAcc
    *
    * Editors
    *	Dustin Sanford
    *
    * Description:
    *	calculates the acceleration on the ions due to the electric field 
    *   created by the ions outside of a simulation sphere.
    *
    * Input:
    *	d_accIon: ion accelerations
    *	d_posIon: ion positions
    *	d_EXTERN_ELC_MULT: constant multiplier for calculating the electric 
    *       field due to the ions outside of the simulation sphere
    *	d_INV_DEBYE: the inverse debye length
    *
    * Output (void):
    *	d_accIon: the acceleration due to the outside electric 
    *		field is added to the initial ion accelerations
    *
    * Assumptions:
    *	All inputs are real values
    *	The simulation region is a sphere
    *	The electric field due to outside ions is  radially symmetric
    *	All ions have the parameters specified in the creation of 
    *		the d_EXTERN_ELC_MULT value
    *	The center of the simulation region is (0,0,0)
    *   The number of ions is a multiple of the block size
    *
    * Includes:
    *	cuda_runtime.h
    *	device_launch_parameters.h
    *
    */
	__global__ void calcExtrnElcAcc(float3*, float3*, float*, float*);

#endif