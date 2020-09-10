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

	__global__ void calcDustIonAcc_103
           (float4 *, 
            float4 *,
			float4 *, 
            int * const,
			int * const,
			float * const, 
            float * const);
   
	__global__ void sumDustIonAcc_103
           (float4 *, 
			int * const,
			int * const);

	__global__ void zeroDustIonAcc_103
           (float4 *, 
			int * const,
			int * const);
#endif
