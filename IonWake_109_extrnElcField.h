/*
* Project: IonWake
* File Type: function library header
* File Name: IonWake_109_extrnElcField.h
*
* Created: 6/13/2017
* Last Modified: 8/29/2017
*
* Description:
*	Includes functions for handeling the electric
*	field outside of the simulation region
*
* Functions:
*	calcExtrnElcForce()
*
*/

#ifndef ION_WAKE_109
#define ION_WAKE_109

/*
* Required By:
*	stepForward()
* For:
*	CUDA
*/
#include "cuda_runtime.h"

/*
* Required By:
*	stepForward()
* For:
*	CUDA
*/
#include "device_launch_parameters.h"

	/*
	* Name: calcExtrnElcForce
	*
	* Editors
	*	Dustin Sanford
	*
	* Description:
	*	calculates the acceleration on the ions due to
	*	the electric field created by the ions outside
	*	of a simulation sphere.
	*
	* Input:
	*	d_accIon: ion accelerations
	*	d_posIon: ion positions
	*	d_EXTERN_ELC_MULT: constant multiplier for calculating
	*		the electric field due to the ions outside of the
	*		simulation sphere
	*	d_INV_DEBYE: the inverse debye length
	*
	* Output (void):
	*	d_accIon: the acceleration due to the outside electric
	*		field is added to the initial ion accelerations
	*
	* Asumptions:
	*	All inputs are real values
	*	The simlation region is a sphere
	*	The electric field due to outside ions is radialy semetric
	*	All ions have the parameters specified in the creation of
	*		the d_EXTERN_ELC_MULT value
	*	The center of the simulation region is (0,0,0)
	*
	* Includes:
	*	cuda_runtime.h
	*	device_launch_parameters.h
	*
	*/
	__global__ void calcExtrnElcForce(float3*, float3*, float*, float*);

#endif