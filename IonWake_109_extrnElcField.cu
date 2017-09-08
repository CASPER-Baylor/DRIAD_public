/*
* Project: IonWake
* File Type: function library implemtation
* File Name: IonWake_109_extrnElcField.cu
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

// header file
#include "IonWake_109_extrnElcField.h" 

/*
* Name: calcExtrnElcForce
* Created: 6/13/2017
* last edit: 8/29/2017
*
* Editors
*	Name: Dustin Sanford
*	Contact: Dustin_Sanford@baylor.edu
*	last edit: 8/29/2017
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
__global__ void calcExtrnElcForce(float3* d_accIon, float3* d_posIon, 
	float* d_EXTERN_ELC_MULT, float * d_INV_DEBYE)
{

	// the thread ID
	int ID = blockIdx.x * blockDim.x + threadIdx.x;

	// get the radius of the ion from the center of the
	// simulation sphere. The center is assumed to be (0,0,0)
	float rad = __fsqrt_rn(
		(d_posIon[ID].x * d_posIon[ID].x) +
		(d_posIon[ID].y * d_posIon[ID].y) +
		(d_posIon[ID].z * d_posIon[ID].z)) ;

	// calculate an intermediate value for use in the
	// acceleration calculation
	float intrmed = rad * *d_INV_DEBYE;

	// calculate a scaler value for the acceleration.
	// to get the acceleration for the ion, multiply the
	// scaler value by the vector distance to the center 
	// of the simulation sphere
	float linForce = -*d_EXTERN_ELC_MULT *
		(
			sinhf(intrmed) - (coshf(intrmed) / intrmed)
		)/(rad * rad);

	// multiply by the vector distance to the center of 
	// the simulation radius and add it to the ion
	// acceleration
	d_accIon[ID].x += d_posIon[ID].x * linForce;
	d_accIon[ID].y += d_posIon[ID].y * linForce;
	d_accIon[ID].z += d_posIon[ID].z * linForce;
}