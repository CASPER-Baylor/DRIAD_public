/*
* Project: IonWake
* File Type: function library implemtation
* File Name: IonWake_108_ionDustAcc.cu
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
*/

// header file
#include "IonWake_108_ionDustAcc.h"

/*
* Name: calcIonDustForces
* Created: 6/13/2017
* last edit: 8/29/2017
*
* Editors
*	Name: Dustin Sanford
*	Contact: Dustin_Sanford@baylor.edu
*	last edit: 8/29/2017
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
__global__ void
calcIonDustForces(float3* d_posIon, float3* d_accIon, unsigned int* d_NUM_ION,
	float* d_SOFT_RAD_SQRD, float* d_ION_DUST_ACC_MULT, float* d_INV_DEBYE, 
	unsigned int* d_NUM_DUST, float3* d_posDust)
{

	// index of the current ion
	int IDcrntIon = blockIdx.x * blockDim.x + threadIdx.x;

	// allocate variables
	float3 dist;
	float distSquared;
	float hardDist;
	float linForce;

	// loop over all of the dust particles
	for (int h = 0; h < *d_NUM_DUST; h++)
	{

		// calculate the distance between the ion and the 
		// current dust particle 
		dist.x = d_posIon[IDcrntIon].x - d_posDust[h].x;
		dist.y = d_posIon[IDcrntIon].y - d_posDust[h].y;
		dist.z = d_posIon[IDcrntIon].z - d_posDust[h].z;

		// calculate the distance squared
		distSquared = dist.x*dist.x + dist.y*dist.y + dist.z*dist.z;

		// calculate the hard distance
		hardDist = __fsqrt_rn(distSquared);


		// calculate a scaler intermediat
		linForce = *d_ION_DUST_ACC_MULT*(1 + (hardDist**d_INV_DEBYE))
			*__expf(-hardDist**d_INV_DEBYE) / (hardDist*hardDist*hardDist);

		// add the acceleration to the current ion's acceleration
		d_accIon[IDcrntIon].x += linForce * dist.x;
		d_accIon[IDcrntIon].y += linForce * dist.y;
		d_accIon[IDcrntIon].z += linForce * dist.z;

	} // end loop over dust

}