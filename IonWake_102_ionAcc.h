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
*	calcIonIonAcc_102()
*   calcIonDustAcc_102()
*   calcIonDustAcc_102_dev()
*   calcExtrnElcAcc_102()
*
* Includes:
*	calcIonIonAcc_102()
*		cuda_runtime.h
*		device_launch_parameters.h
*	calcIonDustAcc_102()
*		cuda_runtime.h
*		device_launch_parameters.h
*	calcExtrnElcAcc_102()
*		cuda_runtime.h
*		device_launch_parameters.h
*	calcExtrnElcAccCyl_102()
*		cuda_runtime.h
*		device_launch_parameters.h
*/

#ifndef IONWAKE_102_IONACC
#define IONWAKE_102_IONACC

	/*
	* Required By:
	*	calcIonIonAcc_102()
    *	calcIonDustAcc_102()
    *	calcExtrnElcAcc_102()
    *	calcExtrnElcAccCyl_102()
	* For:
	*	CUDA
	*/
	#include "cuda_runtime.h"

	/*
	* Required By:
	*	calcIonIonAcc_102()
    *	calcIonDustAcc_102()
    *	calcExtrnElcAcc_102()
    *	calcExtrnElcAccCyl_102()
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
	__global__ void calcIonIonAcc_102
           (float3*, 
            float3*, 
            int * const,
            float * const, 
            float * const, 
            float * const);

	/*
    * calcIonDustAcc_102
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
	__global__ void calcIonDustAcc_102
           (float3*, 
			float3*,
            float3*,
			int* const,
            int* const,
			float* const, 
			float* const, 
			float* const,
			float*);

/*
* Name: calcIonDustAcc_102_dev
* Created: 3/20/2018
*
* Editors
*	Name: Lorin Matthews
*	Contact: Lorin_Matthews@baylor.edu
*	last edit: 3/20/2018
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
*   d_chargeDust: the charge on the dust particles 
*
* Output (void):
*	d_accIon: the acceleration due to all the dust particles
*		is added to the initial ion acceleration
*
* Assumptions:
*	All inputs are real values
*	All ions and dust particles have the parameters specified in the creation 
*		of the d_ION_ION_ACC_MULT value
*   The potential due to the dust particle is a bare coulomb potential
*   The number of ions is a multiple of the block size
*
* Includes:
*	cuda_runtime.h
*	device_launch_parameters.h
*
*/
__device__ void calcIonDustAcc_102_dev(
		float3*, 
		float3*, 
        float3*,
		const int*,
        const int*, 
		const float*, 
		const float*, 
		const float*);
			
/*
* Name: calcIonDustAcc_102_dev
* Created: 3/17/2018
*
* Editors
*	Name: Lorin Matthews
*	Contact: Lorin_Matthews@baylor.edu
*	last edit: 3/17/2018
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
*   d_chargeDust: the charge on the dust particles 
*
* Output (void):
*	d_accIon: the acceleration due to all the dust particles
*		is added to the initial ion acceleration
*
* Assumptions:
*	All inputs are real values
*	All ions and dust particles have the parameters specified in the creation 
*		of the d_ION_ION_ACC_MULT value
*   The potential due to the dust particle is a bare coulomb potential
*   The number of ions is a multiple of the block size
*
* Includes:
*	cuda_runtime.h
*	device_launch_parameters.h
*
*/
__device__ void calcIonDustAcc_102_dev(
		float3*, 
		float3*, 
        float3*,
		int* const,
        int* const, 
		float* const, 
		float* const, 
		float* const);
		
    /*
    * calcExtrnElcAcc_102
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
	__global__ void calcExtrnElcAcc_102(float3*, float3*, float*, float*);

/*
* Name: calcExtrnElcAccCyl_102
* Created: 11/18/2017
* last edit: 11/18/2017
*
* Editors
*       Name: Lorin Matthews
*       Contact: Lorin_Matthews@baylor.edu
*       last edit: 11/18/2017
*
* Description:
*       calculates the acceleration on the ions due to the electric field created
*   by the ions outside of a simulation cylinder.
*
* Input:
*       d_accIon: ion accelerations
*       d_posIon: ion positions
*       d_Q_DIV_M:  charge to mass ratio
*       d_p10x: coefficient for radial E field
*       d_p12x: coefficient for radial E field
*       d_p14x: coefficient for radial E field
*       d_p01z: coefficient for vertical E field
*       d_p21z: coefficient for vertical E field
*       d_p03z: coefficient for vertical E field
*       d_p23z: coefficient for vertical E field
*       d_p05z: coefficient for vertical E field
*
* Output (void):
*       d_accIon: the acceleration due to the outside electric
*               field is added to the initial ion accelerations
*
* Assumptions:
*       All inputs are real values
*       The simulation region is a cylinder
*       The electric field due to outside ions is radially symmetric
*       The center of the simulation region is (0,0,0)
*       The coefficients for the electric fields were calculated using the
*         Matlab routine e_field_in_cylinder.m using the correct dimentions
*         for the cylinder, ion density, and debye length.
*   The number of ions is a multiple of the block size
*
* Includes:
*       cuda_runtime.h
*       device_launch_parameters.h
*
*
*/
__global__ void calcExtrnElcAccCyl_102
       (float3*,
        float3*,
        float*,
        float*, float*,float* ,
        float*, float*,float*, float*,float*);

#endif
