/*
* Project: IonWake
* File Type: function library implementation
* File Name: IonWake_102_ionAcc.cu
*
* Created: 6/13/2017
* Last Modified: 09/09/2020
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
*	calcIonDensityPotential_102()
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
	*	calcIonDensityPotential_102()
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
	*	calcIonDensityPotential_102()
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
    *	d_posIon: the positions and charges of the ions
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
           (float4*, 
            float4*, 
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
    *	d_posIon: the positions of the ions, charge in 4th pos'n
    *	d_accIon: the accelerations of the ions
    *	d_posDust: the dust particle positions, charge in 4th pos'n
    *	d_NUM_ION: the number of ions
    *	d_NUM_DUST: the number of dust particles
    *	d_SOFT_RAD_SQRD: the squared softening radius squared
    *	d_ION_DUST_ACC_MULT: a constant multiplier for the ion-dust interaction
    *	d_INV_DEBYE: the inverse of the debye
    *   d_mindistDust: distance to closest dust particle
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
           (float4*, 
			float4*,
            float4*,
			int* const,
            int* const,
			float* const, 
			float* const, 
			float*);
			
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
    *	d_posIon: ion positions and charges
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
	__global__ void calcExtrnElcAcc_102(float4*, float4*, float*, float*);

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
*       d_posIon: ion positions and charges
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
       (float4*,
        float4*,
        float*,
        float*, float*,float* ,
        float*, float*,float*, float*,float*,
		float*, int);

/*
 *  Name: calcIonDensityPotential_102
 *  Created: 5/4/2018
 *  Last Modified:8.27.2020 
 * 
 *  Editors
 * 	Name: Lorin Matthews
 * 	Contact: Lorin_Matthews@baylor.edu
 * 	last edit: 9/10/2020
 * 	Implemented float2 for grid positions
 * 
 *  Description:
 * 	Calculates electric potential from ions at points on grid in 
 *  	the xz-plane.  Also calculates the number density at each grid 
 * 	point by counting the number of ions in a sphere of radius r_dens
 *  	centered at each grid point.
 * 
 *  Input:
 * 	d_posIion: ion positions
 * 	d_gridPos: the grid points in xz-plane
 * 	d_ION_POTENTIAL_MULT
 * 	d_INV_DEBYE
 * 
 *  Output (void):
 * 	d_ionPotential: potential at each grid point
 * 	d_ionDenisty: ion number density at each grid point
 * 
 *  Assumptions: 
 *    The number of grid points is a multiple of the block size?????
 * 
 *  Includes:
 *	cuda_runtime.h
 *	device_launch_parameters.h
 * 
 */
__global__ void calcIonDensityPotential_102
	(float2*,
	 float4*,
	 float* const,
	 float* const,
	 int* const,
	 float*,
	 float*);

/*
 *  Name: zeroIonDensityPotential_102
 *  Created: 5/21/2018
 *  Last Modified: 5/21/2018
 * 
 *  Editors
 * 	Name: Lorin Matthews
 * 	Contact: Lorin_Matthews@baylor.edu
 * 	last edit: 5/21/2018
 * 
 *  Description:
 * 	Zeros electric potential from ions at points on grid in 
 *  	the xz-plane.  Also zeros the number density at each grid 
 * 
 *  Input:
 * 	d_ionPotential: potential at each grid point
 * 	d_ionDenisty: ion number density at each grid point
 *
 *  Output (void):
 * 	d_ionPotential: potential at each grid point
 * 	d_ionDenisty: ion number density at each grid point
 * 
 *  Assumptions: 
 *    The number of grid points is a multiple of the block size
 * 
 *  Includes:
 *	cuda_runtime.h
 *	device_launch_parameters.h
 * 
 */
__global__ void zeroIonDensityPotential_102
	 (float*,
	 float*);
#endif
