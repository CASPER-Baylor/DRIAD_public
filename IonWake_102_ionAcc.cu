/*
* Project: IonWake
* File Type: function library implementation
* File Name: IonWake_102_ionAcc.cu
*
* Created: 6/13/2017
* Last Modified: 11/14/2017
*
* Description:
*	Includes functions for calculating ion-ion accelerations 
*
* Functions:
*	calcIonIonAcc_102()
*   calcIonDustAcc_102()
*   calcExtrnElcAcc_102()
*   calcExtrnElcAccCyl_102()
*
*/

// header file
#include "IonWake_102_ionAcc.h"

/*
* Name: calcIonIonAcc_102
* Created: 6/13/2017
* last edit: 11/14/2017
*
* Editors
*	Name: Dustin Sanford
*	Contact: Dustin_Sanford@baylor.edu
*	last edit: 11/14/2017
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
* Assumptions:
*	All inputs are real values
*	All ions have the parameters specified in the creation of the 
*		d_ION_ION_ACC_MULT value
*   The ion potential is a Yukawa potentail
*   The number of ions is a multiple of the block size 
*
* Includes:
*	cuda_runtime.h
*	device_launch_parameters.h
*
*/
__global__ void calcIonIonAcc_102
       (float3* d_posIon, 
        float3* d_accIon, 
        int * const d_NUM_ION,
        float * const d_SOFT_RAD_SQRD, 
        float * const d_ION_ION_ACC_MULT,
        float * const d_INV_DEBYE)
{

	// index of the current ion
	int IDcrntIon = blockIdx.x * blockDim.x + threadIdx.x;

	// initialize variables
	float3 dist;
	float3 accCrntIon = { 0,0,0 };
	float distSquared;
	float hardDist;
	float softDist;
	float linForce;
	int tileThreadID;

	// allocate shared memory
	extern __shared__ float3 sharedPos[];

	// loop over all of the ions by using tiles. Where each tile is a section
	// of the ions that is loaded into shared memory. Each tile consists of 
	// as many ions as the block size. Each thread is responsible for loading 
	// one ion position for the tile.
	for (int tileOffset = 0; tileOffset < *d_NUM_ION; tileOffset += blockDim.x) 
    {
		// the index of the ion for the thread to load
		// for the current tile
		tileThreadID = tileOffset + threadIdx.x; 

		// load in an ion position
		sharedPos[threadIdx.x].x = d_posIon[tileThreadID].x;
		sharedPos[threadIdx.x].y = d_posIon[tileThreadID].y;
		sharedPos[threadIdx.x].z = d_posIon[tileThreadID].z;
		
		// wait for all threads to load the current position
		__syncthreads();

		// DEBUGING // 
		/*
		// PTX code used to access shared memory sizes
		// which are save to "ret"
		unsigned ret;
		asm volatile ("mov.u32 %0, %total_smem_size;" : "=r"(ret));
		*/

		// loop over all of the ions loaded in the tile
		for (int h = 0; h < blockDim.x; h++) {

			// calculate the distance between the ion in shared
			// memory and the current thread's ion
			dist.x = d_posIon[IDcrntIon].x - sharedPos[h].x;
			dist.y = d_posIon[IDcrntIon].y - sharedPos[h].y;
			dist.z = d_posIon[IDcrntIon].z - sharedPos[h].z;

			// calculate the distance squared
			distSquared = dist.x*dist.x + dist.y*dist.y + dist.z*dist.z;

			// calculate the hard distance
			hardDist = __fsqrt_rn(distSquared);

			// calculate the soft distance
			softDist = __fsqrt_rn(distSquared + *d_SOFT_RAD_SQRD);

			// calculate a scaler intermediate
			linForce = *d_ION_ION_ACC_MULT*(1 + (hardDist**d_INV_DEBYE))
				*__expf(-hardDist**d_INV_DEBYE) / (softDist*softDist*softDist);

			// add the acceleration to the current ion's acceleration
			accCrntIon.x += linForce * dist.x;
			accCrntIon.y += linForce * dist.y;
			accCrntIon.z += linForce * dist.z;
		} // end loop over ion in tile

		// wait for all threads to finish calculations
		__syncthreads();
	} // end loop over tiles

	// save to global memory
	d_accIon[IDcrntIon].x += accCrntIon.x;
	d_accIon[IDcrntIon].y += accCrntIon.y;
	d_accIon[IDcrntIon].z += accCrntIon.z;
}

/*
* Name: calcIonDustAcc_102
* Created: 6/13/2017
* last edit: 11/14/2017
*
* Editors
*	Name: Dustin Sanford
*	Contact: Dustin_Sanford@baylor.edu
*	last edit: 11/14/2017
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
__global__ void calcIonDustAcc_102(
		float3* d_posIon, 
		float3* d_accIon, 
        float3* d_posDust,
		int* const d_NUM_ION,
        int* const d_NUM_DUST, 
		float* const d_SOFT_RAD_SQRD, 
		float* const d_ION_DUST_ACC_MULT, 
		float* const d_INV_DEBYE, 
		float* const d_chargeDust)
{

	// index of the current ion
	int IDcrntIon = blockIdx.x * blockDim.x + threadIdx.x;

	// allocate variables
	float3 dist;
	float distSquared;
	float hardDist;
	float linForce;

	// loop over all of the dust particles
	for (int h = 0; h < *d_NUM_DUST; h++) {

			// calculate the distance between the ion in shared
			// memory and the current thread's ion
			dist.x = d_posIon[IDcrntIon].x - d_posDust[h].x;
			dist.y = d_posIon[IDcrntIon].y - d_posDust[h].y;
			dist.z = d_posIon[IDcrntIon].z - d_posDust[h].z;

			// calculate the distance squared
			distSquared = dist.x*dist.x + dist.y*dist.y + dist.z*dist.z;

			// calculate the hard distance
			hardDist = __fsqrt_rn(distSquared);

			// calculate a scaler intermediate
			linForce = *d_ION_DUST_ACC_MULT * d_chargeDust[h] / 
                        (hardDist*hardDist*hardDist);

			// add the acceleration to the current ion's acceleration
			d_accIon[IDcrntIon].x += linForce * dist.x;
			d_accIon[IDcrntIon].y += linForce * dist.y;
			d_accIon[IDcrntIon].z += linForce * dist.z;

	} // end loop over dust
}

/*
* Name: calcExtrnElcAcc_102
* Created: 6/13/2017
* last edit: 11/14/2017
*
* Editors
*	Name: Dustin Sanford
*	Contact: Dustin_Sanford@baylor.edu
*	last edit: 11/14/2017
*
* Description:
*	calculates the acceleration on the ions due to the electric field created 
*   by the ions outside of a simulation sphere.
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
*	The electric field due to outside ions is radially symmetric
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
__global__ void calcExtrnElcAcc_102
       (float3* d_accIon, 
        float3* d_posIon, 
        float* const d_EXTERN_ELC_MULT, 
        float* const d_INV_DEBYE)
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
	float linForce = *d_EXTERN_ELC_MULT *
		(
			sinhf(intrmed) - (coshf(intrmed) * intrmed)
		)/(intrmed * intrmed * rad);

	// multiply by the vector distance to the center of 
	// the simulation radius and add it to the ion
	// acceleration
	d_accIon[ID].x += d_posIon[ID].x * linForce;
	d_accIon[ID].y += d_posIon[ID].y * linForce;
	d_accIon[ID].z += d_posIon[ID].z * linForce;
}


/*
* Name: calcExtrnElcAccCyl_102
* Created: 11/18/2017
* last edit: 11/18/2017
*
* Editors
*	Name: Lorin Matthews
*	Contact: Lorin_Matthews@baylor.edu
*	last edit: 11/18/2017
*
* Description:
*	calculates the acceleration on the ions due to the electric field created 
*   by the ions outside of a simulation cylinder.
*
* Input:
*	d_accIon: ion accelerations
*	d_posIon: ion positions
*	d_Q_DIV_M:  charge to mass ratio
*	d_p10x: coefficient for radial E field
*	d_p12x: coefficient for radial E field
*	d_p14x: coefficient for radial E field
*	d_p01z: coefficient for vertical E field
*	d_p21z: coefficient for vertical E field
*	d_p03z: coefficient for vertical E field
*	d_p23z: coefficient for vertical E field
*	d_p05z: coefficient for vertical E field
*
* Output (void):
*	d_accIon: the acceleration due to the outside electric 
*		field is added to the initial ion accelerations
*
* Assumptions:
*	All inputs are real values
*	The simulation region is a cylinder
*	The electric field due to outside ions is radially symmetric
*	The center of the simulation region is (0,0,0)
*	The coefficients for the electric fields were calculated using the
*	  Matlab routine e_field_in_cylinder.m using the correct dimentions
*	  for the cylinder, ion density, and debye length.
*   The number of ions is a multiple of the block size
*
* Includes:
*	cuda_runtime.h
*	device_launch_parameters.h
*
*/
__global__ void calcExtrnElcAccCyl_102
       (float3* d_accIon, 
        float3* d_posIon, 
	float* d_Q_DIV_M,
	float* d_p10x, float* d_p12x,float* d_p14x,
	float* d_p01z, float* d_p21z,float* d_p03z, float* d_p23z,float* d_p05z)
{

	// the thread ID
	int ID = blockIdx.x * blockDim.x + threadIdx.x;

	// get the radius of the ion from the center of the
	// simulation sphere. The center is assumed to be (0,0,0)
	float rad = __fsqrt_rn(
		(d_posIon[ID].x * d_posIon[ID].x) +
		(d_posIon[ID].y * d_posIon[ID].y)) ;

	// get the z position of the ion
	float z = d_posIon[ID].z;
	float zsq = z * z;

	// calculate the radial component of the acceleration
	// Since this has to be turned into vector components, it
	// it divided by rad.
	float radAcc = *d_p10x  +
			*d_p12x * zsq +
			*d_p14x * zsq * zsq;

	// calculate vertical component of the acceleration
	float vertAcc = *d_p01z * z +
			*d_p21z * rad * rad * z +
			*d_p03z * z * zsq +
			*d_p23z * rad * rad * z * zsq +
			*d_p05z * z * zsq * zsq;

	// multiply by the vector distance to the center of 
	// the simulation radius and add it to the ion
	// acceleration
	d_accIon[ID].x += d_posIon[ID].x * radAcc * *d_Q_DIV_M;
	d_accIon[ID].y += d_posIon[ID].y * radAcc * *d_Q_DIV_M;
	d_accIon[ID].z += vertAcc * *d_Q_DIV_M;
}
