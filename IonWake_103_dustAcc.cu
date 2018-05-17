/*
* Project: Ionwake
* File Type: function library implementation
* File Name: IonWake_103_dustAcc.cu
* 
* Description:
* 	Includes functions for calculating dust accelerations
*
* Functions:
*	calcDustIonAcc_103()
*
*/

// header file
#include "IonWake_103_dustAcc.h"

/*
* Name: calcDustIonAcc_103()
*
* Description:
*	Calculates the force on each dust particle due to each ion. 
*
* Input:
*	d_posIon: the ion positions
*	d_posDust: the dust positions
*	d_accDustIon: is clobbered
*	d_NUM_DUST: the number of dust particles
*	d_NUM_ION: the number of ions
*	d_chargeDust: the charge on each dust particle
*
* Output (void):
*	d_accDustIon: the acceleration on each dust partice from each ion 
*		(has length NUM_DUST * NUM_ION)
*	
* Assumptions:
*	The number of ions is a multiple of the block size
*	The number of threads is equal to the number of ions
*
* Includes:
*	cuda_runtime.h
* 	device_launch_parameters.h
*
*/

__global__ void calcDustIonAcc_103(
	float3* d_posIon,
	float3* d_posDust,
	float3* d_accDustIon,
	float* const d_chargeDust,
	int* const d_NUM_DUST,
	int* const d_NUM_ION,
	float* const d_INV_DEBYE,
	float* const d_DUST_ION_ACC_MULT) {

	// index of the current ion
	int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	
	// allocate variables 
	float3 dist;
	float sDist;
	float linForce;

	float3 posIon = d_posIon[threadID];
	int index = threadID;

	//reset the acceleration
	d_accDustIon[index].x = 0;
	d_accDustIon[index].y = 0;
	d_accDustIon[index].z = 0;
	
	// loop over all of the dust particles
	for(int i = 0; i < *d_NUM_DUST; i++) {
		
		// x, y, and z distances between the ion and dust particle
		dist.x = posIon.x - d_posDust[i].x;
		dist.y = posIon.y - d_posDust[i].y;
		dist.z = posIon.z - d_posDust[i].z;
	
		// distance between the ion and dust particle
		sDist = __fsqrt_rn(dist.x * dist.x + dist.y * dist.y + dist.z * dist.z);

		// calculate a scalar intermediate
		linForce = *d_DUST_ION_ACC_MULT * d_chargeDust[i] / 
			(sDist * sDist * sDist)
			*(1 + sDist* *d_INV_DEBYE) * __expf(-sDist* *d_INV_DEBYE);

		// ion dust acceleration acceleration 
		d_accDustIon[index].x = -linForce * dist.x;
		d_accDustIon[index].y = -linForce * dist.y;
		d_accDustIon[index].z = -linForce * dist.z;
	
		index += *d_NUM_ION;		
	}
}


/*
* Name: sumDustIonAcc() 
*
* Description:
*	Sums the forces on each dust particle due to each ion. 
*
* Inputs:
*	d_accDustIon: dust accleration due to each dust-ion pair
*	d_accDust: acceleration of each dust particle 
*	d_NUM_DUST: the number of dust particles
*	d_NUM_ION: the number of ions
*
* Output (void):
*	d_accDust: the acceleration due to the ions in the simulation is 
*		added to the input accDust
*	
* Assumptions:
*	The number of ions is a multiple of the block size
*	The number of threads is equal to half the number of ions
*
* Includes:
*	cuda_runtime.h
* 	device_launch_parameters.h
*
*/

__global__ void sumDustIonAcc_103(
	float3* d_accDustIon,
	int* const d_NUM_DUST,
	int* const d_NUM_ION) {

	extern __shared__ float3 sumData[];

	int localID = threadIdx.x;
	int globalID = blockIdx.x * blockDim.x * 2 + threadIdx.x; 
	
	for(int j = 0; j < *d_NUM_DUST; j++) {

		sumData[localID].x = d_accDustIon[globalID].x + d_accDustIon[globalID + blockDim.x].x;
		sumData[localID].y = d_accDustIon[globalID].y + d_accDustIon[globalID + blockDim.x].y;
		sumData[localID].z = d_accDustIon[globalID].z + d_accDustIon[globalID + blockDim.x].z;
	
		__syncthreads();
	
		for(int i = blockDim.x / 2; i > 0; i>>=1){
			if (localID < i) {
				sumData[localID].x += sumData[localID + i].x;
				sumData[localID].y += sumData[localID + i].y;
				sumData[localID].z += sumData[localID + i].z;
				__syncthreads();
			}
		}
	
		if (localID == 0) {
			d_accDustIon[blockIdx.x + j * *d_NUM_ION] = sumData[0];
			//d_accDustIon[blockIdx.x + j * *d_NUM_ION].y = sumData[0].y;
			//d_accDustIon[blockIdx.x + j * *d_NUM_ION].z = sumData[0].z;
		}

		globalID += *d_NUM_ION;
	}
}






