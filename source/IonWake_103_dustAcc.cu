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
 *	d_posIon: the ion positions and charges
 *	d_posDust: the dust positions and charges
 *	d_accDustIon: is clobbered
 *	d_NUM_DUST: the number of dust particles
 *	d_NUM_ION: the number of ions
 *
 * Output (void):
 *	d_accDustIon: the acceleration on each dust particle from each ion
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

__global__ void calcDustIonAcc_103(float4 *d_posIon, float4 *d_posDust, float4 *d_accDustIon,
                                   int *const d_NUM_DUST, int *const d_NUM_ION,
                                   float *const d_INV_DEBYE, float *const d_DUST_ION_ACC_MULT)
{

    // index of the current ion
    int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    // allocate variables
    float3 dist;
    float sDist;
    float linForce;

    float4 posIon = d_posIon[threadID];
    int index = threadID;
    // float qIon = d_posIon[threadID].w;

    // loop over all of the dust particles
    for (int i = 0; i < *d_NUM_DUST; i++)
    {

        // x, y, and z distances between the ion and dust particle
        dist.x = posIon.x - d_posDust[i].x;
        dist.y = posIon.y - d_posDust[i].y;
        dist.z = posIon.z - d_posDust[i].z;

        // distance between the ion and dust particle
        sDist = __fsqrt_rn(dist.x * dist.x + dist.y * dist.y + dist.z * dist.z);

        // calculate a scalar intermediate
        linForce = *d_DUST_ION_ACC_MULT * d_posDust[i].w * posIon.w / (sDist * sDist * sDist) *
                   (1 + sDist * *d_INV_DEBYE) * __expf(-sDist * *d_INV_DEBYE);

        // ion dust acceleration acceleration
        d_accDustIon[index].x -= linForce * dist.x;
        d_accDustIon[index].y -= linForce * dist.y;
        d_accDustIon[index].z -= linForce * dist.z;

        index += *d_NUM_ION;
    }
}

/*
 * Name: sumDustIonAcc()
 *
 * Description:
 *	Sums the forces on each dust particle due to each ion.
 *	Each dust particle has force from NUM_ION, where there are
 * 	2*blockDim * X ions.  The first step
 * 	in the sum loads and adds from X blocks. The function
 *	is called with as many blocks as there are NUM_DUST.
 *
 * Inputs:
 *	d_accDustIon: dust accleration due to each dust-ion pair
 *	d_NUM_DUST: the number of dust particles
 *	d_NUM_ION: the number of ions
 *
 * Output (void):
 *	d_accDust: the acceleration due to the ions in the simulation is
 *		added to the input accDust
 *
 * Assumptions:
 *	The number of ions is a multiple of 2 * block size
 *	The number of threads is equal to the number of ions
 *
 * Includes:
 *	cuda_runtime.h
 * 	device_launch_parameters.h
 *
 */

__global__ void sumDustIonAcc_103(float4 *d_accDustIon, int *const d_NUM_DUST, int *const d_NUM_ION)
{
    // the kernel sumDustIonAcc_103 to get the total ion acceleration over each dust grain
    // is coded assuming that DIM_BLOCK=1024. If DIM_BLOCK is changed the kernel will
    // NOT work properly

    extern __shared__ float4 sumData[];

    int tid = threadIdx.x;
    int i = blockIdx.x * *d_NUM_ION + threadIdx.x;
    int blockSize = blockDim.x;

    // Add data from all of the blocks of ions into first block
    while (i < (blockIdx.x + 1) * *d_NUM_ION)
    {
        sumData[tid].x = d_accDustIon[i].x + d_accDustIon[i + blockSize].x;
        sumData[tid].y = d_accDustIon[i].y + d_accDustIon[i + blockSize].y;
        sumData[tid].z = d_accDustIon[i].z + d_accDustIon[i + blockSize].z;
        i += 2 * blockSize;
    }

    __syncthreads();

    if (tid < 512)
    {
        sumData[tid].x += sumData[tid + 512].x;
        sumData[tid].y += sumData[tid + 512].y;
        sumData[tid].z += sumData[tid + 512].z;
    }
    __syncthreads();
    if (tid < 256)
    {
        sumData[tid].x += sumData[tid + 256].x;
        sumData[tid].y += sumData[tid + 256].y;
        sumData[tid].z += sumData[tid + 256].z;
    }
    __syncthreads();
    if (tid < 128)
    {
        sumData[tid].x += sumData[tid + 128].x;
        sumData[tid].y += sumData[tid + 128].y;
        sumData[tid].z += sumData[tid + 128].z;
    }
    __syncthreads();
    if (tid < 64)
    {
        sumData[tid].x += sumData[tid + 64].x;
        sumData[tid].y += sumData[tid + 64].y;
        sumData[tid].z += sumData[tid + 64].z;
    }
    __syncthreads();

    if (tid < 32)
    {
        sumData[tid].x += sumData[tid + 32].x;
        sumData[tid].y += sumData[tid + 32].y;
        sumData[tid].z += sumData[tid + 32].z;
        sumData[tid].x += sumData[tid + 16].x;
        sumData[tid].y += sumData[tid + 16].y;
        sumData[tid].z += sumData[tid + 16].z;
        sumData[tid].x += sumData[tid + 8].x;
        sumData[tid].y += sumData[tid + 8].y;
        sumData[tid].z += sumData[tid + 8].z;
        sumData[tid].x += sumData[tid + 4].x;
        sumData[tid].y += sumData[tid + 4].y;
        sumData[tid].z += sumData[tid + 4].z;
        sumData[tid].x += sumData[tid + 2].x;
        sumData[tid].y += sumData[tid + 2].y;
        sumData[tid].z += sumData[tid + 2].z;
        sumData[tid].x += sumData[tid + 1].x;
        sumData[tid].y += sumData[tid + 1].y;
        sumData[tid].z += sumData[tid + 1].z;
    }

    if (tid == 0)
    {
        d_accDustIon[blockIdx.x * *d_NUM_ION].x = sumData[0].x;
        d_accDustIon[blockIdx.x * *d_NUM_ION].y = sumData[0].y;
        d_accDustIon[blockIdx.x * *d_NUM_ION].z = sumData[0].z;
    }
}

/*
 * Name: zeroDustIonAcc()
 *
 * Description:
 *	Zeros the forces on each dust particle due to each ion.
 *
 * Inputs:
 *	d_accDustIon: dust accleration due to each dust-ion pair
 *	d_NUM_DUST: the number of dust particles
 *	d_NUM_ION: the number of ions
 *
 * Output (void):
 *	d_accDustIon: is set to zero
 *
 * Assumptions:
 *	The number of ions is a multiple of the block size
 *
 * Includes:
 *	cuda_runtime.h
 * 	device_launch_parameters.h
 *
 */

__global__ void zeroDustIonAcc_103(float4 *d_accDustIon, int *const d_NUM_DUST,
                                   int *const d_NUM_ION)
{

    int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    int index = threadID;
    for (int i = 0; i < *d_NUM_DUST; i++)
    {
        d_accDustIon[index].x = 0;
        d_accDustIon[index].y = 0;
        d_accDustIon[index].z = 0;

        index += *d_NUM_ION;
    }
}
