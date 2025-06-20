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
*   calcExtrnElcAcc_102()
*   calcExtrnElcAccCyl_102()
*	calcIonDensityPotential_102()
    calcIonAccels_102() -- accel from ions inside and outside boundaries, Efield
*
*/

// header file
#include "IonWake_102_ionAcc.h"

/*
 * Name: calcIonAccels_102
 * Created: 11/20/2020
 *
 * Editors
 *	Name: Lorin Matthews
 *	Contact: Lorin_Matthews@baylor.edu
 *
 * Description:
 *	Calculates the accelerations due to ion-ion
 *	interactions modeled as Yukawa particles, using both ions
 *	inside the simulation and force from ions outside simulation
 * 	regtion (using table lookup).  Also add in acceleration from a
 *	constant external electric field (in z-direction).
 *
 * Input:
 *	d_posIon: the positions and charges of the ions
 *	d_accIon: the accelerations of the ions
 *	d_NUM_ION: the number of ions
 *	d_SOFT_RAD_SQRD: the squared softening radius squared
 *	d_ION_ION_ACC_MULT: a constant multiplier for the yakawa interaction
 *	d_INV_DEBYE: the inverse of the debye
 *	d_Q_DIV_M: charge to mass ratio of ions
 *	d_HT_CYL: half of the cylinder height
 *	d_Vout: table with potentials at points in rz-plane from outside ions
 *	d_NUMR: number of columns in table (radial direction)
 *	d_RESZ: number of rows in table (axial direction)
 *	d_dz: distance increment in z-direction
 *	d_dr: distance increment in radial direction
 *	d_E_FIELD: electric field in the z-direction
 *	E_direction: +/- z, for alternating DC E-field
 *	plasma_counter: used to increment evolving boundary conditions
 *	GEOMETRY: spherical or cylindrical simulation region
 *	d_EXTERN_ELC_MULT: used in calculating outside force for spherical boundary
 *	d_INV_DEBYE: used in calculation of outside force for spherical boundary
 *
 * Output (void):
 *	d_accIon: the acceleration due to all of the other ions
 *		is added to the initial ion acceleration
 *
 * Assumptions:
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
__global__ void calcIonAccels_102(float4 *d_posIon, float4 *d_velIon, float4 *d_accIon,
                                  int *const d_NUM_ION, float *const d_SOFT_RAD_SQRD,
                                  float *const d_ION_ION_ACC_MULT, float *const d_INV_DEBYE,
                                  float *d_Q_DIV_M, float *const d_HT_CYL, float *d_outside_coeff,
                                  int *d_NUMR, int *d_RESZ, float *d_dz, float *d_dr,
                                  float *d_E_FIELD, float *d_E_FIELDR, int E_dir,
                                  float *d_B_FIELD_X, float *d_B_FIELD_Y, float *d_B_FIELD_Z,
                                  int plasma_counter, int GEOMETRY, float *const d_EXTERN_ELC_MULT, int *const d_N_COEFFS)
{
    // index of the current ion
    int ID = blockIdx.x * blockDim.x + threadIdx.x;

    if (ID >= *d_NUM_ION)
    {
        return; // out of bounds
    }

    // ****  Acceleration due to ions inside cylinder: N-body calculation *** //

    // initialize variables
    float4 accCrntIon = {0, 0, 0, 0};

    // position of current ion
    float4 posIon = d_posIon[ID];

    // allocate shared memory
    extern __shared__ float4 sharedPos[];

    // loop over all of the ions by using tiles. Where each tile is a section
    // of the ions that is loaded into shared memory. Each tile consists of
    // as many ions as the block size. Each thread is responsible for loading
    // one ion position for the tile.
    for (int tileOffset = 0; tileOffset < *d_NUM_ION; tileOffset += blockDim.x)
    {
        // the index of the ion for the thread to load
        // for the current tile
        int tileThreadID = tileOffset + threadIdx.x;

        if (tileThreadID < *d_NUM_ION)
        {
            //  load in an ion position and charge
            sharedPos[threadIdx.x] = d_posIon[tileThreadID];
        }

        // wait for all threads to load the current position
        __syncthreads();

        //  loop over all of the ions loaded in the tile
        for (int h = 0; h < blockDim.x; h++)
        {
            // calculate the distance between the ion in shared
            // memory and the current thread's ion
            float3 distance;
            distance.x = posIon.x - sharedPos[h].x;
            distance.y = posIon.y - sharedPos[h].y;
            distance.z = posIon.z - sharedPos[h].z;

            // find the squared distance between the ions
            float distSquared = __fmaf_rn(distance.x, distance.x, __fmaf_rn(distance.y, distance.y, __fmul_rn(distance.z, distance.z)));

            // exclude the self-interaction
            if (distSquared == 0)
            {
                // if the ions are the same, there is no force
                continue;
            }

            //  find the inverse of the squared root of the squared distance. invDistance = 1/√r²
            float invDistance = rsqrtf(distSquared);

            // find the distance between the ions. r = r² * 1/√r²(it is faster than sqrt)
            float hardDistance = __fmul_rn(distSquared, invDistance);

            // define the soft distance to avoid singularity
            float softDistance = hardDistance + 1e-9f;

            // find the inverse of the soft distance
            float invSoftDistance = 1.0f / softDistance;

            // find the inverse of the soft distance cubed
            float invSoftDistance3 = __fmul_rn(invSoftDistance, __fmul_rn(invSoftDistance, invSoftDistance));

            // define a coefficient to write shorter expressions
            float coeff = __fmul_rn(hardDistance, *d_INV_DEBYE);

            // get the exponential term
            float expTerm = __expf(-coeff);

            // find the term: (coeff+1) * exp(-coeff)
            float coeff2 = __fmaf_rn(coeff + 1.0f, expTerm, 0.0f);

            // linForce = w * (coeff+1) * exp(-coeff) / r^3
            float linForce = __fmul_rn(sharedPos[h].w, __fmul_rn(coeff2, invSoftDistance3));

            // add the acceleration to the current ion's acceleration
            accCrntIon.x = __fmaf_rn(linForce, distance.x, accCrntIon.x);
            accCrntIon.y = __fmaf_rn(linForce, distance.y, accCrntIon.y);
            accCrntIon.z = __fmaf_rn(linForce, distance.z, accCrntIon.z);
        } // end loop over ion in tile

        // wait for all threads to finish calculations
        __syncthreads();
    } // end loop over tiles

    // scale the acceleration
    accCrntIon.x = __fmul_rn(*d_ION_ION_ACC_MULT, accCrntIon.x);
    accCrntIon.y = __fmul_rn(*d_ION_ION_ACC_MULT, accCrntIon.y);
    accCrntIon.z = __fmul_rn(*d_ION_ION_ACC_MULT, accCrntIon.z);

    // *** forces from ions outside the boundary *** //
    if (GEOMETRY == 0)
    {
        // Spherical boundary conditions
        //  get the radius of the ion from the center of the
        //  simulation sphere. The center is assumed to be (0,0,0)
        float rad =
            __fsqrt_rn((posIon.x * posIon.x) + (posIon.y * posIon.y) +
                       (posIon.z * posIon.z));

        // calculate an intermediate value for use in the
        // acceleration calculation
        float intrmed = rad * *d_INV_DEBYE;

        // calculate a scalar value for the acceleration.
        // to get the acceleration for the ion, multiply the
        // scalar value by the vector distance to the center
        // of the simulation sphere
        float linForce = *d_EXTERN_ELC_MULT * (sinhf(intrmed) - (coshf(intrmed) * intrmed)) /
                         (intrmed * intrmed * rad);

        // multiply by the vector distance to the center of
        // the simulation radius and add it to the ion
        // acceleration
        d_accIon[ID].x += posIon.x * linForce;
        d_accIon[ID].y += posIon.y * linForce;
        d_accIon[ID].z += posIon.z * linForce;
    }
    else if (GEOMETRY == 1)
    {
        // * Force from ions outside cylinder * //
        // local variables
        int page_coeff = __fmul_rn(plasma_counter, *d_N_COEFFS);

        // get the radius of the ion from the center axis of the
        // simulation cylinder. The center is assumed to be (0,0,z)
        float r =
            __fsqrt_rn(__fmaf_rn(posIon.x, posIon.x, __fmul_rn(posIon.y, posIon.y)));

        // get the z position of the ion
        float z = posIon.z;
        float r2 = r * r;
        float r4 = r2 * r2;
        float z2 = z * z;
        float z4 = z2 * z2;

        // using the read only cache
        float outside_coeff1 = __ldg(&d_outside_coeff[page_coeff + 1]);
        float outside_coeff2 = __ldg(&d_outside_coeff[page_coeff + 2]);
        float outside_coeff3 = __ldg(&d_outside_coeff[page_coeff + 3]);
        float outside_coeff4 = __ldg(&d_outside_coeff[page_coeff + 4]);
        float outside_coeff5 = __ldg(&d_outside_coeff[page_coeff + 5]);
        float outside_coeff6 = __ldg(&d_outside_coeff[page_coeff + 6]);
        float outside_coeff7 = __ldg(&d_outside_coeff[page_coeff + 7]);
        float outside_coeff8 = __ldg(&d_outside_coeff[page_coeff + 8]);
        float outside_coeff9 = __ldg(&d_outside_coeff[page_coeff + 9]);
        float outside_coeff10 = __ldg(&d_outside_coeff[page_coeff + 10]);
        float outside_coeff11 = __ldg(&d_outside_coeff[page_coeff + 11]);
        float outside_coeff12 = __ldg(&d_outside_coeff[page_coeff + 12]);
        float outside_coeff13 = __ldg(&d_outside_coeff[page_coeff + 13]);
        float outside_coeff14 = __ldg(&d_outside_coeff[page_coeff + 14]);
        float outside_coeff15 = __ldg(&d_outside_coeff[page_coeff + 15]);
        float outside_coeff16 = __ldg(&d_outside_coeff[page_coeff + 16]);
        float outside_coeff17 = __ldg(&d_outside_coeff[page_coeff + 17]);
        float outside_coeff18 = __ldg(&d_outside_coeff[page_coeff + 18]);
        float outside_coeff19 = __ldg(&d_outside_coeff[page_coeff + 19]);
        float outside_coeff20 = __ldg(&d_outside_coeff[page_coeff + 20]);
        float outside_coeff21 = __ldg(&d_outside_coeff[page_coeff + 21]);
        float outside_coeff22 = __ldg(&d_outside_coeff[page_coeff + 22]);
        float outside_coeff23 = __ldg(&d_outside_coeff[page_coeff + 23]);
        float outside_coeff24 = __ldg(&d_outside_coeff[page_coeff + 24]);

        float Er = (outside_coeff1 * 1.0f) +                // independent term
                   (outside_coeff2 * r * 2.0f) +            // r
                   (outside_coeff4 * r2 * 3.0f) +           // r^2
                   (outside_coeff5 * z2 * 1.0f) +           // z^2
                   (outside_coeff6 * r2 * r * 4.0f) +       // r^3
                   (outside_coeff7 * z2 * r * 2.0f) +       // z^2 r
                   (outside_coeff9 * r4 * 5.0f) +           // r^4
                   (outside_coeff10 * z2 * r2 * 3.0f) +     // z^2 r^2
                   (outside_coeff11 * z4 * 1.0f) +          // z^4
                   (outside_coeff12 * r4 * r * 6.0f) +      // r^5
                   (outside_coeff13 * z2 * r2 * r * 4.0f) + // z^2 r^3
                   (outside_coeff14 * z4 * r * 2.0f) +      // z^4 r
                   (outside_coeff16 * r4 * r2 * 7.0f) +     // r^6
                   (outside_coeff17 * z2 * r4 * 5.0f) +     // z^2 r^4
                   (outside_coeff18 * z4 * r2 * 3.0f) +     // z^4 r^2
                   (outside_coeff19 * z4 * z2 * 1.0f) +     // z^6
                   (outside_coeff20 * r4 * r2 * r * 8.0f) + // r^7
                   (outside_coeff21 * z2 * r4 * r * 6.0f) + // z^2 r^5
                   (outside_coeff22 * z4 * r2 * r * 4.0f) + // z^4 r^3
                   (outside_coeff23 * z4 * z2 * r * 2.0f);  // z^6 r

        float Ez = (outside_coeff3 * 2.0f * z) +                // z
                   (outside_coeff5 * 2.0f * z * r) +            // z r
                   (outside_coeff7 * 2.0f * z * r2) +           // z r^2
                   (outside_coeff8 * 4.0f * z2 * z) +           // z^3
                   (outside_coeff10 * 2.0f * z * r2 * r) +      // z r^3
                   (outside_coeff11 * 4.0f * z2 * z * r) +      // z^3 r
                   (outside_coeff13 * 2.0f * z * r4) +          // z r^4
                   (outside_coeff14 * 4.0f * z2 * z * r2) +     // z^3 r^2
                   (outside_coeff15 * 6.0f * z4 * z) +          // z^5
                   (outside_coeff17 * 2.0f * z * r4 * r) +      // z r^5
                   (outside_coeff18 * 4.0f * z2 * z * r2 * r) + // z^3 r^3
                   (outside_coeff19 * 6.0f * z4 * z * r) +      // z^5 r
                   (outside_coeff21 * 2.0f * z * r4 * r2) +     // z r^6
                   (outside_coeff22 * 4.0f * z2 * z * r4) +     // z^3 r^4
                   (outside_coeff23 * 6.0f * z4 * z * r2) +     // z^5 r^2
                   (outside_coeff24 * 8.0f * z4 * z2 * z);      // z^7

        // Add acceleration of ions from time-evolving radial E field
        // DEBUG -- this was too big, so commented out
        // Er += *d_E_FIELDR / 2000;

        // calculate the term Er * *d_Q_DIV_M / r
        float Er_Q_DIV_M_r = Er * *d_Q_DIV_M / (r + 1e-12f);

        // Scale from radial accel to Cartesian xy-coordinates
        accCrntIon.x = __fmaf_rn(Er_Q_DIV_M_r, posIon.x, accCrntIon.x);
        accCrntIon.y = __fmaf_rn(Er_Q_DIV_M_r, posIon.y, accCrntIon.y);
        accCrntIon.z = __fmaf_rn(Ez, *d_Q_DIV_M, accCrntIon.z);

    } // end if on GEOMETRY

    // *** add acceleration of ions by external electric field *** //
    if (*d_E_FIELD > 0)
    {
        accCrntIon.z = __fmaf_rn(E_dir, __fmul_rn(*d_Q_DIV_M, *d_E_FIELD), accCrntIon.z);
    }
    else
    {
        accCrntIon.z = __fmaf_rn(-E_dir, __fmul_rn(*d_Q_DIV_M, *d_E_FIELD), accCrntIon.z);
    }

    //*** add aceleration of ions by external B_field ***//
    if (*d_B_FIELD_X != 0 || *d_B_FIELD_Y != 0 || *d_B_FIELD_Z != 0)
    {
        // velocity of current ion
        float4 velIon = d_velIon[ID];

        accCrntIon.x = __fmaf_rn(*d_Q_DIV_M, __fmaf_rn(velIon.y, *d_B_FIELD_Z, __fmul_rn(-velIon.z, *d_B_FIELD_Y)), accCrntIon.x);
        accCrntIon.y = __fmaf_rn(*d_Q_DIV_M, __fmaf_rn(velIon.z, *d_B_FIELD_X, __fmul_rn(-velIon.x, *d_B_FIELD_Z)), accCrntIon.y);
        accCrntIon.z = __fmaf_rn(*d_Q_DIV_M, __fmaf_rn(velIon.x, *d_B_FIELD_Y, __fmul_rn(-velIon.y, *d_B_FIELD_X)), accCrntIon.z);
    }

    // save to global memory
    d_accIon[ID] = accCrntIon;
}

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
__global__ void calcIonIonAcc_102(float4 *d_posIon, float4 *d_accIon, int *const d_NUM_ION,
                                  float *const d_SOFT_RAD_SQRD, float *const d_ION_ION_ACC_MULT,
                                  float *const d_INV_DEBYE)
{

    // index of the current ion
    int IDcrntIon = blockIdx.x * blockDim.x + threadIdx.x;

    // initialize variables
    float3 dist;
    float3 accCrntIon = {0, 0, 0};
    float distSquared;
    float hardDist;
    float softDist;
    float linForce;
    int tileThreadID;

    d_accIon[IDcrntIon].x = 0;
    d_accIon[IDcrntIon].y = 0;
    d_accIon[IDcrntIon].z = 0;

    // allocate shared memory
    extern __shared__ float4 sharedPos[];

    // loop over all of the ions by using tiles. Where each tile is a section
    // of the ions that is loaded into shared memory. Each tile consists of
    // as many ions as the block size. Each thread is responsible for loading
    // one ion position for the tile.
    for (int tileOffset = 0; tileOffset < *d_NUM_ION; tileOffset += blockDim.x)
    {
        // the index of the ion for the thread to load
        // for the current tile
        tileThreadID = tileOffset + threadIdx.x;

        // load in an ion position and charge
        sharedPos[threadIdx.x].x = d_posIon[tileThreadID].x;
        sharedPos[threadIdx.x].y = d_posIon[tileThreadID].y;
        sharedPos[threadIdx.x].z = d_posIon[tileThreadID].z;
        sharedPos[threadIdx.x].w = d_posIon[tileThreadID].w;

        // wait for all threads to load the current position
        __syncthreads();

        // DEBUGGING //
        /*
         * // PTX code used to access shared memory sizes
         * // which are save to "ret"
         * unsigned ret;
         * asm volatile ("mov.u32 %0, %total_smem_size;" : "=r"(ret));
         */

        // loop over all of the ions loaded in the tile
        for (int h = 0; h < blockDim.x; h++)
        {

            // calculate the distance between the ion in shared
            // memory and the current thread's ion
            dist.x = d_posIon[IDcrntIon].x - sharedPos[h].x;
            dist.y = d_posIon[IDcrntIon].y - sharedPos[h].y;
            dist.z = d_posIon[IDcrntIon].z - sharedPos[h].z;

            // calculate the distance squared
            distSquared = dist.x * dist.x + dist.y * dist.y + dist.z * dist.z;

            // calculate the hard distance
            hardDist = __fsqrt_rn(distSquared);

            // calculate the soft distance
            softDist = __fsqrt_rn(distSquared + *d_SOFT_RAD_SQRD);

            // calculate a scalar intermediate
            linForce = *d_ION_ION_ACC_MULT * sharedPos[h].w * (1.0 + (hardDist * *d_INV_DEBYE)) *
                       __expf(-hardDist * *d_INV_DEBYE) / (softDist * softDist * softDist);

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
    d_accIon[IDcrntIon].w = 0.0;
}

/*
 * Name: calcIonDustAcc_102
 * Created: 6/13/2017
 * last edit: 09/10/2020
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
 *	d_posIon: the positions of the ions, charge in 4th pos'n
 *	d_accIon: the accelerations of the ions
 *	d_posDust: the dust particle positions, charge in 4th pos'n
 *	d_NUM_ION: the number of ions
 *	d_NUM_DUST: the number of dust particles
 *	d_SOFT_RAD_SQRD: the squared softening radius squared
 *	d_ION_DUST_ACC_MULT: a constant multiplier for the ion-dust interaction
 *
 * Output (void):
 *	d_accIon: the acceleration due to all the dust particles
 *		is added to the initial ion acceleration
 *	d_minDistDust: the distance to closest dust particle
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
__global__ void calcIonDustAcc_102(float4 *d_posIon, float4 *d_accIon, float4 *d_posDust,
                                   int *const d_NUM_ION, int *const d_NUM_DUST,
                                   float *const d_SOFT_RAD_SQRD, float *const d_ION_DUST_ACC_MULT,
                                   float *d_minDistDust)
{

    // index of the current ion
    int IDcrntIon = blockIdx.x * blockDim.x + threadIdx.x;

    // allocate variables
    float3 dist;
    float distSquared;
    float hardDist;
    float linForce;

    // Set initial minimum distance to dust particle to large number
    float min_dist = 1000;

    // reset the accelerations
    d_accIon[IDcrntIon].x = 0;
    d_accIon[IDcrntIon].y = 0;
    d_accIon[IDcrntIon].z = 0;
    d_accIon[IDcrntIon].w = 0.0;

    // loop over all of the dust particles
    for (int h = 0; h < *d_NUM_DUST; h++)
    {

        // calculate the distance between the ion in shared
        // memory and the current thread's ion
        dist.x = d_posIon[IDcrntIon].x - d_posDust[h].x;
        dist.y = d_posIon[IDcrntIon].y - d_posDust[h].y;
        dist.z = d_posIon[IDcrntIon].z - d_posDust[h].z;

        // calculate the distance squared
        distSquared = dist.x * dist.x + dist.y * dist.y + dist.z * dist.z;

        // calculate the hard distance
        hardDist = __fsqrt_rn(distSquared);

        // calculate a scalar intermediate
        linForce = *d_ION_DUST_ACC_MULT * d_posDust[h].w / (hardDist * hardDist * hardDist);

        // add the acceleration to the current ion's acceleration
        d_accIon[IDcrntIon].x += linForce * dist.x;
        d_accIon[IDcrntIon].y += linForce * dist.y;
        d_accIon[IDcrntIon].z += linForce * dist.z;

        // save the distance to the closest dust particle
        if (hardDist < min_dist)
        {
            min_dist = hardDist;
        }

    } // end loop over dust

    d_minDistDust[IDcrntIon] = min_dist;
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
__global__ void calcExtrnElcAcc_102(float4 *d_accIon, float4 *d_posIon,
                                    float *const d_EXTERN_ELC_MULT, float *const d_INV_DEBYE)
{

    // the thread ID
    int ID = blockIdx.x * blockDim.x + threadIdx.x;

    // get the radius of the ion from the center of the
    // simulation sphere. The center is assumed to be (0,0,0)
    float rad = __fsqrt_rn((d_posIon[ID].x * d_posIon[ID].x) + (d_posIon[ID].y * d_posIon[ID].y) +
                           (d_posIon[ID].z * d_posIon[ID].z));

    // calculate an intermediate value for use in the
    // acceleration calculation
    float intrmed = rad * *d_INV_DEBYE;

    // calculate a scaler value for the acceleration.
    // to get the acceleration for the ion, multiply the
    // scaler value by the vector distance to the center
    // of the simulation sphere
    float linForce = *d_EXTERN_ELC_MULT * (sinhf(intrmed) - (coshf(intrmed) * intrmed)) /
                     (intrmed * intrmed * rad);

    // multiply by the vector distance to the center of
    // the simulation radius and add it to the ion
    // acceleration
    d_accIon[ID].x += d_posIon[ID].x * linForce;
    d_accIon[ID].y += d_posIon[ID].y * linForce;
    d_accIon[ID].z += d_posIon[ID].z * linForce;
    d_accIon[ID].w = 0.0;
}

/*
* Name: calcExtrnElcAccCyl_102
* Created: 11/18/2017
* last edit: 10/13/2020
*
* Editors
*	Name: Lorin Matthews
*	Contact: Lorin_Matthews@baylor.edu
*	last edit: 10/13/2020
*
* Description:
*	calculates the acceleration on the ions due to the electric field created
*   by the ions outside of a simulation cylinder. implemented using a table lookup
*	from the potential of ions from outside the cylinder.  Note that the
*   electric field is the gradient of the potential.  However, the potential of a
*   cylindrical cavity is the negative of the potential of the cylinder of ions,
*   which is what is calculated by boundaryEField. The two negatives cancel.

*
* Input:
*	d_accIon: ion accelerations
*	d_posIon: ion positions and charges
*	d_Q_DIV_M:  charge to mass ratio
*   d_HT_CYL: half the cylinder height
*	d_Vout: potential of ions outside the simulation cylinder
*   d_NUMR: number of grid points in r-direction
*   d_dz: increment in z between grid points
*   d_dr: increment in r between grid points
*	d_Esheath: sheath/DC electric field (z-direction)
*	E_dir: direction of polarity-switched DC field
*	plasma_counter: index for evolving plasma conditions
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
__global__ void calcExtrnElcAccCyl_102(float4 *d_accIon, float4 *d_posIon, float *d_Q_DIV_M,
                                       float *const d_HT_CYL, float *d_Vout, int *d_NUMR,
                                       int *d_RESZ, float *d_dz, float *d_dr, float *d_Esheath,
                                       int E_dir, int plasma_counter)
{

    // the thread ID
    int ID = blockIdx.x * blockDim.x + threadIdx.x;

    // local variables
    float temp, x1, frac_r, z1, frac_z;
    int page;
    int pt0, pt1, pt2, pt3, pt4, pt5, pt6, pt7, ptA, ptB, ptC, ptD;
    float Ex, Ez;

    // determine the offset for table lookup based on the plasma condition
    page = plasma_counter * *d_NUMR * *d_RESZ;

    // get the radius of the ion from the center axis of the
    // simulation cylinder. The center is assumed to be (0,0,z)
    float rad = __fsqrt_rn((d_posIon[ID].x * d_posIon[ID].x) + (d_posIon[ID].y * d_posIon[ID].y));

    // get the z position of the ion
    float z = d_posIon[ID].z;

    // find the column index of xg to left of r
    //  The table limits run from -2dr:dr:RAD_CYL, so
    //  add one to the table index for +r.
    temp = rad / *d_dr + 1;
    x1 = static_cast<int>(temp);
    // fractional remainder
    frac_r = temp - x1;

    // Add HT_CYL to pos.z to make it a positive distance.
    // Find the row index of zg below z.
    temp = (z + *d_HT_CYL) / *d_dz;
    // integer part -- tells how many rows up
    z1 = static_cast<int>(temp);
    // fractional remainder
    frac_z = temp - z1;

    // Find the four grid points surrounding the ion position. Will also
    // need the grid points to the left and right of these for Ex and
    //  the points above and below these for Ez.
    // First, the gridpoints on the row below pos_Ion
    pt1 = x1 + z1 * *d_NUMR; // pt below and to the left
    pt2 = pt1 + 1;           // pt below and to the right
    pt0 = pt1 - 1;           // pt to the left of pt1
    pt3 = pt1 + 2;           // pt to the right of pt2
    // Next, the grid points on the row above pos_Ion
    pt4 = pt0 + *d_NUMR;
    pt5 = pt1 + *d_NUMR;
    pt6 = pt2 + *d_NUMR;
    pt7 = pt3 + *d_NUMR;
    // Points above 5 & 6 and below 1 & 2
    ptA = pt5 + *d_NUMR;
    ptB = pt6 + *d_NUMR;
    ptC = pt1 - *d_NUMR;
    ptD = pt2 - *d_NUMR;

    // Calculate Ex at the posIon by taking the gradient of the potential
    // known on the grid points.
    // Treat special cases for positions which are on edges of grid.
    if (x1 == 0)
    {
        // on the left edge
        // This case should never be met since the minimum r-index
        // is now 1.
        Ex = ((d_Vout[page + pt2] - d_Vout[page + pt1]) * (1.0 - frac_z) +
              (d_Vout[page + pt6] - d_Vout[page + pt5]) * frac_z) /
             (*d_dr);
    }
    else if (x1 == (*d_NUMR - 2))
    {
        // on the right edge
        Ex = ((d_Vout[page + pt2] - d_Vout[page + pt1]) * (1.0 - frac_z) +
              (d_Vout[page + pt6] - d_Vout[page + pt5]) * frac_z) /
             (*d_dr);
    }
    else
    {
        // in the middle of the grid
        // Determine the electric field at the four points
        // surrounding ionPos.
        // Ex1 = -(*d_Vout[pt2] - *d_Vout[pt0]) / (2 * *d_dr);
        // Ex2 = -(*d_Vout[pt3] - *d_Vout[pt1]) / (2 * *d_dr);
        // Ex3 = -(*d_Vout[pt6] - *d_Vout[pt4]) / (2 * *d_dr);
        // Ex4 = -(*d_Vout[pt7] - *d_Vout[pt5]) / (2 * *d_dr);

        // Use an areal-weighting scheme to perform
        // the 2D table lookup.
        Ex = ((d_Vout[page + pt2] - d_Vout[page + pt0]) * (1.0 - frac_r) * (1.0 - frac_z) +
              (d_Vout[page + pt3] - d_Vout[page + pt1]) * frac_r * (1.0 - frac_z) +
              (d_Vout[page + pt6] - d_Vout[page + pt4]) * (1.0 - frac_r) * frac_z +
              (d_Vout[page + pt7] - d_Vout[page + pt5]) * frac_r * frac_z) /
             (2.0 * *d_dr);
    }

    // Calculate Ez at the posIon by taking the gradient of the potential
    // known on the grid points.
    // Treat special cases for positions which are on edges of grid.
    if (z1 == 0)
    {
        // on the top edge
        Ez = ((d_Vout[page + pt5] - d_Vout[page + pt1]) * (1.0 - frac_z) +
              (d_Vout[page + pt6] - d_Vout[page + pt2]) * frac_z) /
             (*d_dz);
    }
    else if (z1 == (*d_RESZ - 2))
    {
        // on the bottom edge
        Ez = ((d_Vout[page + pt5] - d_Vout[page + pt1]) * (1.0 - frac_z) +
              (d_Vout[page + pt6] - d_Vout[page + pt2]) * frac_z) /
             (*d_dz);
    }
    else
    {
        // in the middle of the grid
        // Determine the electric field at the four points
        // surrounding ionPos.
        // Ez1 = -(*d_Vout[page + pt5] - *d_Vout[page + ptC]) / (2 * *d_dz);
        // Ez2 = -(*d_Vout[page + pt6] - *d_Vout[page + ptD]) / (2 * *d_dz);
        // Ez3 = -(*d_Vout[page + ptA] - *d_Vout[page + pt1]) / (2 * *d_dz);
        // Ez4 = -(*d_Vout[page + ptB] - *d_Vout[page + pt2]) / (2 * *d_dz);

        // Use an areal-weighting scheme to perform
        // the 2D table lookup.
        Ez = ((d_Vout[page + pt5] - d_Vout[page + ptC]) * (1.0 - frac_r) * (1.0 - frac_z) +
              (d_Vout[page + pt6] - d_Vout[page + ptD]) * frac_r * (1.0 - frac_z) +
              (d_Vout[page + ptA] - d_Vout[page + pt1]) * (1.0 - frac_r) * frac_z +
              (d_Vout[page + ptB] - d_Vout[page + pt2]) * frac_r * frac_z) /
             (2.0 * *d_dz);
    }

    // scale from radial distance to cartesian coordinates
    d_accIon[ID].x += Ex * *d_Q_DIV_M * d_posIon[ID].x / rad;
    d_accIon[ID].y += Ex * *d_Q_DIV_M * d_posIon[ID].y / rad;

    // add in acceleration by Ez
    d_accIon[ID].z += Ez * *d_Q_DIV_M;

    // add acceleration of ions by external electric field
    d_accIon[ID].z += E_dir * *d_Q_DIV_M * *d_Esheath;
    d_accIon[ID].w = 0.0;
}

/*
* Name: calcIonDensityPotential_102
* Created: 5/4/2018
* Last Modified: 9/10/2020
*
* Editors
*	Name: Lorin Matthews
*	Contact: Lorin_Matthews@baylor.edu
*	last edit: 9/9/2020  GridPos now float2,replaced float3 with float4

* Description:
*	Calculates electric potential from ions at points on grid in
* 	the xz-plane.  Also calculates the number density at each grid
*	point by counting the number of ions in a sphere of radius r_dens
* 	centered at each grid point.
*
* Input:
*	d_posIion: ion positions
*	d_gridPos: the grid points in xz-plane
*	//d_ION_POTENTIAL_MULT
*	d_COULOMB_CONST
*	d_INV_DEBYE
*
* Output (void):
*	d_ionPotential: potential at each grid point
*	d_ionDenisty: ion number density at each grid point
*
* Assumptions:
*   The number of grid points is a multiple of the block size?????
*
* Includes:
*	cuda_runtime.h
*	device_launch_parameters.h
*
*/

__global__ void calcIonDensityPotential_102(const int NUM_GRID_PTS, float2 *d_gridPos, float4 *d_posIon,
                                            float *const d_COULOMB_CONST, float *const d_INV_DEBYE,
                                            int *const d_NUM_ION, float *d_ionPotentialXZ,
                                            float *d_ionDensityXZ)
{
    //  This is done for every grid point, so threadIdx, blockDim, and blockIdx
    // need to be calculated based on the number of grid points.
    // grid point ID
    int IDgrid = threadIdx.x + blockDim.x * blockIdx.x;

    // check if the thread is out of bounds
    if (IDgrid >= NUM_GRID_PTS)
    {
        return; // out of bounds
    }

    // get the current grid point coordinates
    float2 gridPos = d_gridPos[IDgrid];

    // initialize variables
    float potCrntGridXZ = 0;
    // float potCrntGridYZ = 0;
    float densCrntGridXZ = 0;
    // float densCrntGridYZ = 0;
    int tileThreadID;
    float r_dens = 1.0f / (6.0f * *d_INV_DEBYE);
    float volume = 4.0f / 3.0f * 3.141593f * r_dens * r_dens * r_dens;
    float invVolume = 1.0f / volume;

    // allocate shared memory
    extern __shared__ float4 sharedPos[];

    // loop over all of the ions by using tiles, where each tile is a section
    // of the ions that is loaded into shared memory. Each tile consists of
    // as many ions as the block size. Each thread is responsible for loading
    // one ion position for the tile.
    for (int tileOffset = 0; tileOffset < *d_NUM_ION; tileOffset += blockDim.x)
    {
        // the index of the ion for the thread to load
        // for the current tile
        tileThreadID = tileOffset + threadIdx.x;

        // load in an ion position
        if (tileThreadID < *d_NUM_ION)
        {
            sharedPos[threadIdx.x] = d_posIon[tileThreadID];
        }

        // wait for all threads to load the current position
        __syncthreads();

        // loop over all of the ions loaded in the tile
        for (int h = 0; h < blockDim.x; h++)
        {
            // calculate the distance between the ion in shared
            // memory and the current grid point. It is possible to calculate the potential and the
            // density in a plane different than xz but it is necessary to calculate the below
            // distance correctly.

            // get the charge of the current ion
            float ionCharge = sharedPos[h].w;

            // Calculating the ion potential and density in the XZ plane
            float3 distXZ;
            distXZ.x = gridPos.x - sharedPos[h].x;
            distXZ.y = -sharedPos[h].y;
            distXZ.z = gridPos.y - sharedPos[h].z;

            // calculate the distance squared in the XZ plane between the ion and the current grid point
            float distSquaredXZ = __fmaf_rn(distXZ.x, distXZ.x, __fmaf_rn(distXZ.y, distXZ.y, __fmul_rn(distXZ.z, distXZ.z)));

            //  find the inverse of the squared root of the squared distance in the XZ plane. invDistanceXZ = 1/√r²
            float invDistanceXZ = rsqrtf(distSquaredXZ);

            // find the distance between the ion and the current grid point. r = r² * 1/√r²(it is faster than sqrt)
            float hardDistanceXZ = __fmul_rn(distSquaredXZ, invDistanceXZ);

            // define the soft distance to avoid singularity
            float softDistanceXZ = hardDistanceXZ + 1e-9f;

            // find the inverse of the soft distance
            float invSoftDistanceXZ = 1.0f / softDistanceXZ;

            // define a coefficient to write shorter expressions
            float coeff = __fmul_rn(hardDistanceXZ, *d_INV_DEBYE);

            // get the exponential term
            float expTerm = __expf(-coeff);

            // Calculate the potential
            potCrntGridXZ = __fmaf_rn(ionCharge, __fmul_rn(expTerm, invSoftDistanceXZ), potCrntGridXZ);

            if (hardDistanceXZ <= r_dens)
            {
                // Update density.  sharedPos[h].w is the total charge on
                // this super-ion=SUPER_ION_MULT*CHARGE_SINGLE_ION. The density
                // is SUPER_ION_MULT*DEN_IONS. Need to divide by
                // CHARGE_SINGLE_ION before printing.
                densCrntGridXZ += ionCharge;
            }

        } // end loop over ion in tile

        // wait for all threads to finish calculations
        __syncthreads();
    } // end loop over tiles

    // save to global memory
    d_ionPotentialXZ[IDgrid] = __fmaf_rn(*d_COULOMB_CONST, potCrntGridXZ, d_ionPotentialXZ[IDgrid]);
    d_ionDensityXZ[IDgrid] = __fmaf_rn(densCrntGridXZ, invVolume, d_ionDensityXZ[IDgrid]);
}

/*
 * Name: zeroIonDensityPotential_102
 * Created: 5/21/2018
 * Last Modified: 5/21/2018
 *
 * Editors
 *	Name: Lorin Matthews
 *	Contact: Lorin_Matthews@baylor.edu
 *	last edit: 5/21/2018
 *
 * Description:
 *	Zeros electric potential from ions at points on grid in
 * 	the xz-plane.  Also zeros the number density at each grid pt.
 *
 * Input:
 *	d_ionPotential: potential at each grid point
 *	d_ionDenisty: ion number density at each grid point
 *
 * Output (void):
 *	d_ionPotential: potential at each grid point
 *	d_ionDenisty: ion number density at each grid point
 *
 * Assumptions:
 *   The number of grid points is a multiple of the block size
 *
 * Includes:
 *	cuda_runtime.h
 *	device_launch_parameters.h
 *
 */

__global__ void zeroIonDensityPotential_102(float *d_ionPotential, float *d_ionDensity)
{

    //  This is done for every grid point, so threadIdx, blockDim, and blockIdx
    // need to be calculated based on the number of grid points.
    // grid point ID
    int IDgrid = threadIdx.x + blockDim.x * blockIdx.x;

    d_ionDensity[IDgrid] = 0;
    d_ionPotential[IDgrid] = 0;
}
