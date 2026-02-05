/*
 * Project: IonWake
 * File Type: function library implementation
 * File Name: IonWake_100_integrate.cu
 *
 * Created: 6/13/2017
 * Last Modified: 9/10/2020
 *
 * Description:
 *	Includes time step integrators
 *
 * Functions:
 *	kick_100()
 *	drift_100()
 *	select_100()
 *	KDK_100()
 * Local Functions
 *   drift_dev()
 *   kick_dev()
 *	checkIonDustbounds_100_dev()
 *	checkIonSphereBounds_100_dev()
 *	checkIonCylinderBounds_100_dev()
 *	calcIonDustAcc_100_dev()
 *
 */

// header file
#include "IonWake_100_integrate.h"

/*
 * Name: kick_100
 * Created: 6/13/2017
 * last edit: 01/24/2018
 *
 * Editors
 *	Name: Dustin Sanford
 *	Contact: Dustin_Sanford@baylor.edu
 *	last edit: 10/22/2017
 *	Name: Lorin Matthews
 *	Contact: Lorin_Matthews@baylor.edu
 *	last edit: 01/24/2018
 *
 * Description:
 *	Kick for one timestep in a leapfrog integration
 *
 * Input:
 *	vel: velocities
 *	acc: accelerations
 *	d_TIME_STEP: half of a time step
 *
 * Output (void):
 *	vel: updated velocity from the integration
 *	acc: reset to zero
 *
 * Assumptions:
 *	All inputs are real values
 *
 * Includes:
 *	cuda_runtime.h
 *	device_launch_parameters.h
 *
 */
__global__ void kick_100(float4 *vel, float4 *acc, const float *d_TIME_STEP)
{
    // thread ID
    int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    /* calculate velocity
     * v = v0 + dT * a
     *******************
     * v  - velocity
     * v0 - initial velocity
     * dT - time step
     * a  - acceleration
     *******************/
    vel[threadID].x += ((*d_TIME_STEP) * acc[threadID].x);
    vel[threadID].y += ((*d_TIME_STEP) * acc[threadID].y);
    vel[threadID].z += ((*d_TIME_STEP) * acc[threadID].z);

    // reset acceleration
    acc[threadID].x = 0;
    acc[threadID].y = 0;
    acc[threadID].z = 0;
}

/*
 * Name: drift_100
 * Created: 6/13/2017
 * last edit: 01/25/2018
 *
 * Editors
 *	Name: Dustin Sanford
 *	Contact: Dustin_Sanford@baylor.edu
 *	last edit: 10/22/2017
 *	Name: Lorin Matthews
 *	Contact: Lorin_Matthews@baylor.edu
 *	last edit: 01/25/2018
 *
 * Description:
 *	Kick for whole  timestep in a leapfrog integration
 *
 * Input:
 *	pos: positions
 *	vel: velocities
 *	d_HALF_TIME_STEP: half of a time step
 *
 * Output (void):
 *	pos: updated position from the integration
 *
 * Assumptions:
 *	All inputs are real values.
 *
 * Includes:
 *	cuda_runtime.h
 *	device_launch_parameters.h
 *
 */
__global__ void drift_100(float4 *pos, float4 *vel, const float *d_HALF_TIME_STEP)
{
    // thread ID
    int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    /* calculate position
     * x = x0 +  dT/2* v(n+1/2)
     *******************
     * x  - position
     * x0 - initial position
     * dT - time step
     * v  - velocity
     *******************/
    pos[threadID].x += ((*d_HALF_TIME_STEP) * vel[threadID].x);
    pos[threadID].y += ((*d_HALF_TIME_STEP) * vel[threadID].y);
    pos[threadID].z += ((*d_HALF_TIME_STEP) * vel[threadID].z);
}

/*
 * Name: select_100
 * Created: 3/6/2018
 * last edit: 3/6/2018
 *
 * Editors
 *	Name: Lorin Matthews
 *	Contact: Lorin_Matthews@baylor.edu
 *	last edit: 3/06/2018
 *
 * Description:
 *	Select timestep division for adaptive time step
 *
 * Input:
 *	pos: ion positions and charge
 *	posDust: dust positions and charge
 *	vel: ion velocities
 *   minDistDust: distance to closest dust grain
 *   d_RAD_DUST: radius of dust grains
 *	d_TIME_STEP: time step
 *   d_M_FACTOR: constant used in determining time step division
 *   d_MAX_DEPTH: maximum divisions of time step
 *
 * Output (void):
 *	m: the nuspeedber of times timestep is divided by factor of 2
 *   tsFactor: 2^(m-1)
 *
 * Assumptions:
 *	All inputs are real values.
 *
 * Includes:
 *	cuda_runtime.h
 *	device_launch_parameters.h
 *
 */
__global__ void select_100(float4 *d_posIon, float4 *d_posDust, float4 *velIon, float *minDistDust,
                           const float *d_RAD_DUST, const float *d_TIME_STEP,
                           const int *d_MAX_DEPTH, const float *d_M_FACTOR, int *const d_NUM_DUST,
                           int *m, int *timeStepFactor)
{

    // thread ID
    int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    // initialize variables
    float v2;
    float speed;
    int mtemp;
    int tsf;
    float3 dist;
    float distSquared;
    float hardDist;
    float min_dist = 1000; // Set at initial large value

    // loop over all of the dust particles to find closest dust to the ion
    for (int h = 0; h < *d_NUM_DUST; h++)
    {

        // calculate the distance between the ion in shared memory
        // and the current thread's ion
        dist.x = d_posIon[threadID].x - d_posDust[h].x;
        dist.y = d_posIon[threadID].y - d_posDust[h].y;
        dist.z = d_posIon[threadID].z - d_posDust[h].z;

        // calculate the distance squared
        distSquared = dist.x * dist.x + dist.y * dist.y + dist.z * dist.z;

        // calculate the hard distance
        hardDist = __fsqrt_rn(distSquared);

        // save the distance to the closest dust particle
        if (hardDist < min_dist)
        {
            min_dist = hardDist;
        }
    }

    //                Calculate Timestep Depth
    /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
     *      m = ceil(ln( M * dT * v / abs(r - R))/ln(2))       *
     * ------------------------------------------------------- *
     * m  - timestep depth                                     *
     * M  - d_M_FACTOR                                         *
     * dT - time step                                          *
     * v  - magnitude of velocity                              *
     * r  - distance to closest dust particle                  *
     * R  - radius of the dust particles                       *
     *                                                         *
     * 30 is a factor which initial tests showed to work well  *
     * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

    // ion speed squared
    v2 = velIon[threadID].x * velIon[threadID].x + velIon[threadID].y * velIon[threadID].y +
         velIon[threadID].z * velIon[threadID].z;

    // ion speed
    speed = __fsqrt_rn(v2);

    // m = ceil(v2)
    // v2 is being used to as an intermediat step in calculating m
    v2 = __logf(*d_M_FACTOR * *d_TIME_STEP * speed / (minDistDust[threadID] - *d_RAD_DUST)) /
         __logf(2.0);

    // timestep depth
    mtemp = ceil(v2);

    // check that the timestep depth is within allowed bounds
    if (mtemp < 0)
    {
        mtemp = 0;
    }
    else if (mtemp > *d_MAX_DEPTH)
    {
        mtemp = *d_MAX_DEPTH;
    }

    // calculate 2^(time step depth)
    tsf = 1;
    for (int i = 0; i < mtemp; i++)
    {
        tsf = tsf * 2;
    }

    // save to global memory
    m[threadID] = mtemp;
    timeStepFactor[threadID] = tsf;
}

/*
 * Name: KDK_100
 * Created: 3/14/2018
 * last edit: 9/10/2020
 *
 * Editors
 *	Name: Lorin Matthews
 *	Contact: Lorin_Matthews@baylor.edu
 *
 * Description:
 *	Performs KDK 2^m times, using the kick just for the ion-dust
 *   dust acceleration.  'm' is a divisor for the time step so that
 *   ions will be accurately captured by dust grains.
 *
 * Input:
 *	pos: positions and charge of ions
 *	vel: velocities of ions
 *	acc: accelerations of ions
 *   d_m: depth for time step divisions
 *	d_tsFactor: divisor for time step
 *	boundsIon -- flags for boundary crossings
 *	d_TIME_STEP: time step
 *   d_RAD_SIM_SQRD -or- d_RAD_CYL_SQRT: simulation bounds
 *   (empty) -or- d_HT_CYL: simulation bounds
 *	d_GEOMETRY (0=spherical or 1=cylindrical)
 *	d_RAD_DUST
 *	d_NUM_DUST
 *   d_posDust position and charge of dust
 *	d_NUM_ION
 *	d_SOFT_RAD_SQRD
 *	d_ION_DUST_ACC_MULT
 *	d_RAD_COLL_MULT
 *
 * Output (void):
 *   pos: updated positions of ions
 *	vel: updated velocities of ions
 *	boundsIon: updated boundary crossings
 *
 * Assumptions:
 *	All inputs are real values
 *
 * Includes:
 *	cuda_runtime.h
 *	device_launch_parameters.h
 *
 */
__global__ void KDK_100(float4 *posIon, float4 *velIon, float4 *accIon, float4 *ionDustAcc,
                        int *d_boundsIon, float *d_minDistDust, const float *d_M_FACTOR,
                        const float *d_TIME_STEP, const int GEOMETRY, const float *d_bndry_sqrd,
                        const float *d_HT_CYL, const float *d_RAD_DUST, const int *d_NUM_DUST,
                        float4 *d_posDust, const int *d_NUM_ION, const float *d_SOFT_RAD_SQRD,
                        const float *d_ION_DUST_ACC_MULT, const float *d_RAD_COLL_MULT)
{

    // thread ID
    int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadID >= *d_NUM_ION)
    {
        return; // out of bounds
    }

    // local variables
    int mtemp;
    float v2, speed;
    int max_depth = 8; // Maximum division of timestep = 2^8
    float timeStep, halfTimeStep;
    int timeStepFactor = 1;

    // Calculate the time step depth.  The timestep will be divided by a multiple
    // of two depending on its speed and distance from dust.
    // m = ceil(ln( M * dT * v / abs(r - R)/ln(2);  tsf = 2^m
    /* ------------------------------------------------------- *
     * m  - timestep depth                                     *
     * M  - d_M_FACTOR                                         *
     * dT - time step                                          *
     * v  - magnitude of velocity                              *
     * r  - distance to closest dust particle                  *
     * R  - radius of the dust particles                       *
     *                                                         *
     * 30 is a factor which initial tests showed to work well  *
     * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

    // Reset the ion bounds flag to 0
    int currentBoundIon = 0;

    // get the position, velocity and accelerations for the current ion
    float4 currentPosIon = posIon[threadID];
    float4 currentVelIon = velIon[threadID];
    float4 currentAccIon = accIon[threadID];
    float4 currentIonDustAcc = ionDustAcc[threadID];

    if (*d_NUM_DUST > 0)
    {
        // ion speed squared
        v2 = __fmaf_rn(currentVelIon.x, currentVelIon.x, __fmaf_rn(currentVelIon.y, currentVelIon.y, currentVelIon.z * currentVelIon.z));

        // ion speed
        speed = __fsqrt_rn(v2);

        // v2 is being used to as an intermediat step in calculating m
        v2 = __log2f(*d_M_FACTOR * *d_TIME_STEP * speed / (d_minDistDust[threadID] - *d_RAD_DUST));

        // timestep depth
        mtemp = ceil(v2);

        // check that the timestep depth is within allowed bounds
        if (mtemp < 0)
        {
            mtemp = 0;
        }
        else if (mtemp > max_depth)
        {
            mtemp = max_depth;
        }

        // calculate 2^(time step depth)
        for (int i = 0; i < mtemp; i++)
        {
            timeStepFactor = timeStepFactor * 2;
        }
    }
    // Kick a one timestep with all forces except ion-dust force
    kick_dev(currentVelIon, currentAccIon, *d_TIME_STEP);

    // Add in accel from dust in smaller steps if needed
    timeStep = *d_TIME_STEP / timeStepFactor;
    halfTimeStep = timeStep * 0.5f;

    if (*d_NUM_DUST > 0)
    {
        // Kick for 1/2 a timestep to get started
        kick_dev(currentVelIon, currentIonDustAcc, halfTimeStep);
    }

    // now do Drift, check, calc_accels, Kick, for tsf = 2^(m-1) times
    int depth = 1;
    while (currentBoundIon == 0 && depth <= timeStepFactor)
    {

        depth++;

        // save the current ion position before the drift
        float4 oldIonPos = currentPosIon;

        // update the current ion position
        drift_dev(currentPosIon, currentVelIon, timeStep);

        // Check outside bounds
        if (GEOMETRY == 0)
        {
            // check if any ions are outside of the simulation sphere
            checkIonSphereBounds_100_dev(currentPosIon, currentBoundIon, d_bndry_sqrd);
        }
        else if (GEOMETRY == 1)
        {
            // check if any ions are outside of the simulation cylinder
            checkIonCylinderBounds_100_dev(currentPosIon, currentBoundIon, d_bndry_sqrd,
                                           d_HT_CYL);
        }

        if (*d_NUM_DUST > 0)
        {
            // check if any ions are inside a dust particle
            checkIonDustBounds_100_dev(currentPosIon, currentBoundIon,
                                       d_RAD_DUST, d_NUM_DUST, d_posDust, oldIonPos, d_RAD_COLL_MULT);

            if (currentBoundIon == 0)
            {
                // calculate the acceleration due to ion-dust interactions
                calcIonDustAcc_100_dev(currentPosIon, currentIonDustAcc, d_posDust, d_NUM_ION,
                                       d_NUM_DUST, d_SOFT_RAD_SQRD, d_ION_DUST_ACC_MULT);

                // Kick with IonDust accels for deltat/2^(m-1)
                if (depth == timeStepFactor)
                {
                    // on last time step, do a half kick
                    kick_dev(currentVelIon, currentIonDustAcc, halfTimeStep);
                }
                else
                {
                    kick_dev(currentVelIon, currentIonDustAcc, timeStep);
                }
            }
        }
    } // end for loop over depth

    // save the ion parameters into global memory
    d_boundsIon[threadID] = currentBoundIon;
    posIon[threadID] = currentPosIon;
    velIon[threadID] = currentVelIon;
    ionDustAcc[threadID] = currentIonDustAcc;
}

/*
 * Name: kick_dev
 * Created: 3/14/2017
 * last edit: 3/14/2018
 *
 * Editors
 *	Name: Lorin Matthews
 *	Contact: Lorin_Matthews@baylor.edu
 *
 * Description:
 *	Kick for half timestep in a leapfrog integration
 *
 * Input:
 *	vel: velocity
 *	acc: acceleration
 *	ts: time step
 *
 * Output (void):
 *	vel: updated velocity from the integration
 *
 * Assumptions:
 *	All inputs are real values
 *
 * Includes:
 *	cuda_runtime.h
 *	device_launch_parameters.h
 *
 */
__device__ void kick_dev(float4 &vel, float4 &acc, float timestep)
{
    // calculate velocity
    // v = v0 + dT * a

    vel.x = __fmaf_rn(acc.x, timestep, vel.x);
    vel.y = __fmaf_rn(acc.y, timestep, vel.y);
    vel.z = __fmaf_rn(acc.z, timestep, vel.z);
}

/*
 * Name: drift_dev
 * Created: 3/14/2017
 * last edit: 3/14/2018
 *
 * Editors
 *	Name: Lorin Matthews
 *	Contact: Lorin_Matthews@baylor.edu
 *
 * Description:
 *	Drift for whole  timestep in a leapfrog integration
 *
 * Input:
 *	pos: position
 *	vel: velocity
 *	timestep: time step
 *
 * Output (void):
 *	pos: updated position from the integration
 *
 * Assumptions:
 *	All inputs are real values.
 *
 * Includes:
 *	cuda_runtime.h
 *	device_launch_parameters.h
 *
 */
__device__ void drift_dev(float4 &pos, float4 &vel, float timestep)
{
    // calculate position
    // x = x0 +  dT* v(n+1/2)

    pos.x = __fmaf_rn(vel.x, timestep, pos.x);
    pos.y = __fmaf_rn(vel.y, timestep, pos.y);
    pos.z = __fmaf_rn(vel.z, timestep, pos.z);
}
/*
 * Name: checkIonSphereBounds_100_dev
 * Created: 3/17/2018
 *
 * Editors
 *	Name: Lorin Matthews
 *	Contact: Lorin_Matthews@baylor.edu
 *
 * Description:
 *	Checks if an ion has left the simulation sphere
 *
 * Input:
 *	d_posIion: ion positions
 *	d_boundsIon: a flag for if an ion is out of bounds
 *	d_RAD_SIM_SQRD: the simulation radius squared
 *
 * Output (void):
 *	d_boundsIon: set to -1 for ions that are outside of the
 *		simulation sphere.
 *
 * Assumptions:
 *	The simulation region is a sphere with (0,0,0) at its center
 *   The number of ions is a multiple of the block size
 *   the flag -1 is unique value for the ion bounds flag
 *
 * Includes:
 *	cuda_runtime.h
 *	device_launch_parameters.h
 *
 */

__device__ void checkIonSphereBounds_100_dev(float4 &d_posIon, int &d_boundsIon,
                                             const float *d_RAD_SIM_SQRD)
{

    // distance
    float dist;

    // Only check ions which are in bounds
    if (d_boundsIon == 0)
    {

        // distance from the center of the simulation
        dist = d_posIon.x * d_posIon.x + d_posIon.y * d_posIon.y + d_posIon.z * d_posIon.z;

        // check if the ion is out of the simulation sphere
        if (dist > *d_RAD_SIM_SQRD)
        {
            // flag the ion as out of the simulation sphere
            d_boundsIon = -1;
        }
    }
}

/*
 * Name: checkIonCylinderBounds_100_dev
 *
 * Created: 3/17/2018
 *
 * Editors
 *	Name: Lorin Matthews
 *	Contact: Lorin_Matthews@baylor.edu
 *	last edit: 11/18/2017
 *
 * Description:
 *	Checks if an ion has left the simulation cylinder
 *
 * Input:
 *	d_posIion: ion positions
 *	d_boundsIon: a flag for if an ion is out of bounds
 *	d_RAD_CYL_SQRD: the simulation radius squared
 *	d_HT_CYL: the (half)height of the cylinder
 *
 * Output (void):
 *	d_boundsIon: set to -1 for ions that are outside of the
 *		simulation sphere.
 *
 * Assumptions:
 *	The simulation region is a cylinder with (0,0,0) at its center
 *   The number of ions is a multiple of the block size
 *   the flag -1 is unique value for the ion bounds flag
 *
 * Includes:
 *	cuda_runtime.h
 *	device_launch_parameters.h
 *
 */
__device__ void checkIonCylinderBounds_100_dev(float4 &d_posIon, int &d_boundsIon,
                                               const float *d_RAD_CYL_SQRD, const float *d_HT_CYL)
{
    // distance
    float dist;
    float distz;

    // Only check ions which are in bounds
    if (d_boundsIon == 0)
    {
        // radial distance from the center of the cylinder
        dist = __fmaf_rn(d_posIon.x, d_posIon.x, d_posIon.y * d_posIon.y);

        // height from the center of the cylinder
        distz = fabsf(d_posIon.z);

        // check if the ion is out of the simulation cylinder
        if (dist > *d_RAD_CYL_SQRD || distz > *d_HT_CYL)
        {
            // flag the ion as out of the simulation sphere
            d_boundsIon = -1;
        }
    }
}

/*
 * Name: checkIonDustBounds_100_dev
 * Created: 3/17/2018
 * Edited: 4/30/2019 to save ion momentum transfer
 *
 * Editors
 *	Name: Lorin_Matthews
 *
 * Description:
 *	checks if an ion is within  a dust particle
 *	or if path connecting current and past ion position intersect
 *	the dust particle
 *
 * Input:
 *	d_posIon: the ion positions and charges
 *	posIon2: past ion position
 * 	d_velIon: ion velocities
 *	d_boundsIon: a flag for if an ion position is out of bounds
 *	d_RAD_DUST: the radius of the dust particles
 *	d_NUM_DUST: the number of dust particles
 *	d_posDust: the dust particle positions
 *	d_RAD_COLL_MULT: constant multiplier used for collection radius
 *
 * Output (void):
 *	d_boundsIon: set to the index of the dust particle the ion is
 *		in if the ion is in or has crossed a dust particle.
 *
 * Assumptions:
 *	All dust particles have the same radius
 *   The dust particles and ions are on the same coordinate axis
 *   The number of ions is a multiple of the block size
 *
 */
__device__ void checkIonDustBounds_100_dev(float4 &d_posIon, int &d_boundsIon,
                                           const float *d_RAD_DUST, const int *d_NUM_DUST,
                                           float4 *const d_posDust, float4 posIon2,
                                           const float *d_RAD_COLL_MULT)
{

    // distance
    float dist, b_c;
    float tiny_num = 1e-12f;

    // Only check ions which are in bounds
    if (d_boundsIon == 0)
    {

        // temporary distance holders
        float deltaX, deltaY, deltaZ, u, dpX, dpY, dpZ, Px, Py, Pz, P_sq;

        // loop over all of the dust particles
        for (int i = 0; i < *d_NUM_DUST; i++)
        {
            // get the current dust grain
            float4 currentPosDust = d_posDust[i];

            // x, y, and z distances between the current
            // ion and dust particle
            deltaX = currentPosDust.x - d_posIon.x;
            deltaY = currentPosDust.y - d_posIon.y;
            deltaZ = currentPosDust.z - d_posIon.z;

            // the squared distance between the current ion
            // and dust particle
            dist = __fmaf_rn(deltaX, deltaX, __fmaf_rn(deltaY, deltaY, deltaZ * deltaZ));

            // check if the dust particle and ion have collided
            // calculate the collection radius
            // RAD_COLL_MULT = 2*qi/mi/vs^2 * COULOMB_CONST/RAD_DUST
            b_c = *d_RAD_DUST * *d_RAD_DUST * (1 - *d_RAD_COLL_MULT * currentPosDust.w);
            if (dist < b_c)
            // if (dist < *d_RAD_DUST * *d_RAD_DUST)
            {
                // flag which dust particle the ion is in
                d_boundsIon = (i + 1);
            }
            else
            {
                // Line segment of ion's trajectory
                // The equation of point P on line is P = P1 + u*(P2-P1)
                dpX = posIon2.x - d_posIon.x;
                dpY = posIon2.y - d_posIon.y;
                dpZ = posIon2.z - d_posIon.z;

                // Determine if line segment intersects dust.
                // Find point P on line closest to dust center
                // by using fact that this is perpendicular
                // distance and the dot product is zero.
                // (posDust - P) dot (P2 - P1) = 0
                // (posDust - P1 - u(P2-P1)) dot (P2-P1) = 0
                //  If 0<= u <=1, and |P| <= radDust, then
                //  line segment intersects sphere.
                u = __fmaf_rn(deltaX, dpX, __fmaf_rn(deltaY, dpY, deltaZ * dpZ)) / __fmaf_rn(dpX, dpX, __fmaf_rn(dpY, dpY, dpZ * dpZ));

                if (u > tiny_num && u < 1)
                {
                    // Closest point on line segment; check if |P| < dustRad
                    Px = __fmaf_rn(u, dpX, d_posIon.x);
                    Py = __fmaf_rn(u, dpY, d_posIon.y);
                    Pz = __fmaf_rn(u, dpZ, d_posIon.z);

                    // get the distance between the point P and the dust grain's center
                    float dist_P_dust_x = currentPosDust.x - Px;
                    float dist_P_dust_y = currentPosDust.y - Py;
                    float dist_P_dust_z = currentPosDust.z - Pz;
                    P_sq = __fmaf_rn(dist_P_dust_x, dist_P_dust_x, __fmaf_rn(dist_P_dust_y, dist_P_dust_y, dist_P_dust_z * dist_P_dust_z));

                    if (P_sq <= b_c)
                    {
                        // if (P_sq <= *d_RAD_DUST * *d_RAD_DUST) {
                        d_boundsIon = (i + 1);
                    }
                }
            } // close else
        } // for loop over dust
    } // if for checking if ions in bounds
} // close function

/*
 * Name: calcIonDustAcc_100_dev
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
 *	d_posIon: the positions and charge of the ions
 *	d_accIon: the accelerations of the ions
 *	d_posDust: the dust particle positions and charges
 *	d_NUM_ION: the number of ions
 *	d_NUM_DUST: the number of dust particles
 *	d_SOFT_RAD_SQRD: the squared softening radius squared
 *	d_ION_DUST_ACC_MULT: a constant multiplier for the ion-dust interaction
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
__device__ void calcIonDustAcc_100_dev(float4 &d_posIon, float4 &d_accIon, float4 *d_posDust,
                                       const int *d_NUM_ION, const int *d_NUM_DUST,
                                       const float *d_SOFT_RAD_SQRD,
                                       const float *d_ION_DUST_ACC_MULT)
{

    // allocate variables
    float3 dist;
    float distSquared;
    float invDistance;
    float invDistance3;
    float linForce;
    float softDistSqrd = 1e-20f;

    // reset the acceleration
    d_accIon = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    // loop over all of the dust particles
    for (int h = 0; h < *d_NUM_DUST; h++)
    {
        // get the current dust grain
        float4 currentPosDust = d_posDust[h];

        // calculate the distance between the ion in shared
        // memory and the current thread's ion
        dist.x = d_posIon.x - currentPosDust.x;
        dist.y = d_posIon.y - currentPosDust.y;
        dist.z = d_posIon.z - currentPosDust.z;

        // calculate the distance squared
        distSquared = __fmaf_rn(dist.x, dist.x, __fmaf_rn(dist.y, dist.y, dist.z * dist.z)) + softDistSqrd;

        // calculate the inverse distance
        invDistance = rsqrtf(distSquared);

        // find the inverse of the soft distance cubed
        invDistance3 = invDistance * invDistance * invDistance;

        // calculate a scalar intermediate
        linForce = currentPosDust.w * invDistance3;

        // add the acceleration to the current ion's acceleration
        d_accIon.x = __fmaf_rn(linForce, dist.x, d_accIon.x);
        d_accIon.y = __fmaf_rn(linForce, dist.y, d_accIon.y);
        d_accIon.z = __fmaf_rn(linForce, dist.z, d_accIon.z);

    } // end loop over dust

    // scale the acceleration
    d_accIon.x = *d_ION_DUST_ACC_MULT * d_accIon.x;
    d_accIon.y = *d_ION_DUST_ACC_MULT * d_accIon.y;
    d_accIon.z = *d_ION_DUST_ACC_MULT * d_accIon.z;
}
