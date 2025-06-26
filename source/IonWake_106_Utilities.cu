/*
 * Project: IonWake
 * File Type: function library implementation
 * File Name: IonWake_106_Utilities.cpp
 *
 * Created: 6/20/2017
 * Last Modified: 09/17/2020
 *
 * Description:
 *	Includes general purpose algorithms
 *
 * Functions:
 *	findMax_106()
 *	findMin_106()
 *	normRand_106()
 *
 */

// header file
#include "IonWake_106_Utilities.hpp"

/*
 * Name: findMax_106
 * Created: 6/20/2017
 * last edit: 11/14/2017
 *
 * Editors
 *	Name: Dustin Sanford
 *	Contact: Dustin_Sanford@baylor.edu
 *	last edit: 11/14/2017
 *
 * Description:
 *	Takes a 1D array and returns the maximum value
 *
 * Input:
 *	val[]: the input values
 *	numVals: the number of input values
 *
 * Output:
 *	max: the maximum value in the array
 *
 * Data Abstraction:
 *	val[]: the input values
 *	numVals: the number of input values
 *	max: the maximum value in the array
 *
 * Assumptions:
 *	val is 1D and contains ints or floats
 *
 * Includes:
 *	none
 *
 */

float findMax_106(float val[], int numVals)
{
    // set an initial maximum
    float max = val[0];

    // loop over each of the ions
    for (int i = 0; i < numVals; i++)
    {
        // check if the current is greater than the max
        if (val[i] > max)
        {
            max = val[i];
        }
    }
    return max;
}

int findMax_106(int val[], int numVals)
{
    // set an initial maximum
    int max = val[0];

    // loop over each of the ions
    for (int i = 0; i < numVals; i++)
    {
        // check if the current is greater than the max
        if (val[i] > max)
        {
            max = val[i];
        }
    }
    return max;
}

/*
 * Name: findMin_106
 * Created: 6/20/2017
 * last edit: 1/14/2017
 *
 * Editors
 *	Name: Dustin Sanford
 *	Contact: Dustin_Sanford@baylor.edu
 *	last edit: 11/14/2017
 *
 * Description:
 *	Takes a 1D array and returns the minimum value
 *
 * Input:
 *	val[]: the input values
 *	numVals: the number of input values
 *
 * Output:
 *	max: the minimum value in the array
 *
 * Data Abstraction:
 *	val[]: the input values
 *	numVals: the number of input values
 *	max: the minimum value in the array
 *
 * Assumptions:
 *	val is 1D and contains ints or floats
 *
 * Includes:
 *	none
 *
 */

float findMin_106(float val[], int numVals)
{
    // set an initial min
    float min = val[0];

    // loop over the ions
    for (int i = 1; i < numVals; i++)
    {
        // check if the
        if (val[i] < min)
        {
            min = val[i];
        }
    }
    return min;
}

int findMin_106(int val[], int numVals)
{
    // set an initial min
    int min = val[0];

    // loop over the ions
    for (int i = 1; i < numVals; i++)
    {
        // check if the
        if (val[i] < min)
        {
            min = val[i];
        }
    }
    return min;
}

/*
 * Name: normRand_106
 * Created: 9/17/2020
 * last edit: 9/17/2020
 *
 * Editors
 *	Name: Lorin Matthews
 *	Contact: Lorin_Matthews@baylor.edu
 *	last edit: 09/17/2020
 *
 * Description:
 *	Generates a random number from a normal distribution
 *
 * Input:
 *	void
 *
 * Output:
 *	num: the random number
 *
 *
 * Assumptions:
 *
 * Includes:
 *	none
 */

float normRand_106()
{
    // get two random numbers from a uniform distribution
    double x1, x2;
    x1 = ((rand()) + 1.) / ((double)(RAND_MAX) + 1.);
    x2 = ((rand()) + 1.) / ((double)(RAND_MAX) + 1.);

    return cos(2 * 3.14159 * x2) * sqrt(-2 * log(x1));
}

int findMaxNumberOfBlocksForKernel(int threadsPerBlock, int numberElements, const void *kernelPtr, size_t sharedMemBytes)
{
    // store the device id and number of SMs
    int deviceId;
    int numberOfSMs;

    // get the device id and number of SMs
    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

    // calculate the maximum number of blocks per SM for a given kernel
    int blocksPerSm;

    // calculate the maximum number of blocks per SM for the calcIonAccels_102 kernel
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSm, kernelPtr, threadsPerBlock, sharedMemBytes);

    // calculate the maximum number of blocks that can be running at the same time in the GPU
    int waveSize = numberOfSMs * blocksPerSm;

    // set the minimum number of blocks to coverage all the ions
    int minimumNumberOfBlocks = (numberElements + 1) / threadsPerBlock;

    // set the number of wave to cover all the ions
    int numWaves = (minimumNumberOfBlocks + waveSize - 1) / waveSize;

    // set the number of blocks to cover all the ions and to avoid partial waves
    int numberOfBlocks = waveSize * numWaves;

    return numberOfBlocks;
}

std::vector<float3> create3DMesh(float3 centerCube, float3 gridHalfSize, int3 gridResolution)
{
    // get the cube center
    float centerCubeX = centerCube.x;
    float centerCubeY = centerCube.y;
    float centerCubeZ = centerCube.z;

    // get the grid half size in x, y, and z directions
    float gridHalfSizeX = gridHalfSize.x;
    float gridHalfSizeY = gridHalfSize.y;
    float gridHalfSizeZ = gridHalfSize.z;

    // get the grid resolution in x, y, and z directions
    int gridResolutionX = gridResolution.x;
    int gridResolutionY = gridResolution.y;
    int gridResolutionZ = gridResolution.z;

    // increment in x, y, and z directions
    float incrementX = 2 * gridHalfSizeX / (gridResolutionX - 1);
    float incrementY = 2 * gridHalfSizeY / (gridResolutionY - 1);
    float incrementZ = 2 * gridHalfSizeZ / (gridResolutionZ - 1);

    // create the  3D mesh centered around the grain
    std::vector<float3> mesh3D;

    // loop over the z direction
    for (int k = 0; k < gridResolutionZ; k++)
    {
        // node in the z direction
        float z = -gridHalfSizeZ + k * incrementZ + centerCubeZ;

        // loop over the j direction
        for (int j = 0; j < gridResolutionY; j++)
        {
            // node in the x direction
            float y = -gridHalfSizeY + j * incrementY + centerCubeY;

            // loop over the x direction
            for (int i = 0; i < gridResolutionX; i++)
            {
                // node in the x direction
                float x = -gridHalfSizeX + i * incrementX + centerCubeX;

                // add the node to the mesh
                mesh3D.push_back(make_float3(x, y, z));
            }
        }
    }

    return mesh3D;
}