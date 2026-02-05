/*
 * Project: IonWake
 * File Type: function library header
 * File Name: IonWake_106_Utilities.h
 *
 * Created: 6/20/2017
 * Last Modified: 11/14/2017
 *
 * Description:
 *	Includes general purpose algorithms
 *
 * Functions:
 *	findMax_106()
 *	findMin_106()
 *	normRand_106()
 *
 * Templates:
 * 	getParam_106()
 *
 * Includes:
 *	none
 *
 */

#ifndef IONWAKE_106_UTILITIES
#define IONWAKE_106_UTILITIES

#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

/*
 * Name: findMax_106
 *
 * Editors
 *	Dustin Sanford
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
 * Assumptions:
 *	val is 1D and contains ints or floats
 *
 * Includes:
 *	none
 *
 */

float findMax_106(float val[], int numVals);
int findMax_106(int val[], int numVals);

template <typename T>
T getParam_106(std::ifstream &file, std::string param_name)
{
    // holds lines from the file
    std::string string_line;
    std::stringstream stream_line;

    // clear any end of file error flag
    file.clear();
    // jump to the beginning of the file
    file.seekg(0, std::ios::beg);

    // Loop until the parameter is found or the end of the file is reached
    while (std::getline(file, string_line))
    {
        stream_line.str(string_line);
        stream_line >> string_line;
        if (string_line == param_name)
        {
            T param;
            stream_line >> param;
            return param;
        }
    }

    fprintf(stderr,
            "ERROR: the parameter %s was not found in "
            "parameter file.\n",
            param_name.c_str());

    exit(-1);
}

/*
 * Name: findMin_106
 *
 * Editors
 *	Dustin Sanford
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
 * Assumptions:
 *	val is 1D and contains ints or floats
 *
 * Includes:
 *	none
 *
 */

float findMin_106(float val[], int numVals);
int findMin_106(int val[], int numVals);

/*
 *  Name: normRand_106
 *
 *  Editors
 *  	Lorin Matthews
 *
 *  Description:
 *  	Returns a random number from a normal distribution
 *
 *  Input: void
 *
 *  Output: random number
 *
 */

float normRand_106();

/**
 * @brief function to get the maximum number of blocks for a kernel
 *
 */
int findMaxNumberOfBlocksForKernel(int, int, const void *, size_t);

/**
 * @brief function to create a tridimensional mesh
 *
 * @return std::vector<float3>
 */
std::vector<float3> create3DMesh(float3, float3, int3);

#endif // IONWAKE_106_UTILITIES
