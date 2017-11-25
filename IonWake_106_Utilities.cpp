/*
* Project: IonWake
* File Type: function library implementation
* File Name: IonWake_106_Utilities.cpp
*
* Created: 6/20/2017
* Last Modified: 8/29/2017
*
* Description:
*	Includes general purpose algorithms  
*
* Functions:
*	findMax()
*	findMin()
*
*/

// header file
#include "IonWake_106_Utilities.h"

/*
* Name: findMax
* Created: 6/20/2017
* last edit: 6/21/2017
*
* Editors
*	Name: Dustin Sanford
*	Contact: Dustin_Sanford@baylor.edu
*	last edit: 6/21/2017
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

float findMax(float val[], int numVals)
{
	// set an initial maximum
	float max = val[0];

	// loop over each of the ions
	for (int i = 0; i < numVals; i++)
	{
		// check if the current is greator than the max
		if (val[i] > max)
		{
			max = val[i];
		}
	}
	return max;
}

int findMax(int val[], int numVals)
{
	// set an initial maximum
	int max = val[0];

	// loop over each of the ions
	for (int i = 0; i < numVals; i++)
	{
		// check if the current is greator than the max
		if (val[i] > max)
		{
			max = val[i];
		}
	}
	return max;
}

/*
* Name: findMin
* Created: 6/20/2017
* last edit: 6/21/2017
*
* Editors
*	Name: Dustin Sanford
*	Contact: Dustin_Sanford@baylor.edu
*	last edit: 6/21/2017
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

float findMin(float val[], int numVals)
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

int findMin(int val[], int numVals)
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