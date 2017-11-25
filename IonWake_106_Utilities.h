/*
* Project: IonWake
* File Type: function library header
* File Name: IonWake_106_Utilities.h
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
* Includes:
*	none
*
*/

#ifndef IONWAKE_106_UTILITIES
#define IONWAKE_106_UTILITIES

	/*
	* Name: findMax
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

	float findMax(float val[], int numVals);
	int   findMax(  int val[], int numVals);

	/* 
	* Name: findMin
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

	float findMin(float val[], int numVals);
	int   findMin(  int val[], int numVals);

#endif // IONWAKE_106_UTILITIES