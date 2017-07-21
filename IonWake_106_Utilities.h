/*
*
* File Name: IonWake_106_Utilities.cpp
* File Type: function library header
* Created: 6/20/2017
*
* Description:
*	Includes general purpose algorythms
*
* Functions:
*	findMax()
*	findMin()
*
* Dependencies:
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
* Asumptions:
*	val is 1D and contains ints or floats
*
* Dependencies:
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
* Asumptions:
*	val is 1D and contains ints or floats
*
* Dependencies:
*	none
*
*/

float findMin(float val[], int numVals);
int   findMin(  int val[], int numVals);

#endif // IONWAKE_106_UTILITIES