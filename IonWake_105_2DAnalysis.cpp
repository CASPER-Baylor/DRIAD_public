/*
*
* File Name: IonWake_105_2DAnalysis.cpp
* File Type: function library implemtation
* Created: 6/20/2017
* Last Edit: 6/21/2017
*
* Description:
*	Includes fuctions for data analysis which output
*	2D arrays. The output is primarily intended for 
*	use in 2D visualizations. 
*
* Functions:
*	getNumDen()
*
*/

// header file
#include "IonWake_105_2DAnalysis.h"

/*
* required by:
*	getNumDen()
*/
#include "IonWake_106_Utilities.h"

/*
* required by:
*	getNumDen()
*/
#include <stdlib.h>

/*
* Name: getNumDen
* Created: 6/20/2017
* last edit: 6/21/2017
*
* Editors
*	Name: Dustin Sanford
*	Contact: Dustin_Sanford@baylor.edu
*	last edit: 6/21/2017
*
*	Name: Beau Brooks
*	Contact: Beau_Brooks@baylor.edu
*	last edit: 6/20/2017
*
* Description: 
*	Takes a list of 2-tuples and creates a square 
*	number density map with the specified grid resolution 
*
* Input:
*	gridRes: the number of rows and columns in the output array
*	numVal: the number of 2-tuples
*	horizData: a 1D array with the horizontal data for the 2-tuples
*	vertData: a 1D array with the veritcle data for the 2-tuples
*
* Output: 
*	numDen: a square 2D matrix density map
*
* Data Abstraction:
*	gridRes: the number of rows and columns in the output array
*	numVal: the number of 2-tuples
*	horizData: a 1D array with the horizontal data for the 2-tuples
*	vertData: a 1D array with the veritcle data for the 2-tuples
*	arrayMem: the size of the memory required for the output array
*	numDen: a square 2D matrix density map
*	maxVal: the maximum value from the input 2-tupes
*	tempMax: a temporary holder for the maximum of the
*		vertical data from the 2-tuples
*	horizMin: the minimum value from the horizontal data
*		from the 2-tuples
*	vertMin: the minimum value from the vertical data
*		from the 2-tuples
*
* Asumptions:
*	horizData and vertData are the same length and 1D
*	gridRes is positive
*	both horizData and vertData have a range of values
*	the output matrix is square
*	the values in horizData and vertData are on the same scale 
*
* Includes:
*	<stdlib.h>
*		malloc()
*	"IonWake_106_Utilities.h"
*		findMin()
*		findMax()
*
*/

#include <iostream>
#include <stdlib.h>

int* getNumDen(int gridRes, int numVal, float horizData[], float vertData[])
{

	// memory required for the output array
	int arrayMem = gridRes * gridRes * sizeof(int);

	// allocate the memory for the output array
	int* numDen = (int*)malloc(arrayMem);

	// Initialize each output array entry to 0
	for (int i = 0; i < gridRes; i++)
	{
		for (int j = 0; j < gridRes; j++)
		{
			*(numDen + i*gridRes + j) = 0;
		}
	}

	// get the minimum horizontal data value
	float horizMin = findMin(horizData, numVal);

	// get the minimum verticle ion data value
	float vertMin = findMin(vertData, numVal);

	for (int i = 0; i < numVal; i++)
	{
		vertData[i] -= vertMin;
		horizData[i] -= horizMin;
	}

	// get the maximum data value
	float maxVal = findMax(horizData, numVal);
	float tempMax = findMax(vertData, numVal);
	if (tempMax > maxVal)
	{
		maxVal = tempMax;
	}

	int counter, a, b;
	// loop over each element and add it to
	// the coresponding position in the numDen array
	for (int i = 0; i < numVal; i++)
	{
		a = (vertData[i] / maxVal)* gridRes;
		b = (horizData[i] / maxVal)* gridRes;
		counter = a * gridRes + b;
		
		*(numDen + counter) += 1;
	}

	return numDen;
}

