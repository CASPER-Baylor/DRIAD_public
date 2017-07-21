/*
*
* File Name: IonWake_105_2DAnalysis.cpp
* File Type: function library header
* Created: 6/20/2017
*
* Description:
*	Includes fuctions for data analysis which output
*	2D arrays. The output is primarily intended for
*	use in 2D visualizations.
*
* Functions:
*	getNumDen()
*
* Dependencies:
*	"IonWake_106_Utilities.h"
*	<stdlib.h>
*
*/

#ifndef IONWAKE_105_2DANALYSIS
#define IONWAKE_105_2DANALYSIS

/*
* Name: getNumDen
*
* Editors:
*	Dustin Sanford
*	Beau Brooks
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
* Asumptions:
*	horizData and vertData are the same length and 1D
*	gridRes is positive
*	both horizData and vertData have a range of values
*	the output matrix is square
*	the values in horizData and vertData are on the same scale
*
* Dependencies:
*	<stdlib.h>
*	"IonWake_106_Utilities.h"
*
*/

int* getNumDen(int gridRes, int numVal, float horizData[], float vertData[]);

#endif // IONWAKE_105_2DANALYSIS