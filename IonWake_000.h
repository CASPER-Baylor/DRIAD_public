/*
* Project: IonWake
* File Type: header
* File Name: IonWake_000.h
*
* Created: 6/13/2017
* Last Modified: 8/26/2017 
*/

#ifndef IONWAKE_000
#define IONWAKE_000

	// included for sqrt()
	#include <cmath>
	// includes malloc()
	#include <cstdlib>
	// used to handle input and output files
	#include <fstream>
	// used for output formating 
	#include <iomanip>
	// used for storing file names
	#include <string>
	// srand 
	#include <stdlib.h>   
	// time()
	#include <time.h> 

	// includes the leapfrog integrator used 
	// in the time step
	#include "IonWake_100_integrate.h"
	// includes functions for finding and 
	// replacing out of bounds ions
	#include "IonWake_101_bounds.h"
	// includes functions used for calculating
	// the ion accelerations
	#include "IonWake_102_ionAcc.h"
	// includes utility functions
	#include "IonWake_106_Utilities.h"

	// required for CUDA
	#include "cuda_runtime.h"
	#include "device_launch_parameters.h"

#endif // IONWAKE_000