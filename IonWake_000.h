/*
* Project: IonWake
* File Type: header
* File Name: IonWake_000.h
*
* Created: 6/13/2017
* Last Modified: 11/12/2017 
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
	// includes function for calculating dust accelerations
	#include "IonWake_103_dustAcc.h"
	// includes functions for calculation ion-gas collisions 
	#include "IonWake_105_ionColl.h"
	// includes utility functions
	#include "IonWake_106_Utilities.h"
	// includes output file handler class 
	#include "OFiles.h"
	#include "OFile.h"
	
    // includes abstractions for CUDA
    #include "CUDAvar.h"
    #include "constCUDAvar.h"
    #include "CUDAerr.h"
    #include "ErrorBase.h"
    
	// required for CUDA
	#include "cuda_runtime.h"
	#include "device_launch_parameters.h"

#endif // IONWAKE_000
