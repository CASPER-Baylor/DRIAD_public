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


// includes the leapfrog integrator used 
// in the time step
#include "IonWake_100_integrate.h"
// includes functions for finding and 
// replacing out of bounds ions
#include "IonWake_101_bounds.h"
// includes functions used for calculating
// the ion accelerations
#include "IonWake_102_ionIonAcc.h"
// includes functions used to get
// user parameters from a text file
#include "IonWake_103_getUsserParams.h"
// includes functions used to plot 
// ion data. 
#include "IonWake_104_plotIonData.h"
// includes functions used to analyze
// ion data in two dimentions
#include "IonWake_105_2DAnalysis.h"
// includes utility functions
#include "IonWake_106_Utilities.h"
// includes functions used to create BMP
// images using the easyBMP library
#include "IonWake_107_PlotBMP.h"
// includes functions to calculate 
// ion dust accelerations
#include "IonWake_108_ionDustAcc.h"

// required for CUDA
#include "cuda_runtime.h"
// required for CUDA
#include "device_launch_parameters.h"
// includes functions used for writing 
// data to a bmp file
#include "EasyBMP.h"

#endif // IONWAKE_000