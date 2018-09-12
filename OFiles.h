#ifndef IONWAKE_OFILES_INCLUDED
#define IONWAKE_OFILES_INCLUDED

#include <string>
#include "OFile.h"

struct OFiles {

    OFiles( std::string prefix ) :
        debug(prefix + "_debug.txt" ),
        debugSpecific( prefix + "_debug-specific.txt" ),
        trace( prefix + "_trace.txt" ),
        status( prefix + "_status.txt" ),
        ionPos( prefix + "_ion-pos.txt" ),
        dustPos( prefix + "_dust-pos.txt" ),
        dustCharge( prefix + "_dust-charge.txt" ),
        dustPosTrace( prefix + "_dust-pos-trace.txt" ),
        params( prefix + "_params.txt" ),
        ionDen( prefix + "_ion-den.txt" )
    {
		debug.precision(5);
		debug << std::showpoint;
	}   
    
    // file for general debugging output
    OFile debug;
    // file for specific debugging output
    OFile debugSpecific;
    // file for tracing values throughout the timestep
    OFile trace;
    // file for holding the status of the simulation
    OFile status;
    // file for holding ion positions
    OFile ionPos;
    // file for holding dust positions
    OFile dustPos;
    // file for holding dust charges
    OFile dustCharge;
    // file for tracing dust positions during the simulation
    OFile dustPosTrace;
    // file for outputting the input parameters
    OFile params;
    // file for outputting the grid data
    OFile ionDen;
};

#endif // include guard
