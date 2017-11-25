
/*
* Project: IonWake
* File Type: script - main 
* File Name: IonWake_000.cu
*
* Created: 6/13/2017
*
* Editors
*	Last Modified: 8/26/2017
*	Contributor(s):
*		Name: Dustin Sanford
*		Contact: Dustin_Sanford@baylor.edu
*		Last Contribution: 10/21/2017
*
*       Name: Lorin Matthews
*       Contact: Lorin_Matthews@baylor.edu
*       Last Contribution: 10/19/2017
*
* Description:
*	Handles the execution of the IonWake simulation. Provides a user interface 
*   in the form of input and output files. Handles memory allocation and 
*   declaration for non-function specific variables on both the CPU host and 
*   GPU device. Includes a modular time-step for rapid development and testing 
*   of various time step schemes. For descriptions of the scope of the IonWake 
*   simulation as well as user interface, program input and output, and 
*   time-step options please see the respective sections of the README file. 
*
* Output:
*	Determined by user settings. For a complete description of all available 
*   program output please see the README file.
*
* Input:
*	Determined by user settings. For a complete description of all available 
*   program input please see the README file.
*
* Implementation:
*	The program begins by opening all the input and output text files.  
*   Constant values are defined and general user parameters are pulled from 
*   the params.txt file. Derived parameters are then calculated from the user 
*   parameters. The charges and positions of the dust particles, if any, are 
*   pulled from the dust-params.txt file. The time step commands are then 
*   pulled from the timestep.txt file and parsed. Host memory is initialized 
*   and initial values are assigned. Then Device memory is allocated and the 
*   respective host variables are copied to the device. Then the time step is 
*   begun. For each time step, all the time step commands are processed in 
*   order. The time step is repeated the number of times specified in the 
*   parameters. After the time step, the calculated values are copied to the 
*   host and saved to their respective output file. Device variables that 
*   should remain constant are copied to the host and checked against their 
*   const counterparts. Finally, dynamically allocated host memory is freed, 
*   all files are closed and the program terminates.              
*
* Assumptions:
*	Determined by user time-step settings. For a complete description of all 
*	available time-step settings please see the README file.
*
*/

// header file
#include "IonWake_000.h"

void fatalError() {
	exit(-1);
}

int main(int argc, char* argv[])
{

	/*************************
		   open files
	*************************/
	
	// directory name where program inputs are read from
	std::string inputDirName = argv[1];
	// directory name where data output is saved
	std::string dataDirName = argv[2];
	// read in command line argument for the run name
	std::string runName = argv[3];
	
	// holder for full file path names
	std::string fileName;
	
	// open input file for general input parameters 
	fileName = inputDirName + "params.txt";
	std::ifstream paramFile(fileName.c_str());
	if (!paramFile){ 
		fprintf(stderr, "ERROR on line number %d in file %s\n",
            __LINE__, __FILE__);
		fprintf(stderr, "ERROR: paramFile not open\n");
		fatalError();
	}
	
	// open input file for dust parameters 
	fileName = inputDirName + "dust-params.txt";
	std::ifstream dustParamFile(fileName.c_str());
	if (!dustParamFile){
		fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "ERROR: dustParamFile not open\n");
		fatalError();
	}
	
	// open an input file for time step parameters 
	fileName = inputDirName + "timestep.txt";
	std::ifstream timestepFile(fileName.c_str());
	if (!timestepFile){
		fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "ERROR: timestepFile not open\n");	
		fatalError();
	}

	// open an output file for general debugging output
	fileName = dataDirName + runName + "_debug.txt";
	std::ofstream debugFile(fileName.c_str());
	
	// open an output file for specific debugging output 
	fileName = dataDirName + runName + "_debug-specific.txt";
	std::ofstream debugSpecificFile(fileName.c_str());
	
	// open an output file for tracing values throughout the timestep
	fileName = dataDirName + runName + "_trace.txt";
	std::ofstream traceFile(fileName.c_str());

	// open an output file for holding the status of the simulation
	fileName = dataDirName + runName + "_status.txt";
	std::ofstream statusFile(fileName.c_str());
	
	// open an output file for holding ion positions 
	fileName = dataDirName + runName + "_ion-pos.txt";
	std::ofstream ionPosFile(fileName.c_str());
	
	// open an output file for holding dust positions 
	fileName = dataDirName + runName + "_dust-pos.txt";
	std::ofstream dustPosFile(fileName.c_str());
	
	// open an output file for holding dust charges 
	fileName = dataDirName + runName + "_dust-charge.txt";
	std::ofstream dustChargeFile(fileName.c_str());
	
	// open an output file for outputting the input parameters 
	fileName = dataDirName + runName + "_params.txt";
	std::ofstream paramOutFile(fileName.c_str());
	
	/*************************
       debugging parameters
	*************************/

	// turns on or off debugging output 
	bool debugMode = true;
	
	// sets which ion to trace 
	int ionTraceIndex = 60;
	
	// set the debugFule file to display 5 digits the right of the decimal 
	debugFile.precision(5);
	debugFile << std::showpoint;
	

    /**************************
      print device properties
	**************************/
    
	if (debugMode)
	{
		// holds GPU device properties 
		cudaDeviceProp prop;
		
		// get GPU device properties
		cudaGetDeviceProperties(&prop, 0);

		// display GPU device properties
		debugFile << "-- Debugging: GPU Properties --" << '\n'
			<< "sharedMemPerBlock: " << prop.sharedMemPerBlock << '\n'
			<< "totalGlobalMem: " << prop.totalGlobalMem << '\n'
			<< "regsPerBlock: " << prop.regsPerBlock << '\n'
			<< "warpSize: " << prop.warpSize << '\n'
			<< "maxThreadsPerBlock: " << prop.maxThreadsPerBlock << '\n'
			<< "maxGridSize: " << prop.maxGridSize[0] << ", "
			                   << prop.maxGridSize[1] << ", "
			                   << prop.maxGridSize[2] << '\n'
			<< "clockRate: " << prop.clockRate << '\n'
			<< "deviceOverlap: " << prop.deviceOverlap << '\n'
			<< "multiProcessorCount: " << prop.multiProcessorCount << '\n'
			<< "kernelExecTimeoutEnabled: " 
                << prop.kernelExecTimeoutEnabled << '\n'
			<< "integrated: " << prop.integrated << '\n'
			<< "canMapHostMemory: " << prop.canMapHostMemory << '\n'
			<< "computeMode: " << prop.computeMode << std::endl
			<< "concurrentKernels: " << prop.concurrentKernels << '\n'
			<< "ECCEnabled: " << prop.ECCEnabled << '\n'
			<< "pciBusID: " << prop.pciBusID << '\n'
			<< "pciDeviceID: " << prop.pciDeviceID << '\n'
			<< "tccDriver: " << prop.tccDriver << '\n' << '\n';
        debugFile.flush();
	}

	/*************************
			constants
	*************************/

	// electron charge (Q)
	const float CHARGE_ELC = -1.602177e-19;

	// permittivity of free pace in a vacuum (F/m)
	const float PERM_FREE_SPACE = 8.854e-12;

	// Boltzmann Constant (Kgm^2)/(K*s^2)
	const float BOLTZMANN = 1.380649e-23;

	// Pi
	const float PI = 3.141593;

	// Number of threads per block
	// has a limit of 1024 and should be a multiple of warp size
	const unsigned int DIM_BLOCK = 1024;

	// number of threads in a warp
	const int WARP_SIZE = 32;

	if (debugMode)
	{
		debugFile << "-- Constants --" << '\n'
            << "CHARGE_ELC      " << CHARGE_ELC      << '\n'
            << "PERM_FREE_SPACE " << PERM_FREE_SPACE << '\n'
            << "BOLTZMANN       " << BOLTZMANN       << '\n'
            << "PI:             " << PI              << '\n'
            << "DIM_BLOCK:      " << DIM_BLOCK       << '\n'
            << "WARP_SIZE:      " << WARP_SIZE       << '\n' << '\n';
        debugFile.flush();   
	}

	/*************************
			parameters
	*************************/

	// number of user defined parameters
	const int NUM_USSER_PARAMS = 15;

	// allocate memory for user parameters
	float* params = (float*)malloc(NUM_USSER_PARAMS * sizeof(float));

    // string to dump unwanted text from the parameter file 
    std::string dump;

    // loop over the contents of the file
    for (int i = 0; i < NUM_USSER_PARAMS; i++)
    {
        // skip two columns
        paramFile >> dump >> dump;

        // save the parameter
        paramFile >> params[i];
    }
    
	// assign user defined parameters
	const int   NUM_ION = static_cast<int>(params[0] / DIM_BLOCK) * DIM_BLOCK;
	const float DEN_FAR_PLASMA = params[1];
	const float TEMP_ELC = params[2];
	const float TEMP_ION = params[3];
	const short DEN_DUST = params[4];
	const float MASS_SINGLE_ION = params[5];
	const float MACH = params[6];
	const float SOFT_RAD = params[7];
	const float RAD_DUST = params[8];
	const float CHARGE_SINGLE_ION = params[9] * CHARGE_ELC;
	const float TIME_STEP = params[10];
	const int   NUM_TIME_STEP = params[11];
	const float RAD_SIM_DEBYE = params[12];
	const int   NUM_DIV_VEL = params[13];
	const int   NUM_DIV_QTH = params[14];

	// free memory allocated for user parameters
	free(params);

	// debye length (m)
	const float DEBYE =
		sqrt(
		    (PERM_FREE_SPACE * BOLTZMANN * TEMP_ELC) /
			(DEN_FAR_PLASMA * CHARGE_ELC * CHARGE_ELC)
		);

	// dust particle mass assumes spherical particle (Kg)
	const float MASS_DUST =
		DEN_DUST * (4 / 3) * PI * RAD_DUST * RAD_DUST * RAD_DUST;

	// radius of the spherical simulation volume (m)
	const float RAD_SIM = RAD_SIM_DEBYE * DEBYE;

	// inverse debye (1/m)
	const float INV_DEBYE = 1 / DEBYE;

	// soft radius squared (m^2)
	const float SOFT_RAD_SQRD = SOFT_RAD * SOFT_RAD;

	// simulation radius squared (m^2)
	const float RAD_SIM_SQRD = RAD_SIM * RAD_SIM;

	// half of a time step (s)
	const float HALF_TIME_STEP = TIME_STEP / 2;

	// dust radius squared (m^2)
	const float RAD_DUST_SQRD = RAD_DUST * RAD_DUST;

	// volume of the simulation sphere (m^3)
	const float SIM_VOLUME = (4.0 / 3.0) * PI * RAD_SIM * RAD_SIM * RAD_SIM;

	// multiplier for super ions
	const float SUPER_ION_MULT = SIM_VOLUME * DEN_FAR_PLASMA / NUM_ION;

	// charge on each super ion (C)
	const float CHARGE_ION = CHARGE_SINGLE_ION * SUPER_ION_MULT;

	// mass of a super ion (Kg)
	const float MASS_ION = MASS_SINGLE_ION * SUPER_ION_MULT;

	// a constant multiplier for acceleration due to Ion Ion forces
	const float ION_ION_ACC_MULT = 
        (CHARGE_ION * CHARGE_ION) / (4 * PI * PERM_FREE_SPACE * MASS_ION);

	// a constant multiplier for acceleration due to Ion Dust forces
	const float ION_DUST_ACC_MULT = (9e9) * CHARGE_ION / MASS_ION;

	// a constant multiplier for acceleration due to the 
	// electric field due to plasma outside of the simulation
	const float EXTERN_ELC_MULT =
		((RAD_SIM / DEBYE) + 1) * exp(-RAD_SIM / DEBYE) *
		(CHARGE_SINGLE_ION * DEN_FAR_PLASMA * DEBYE) *
		(CHARGE_ION / MASS_ION) / (PERM_FREE_SPACE);

	// sound speed of the plasma (m/s)
	const float SOUND_SPEED = sqrt(BOLTZMANN * TEMP_ELC / MASS_SINGLE_ION);

	// the drift velocity of the ions 
	const float DRIFT_VEL_ION = MACH * SOUND_SPEED;	
		
	if (debugMode)
	{
		
		debugFile << "-- User Parameters --" << '\n'
            << "NUM_ION           " << NUM_ION           << '\n'
            << "DEN_FAR_PLASMA    " << DEN_FAR_PLASMA    << '\n'
            << "TEMP_ELC          " << TEMP_ELC          << '\n'
            << "TEMP_ION          " << TEMP_ION          << '\n'
            << "DEN_DUST          " << DEN_DUST          << '\n'
            << "MASS_SINGLE_ION   " << MASS_SINGLE_ION   << '\n'
            << "MACH              " << MACH              << '\n'
            << "SOFT_RAD          " << SOFT_RAD          << '\n'
            << "RAD_DUST          " << RAD_DUST          << '\n'
            << "CHARGE_SINGLE_ION " << CHARGE_SINGLE_ION << '\n'
            << "TIME_STEP         " << TIME_STEP         << '\n'
            << "NUM_TIME_STEP     " << NUM_TIME_STEP     << '\n'
            << "RAD_SIM_DEBYE     " << RAD_SIM_DEBYE     << '\n'
            << "NUM_DIV_VEL       " << NUM_DIV_VEL       << '\n'
            << "NUM_DIV_QTH       " << NUM_DIV_QTH       << '\n' << '\n';

		debugFile << "-- Derived Parameters --" << '\n'
            << "DEBYE       " << DEBYE       << '\n'
            << "RAD_SIM     " << RAD_SIM     << '\n'
            << "SIM_VOLUME  " << SIM_VOLUME  << '\n'
            << "SOUND_SPEED " << SOUND_SPEED << '\n'
            << "MASS_DUST   " << MASS_DUST   << '\n' << '\n';

		debugFile << "-- Super Ion Parameters --" << '\n'
            << "SUPER_ION_MULT " << SUPER_ION_MULT << '\n'
            << "CHARGE_ION     " << CHARGE_ION     << '\n'
            << "MASS_ION       " << MASS_ION       << '\n' << '\n';


		debugFile << "-- Further Derived Parameters --" << '\n'
            << "INV_DEBYE         " << INV_DEBYE         << '\n'
            << "SOFT_RAD_SQRD     " << SOFT_RAD_SQRD     << '\n'
            << "RAD_SIM_SQRD      " << RAD_SIM_SQRD      << '\n'
            << "HALF_TIME_STEP    " << HALF_TIME_STEP    << '\n'
            << "ION_ION_ACC_MULT  " << ION_ION_ACC_MULT  << '\n'
            << "ION_DUST_ACC_MULT " << ION_DUST_ACC_MULT << '\n'
            << "RAD_DUST_SQRD     " << RAD_DUST_SQRD     << '\n'
            << "EXTERN_ELC_MULT   " << EXTERN_ELC_MULT   << '\n' << '\n';
        
        debugFile.flush();
	}  

	/*************************
	    print parameters
	*************************/
	
	// set the output file to display 7 digits the right of the decimal 
	paramOutFile.precision(7);
	paramOutFile << std::showpoint << std::left;
	
	// ouput all of the parameters such that matlab can read them in	
	paramOutFile 
        << std::setw(14) << NUM_ION           << " % NUM_ION"           << '\n'
        << std::setw(14) << DEN_FAR_PLASMA    << " % DEN_FAR_PLASMA"    << '\n'
        << std::setw(14) << TEMP_ELC          << " % TEMP_ELC"          << '\n'
        << std::setw(14) << TEMP_ION          << " % TEMP_ION"          << '\n'
        << std::setw(14) << DEN_DUST          << " % DEN_DUST"          << '\n'
        << std::setw(14) << MASS_SINGLE_ION   << " % MASS_SINGLE_ION"   << '\n'
        << std::setw(14) << MACH              << " % MACH"              << '\n'
        << std::setw(14) << SOFT_RAD          << " % SOFT_RAD"          << '\n'
        << std::setw(14) << RAD_DUST          << " % RAD_DUST"          << '\n'
        << std::setw(14) << CHARGE_SINGLE_ION << " % CHARGE_SINGLE_ION" << '\n'
        << std::setw(14) << TIME_STEP         << " % TIME_STEP"         << '\n'
        << std::setw(14) << NUM_TIME_STEP     << " % NUM_TIME_STEP"     << '\n'
        << std::setw(14) << RAD_SIM_DEBYE     << " % RAD_SIM_DEBYE"     << '\n'
        << std::setw(14) << NUM_DIV_VEL       << " % NUM_DIV_VEL"       << '\n'
        << std::setw(14) << NUM_DIV_QTH       << " % NUM_DIV_QTH"       << '\n'
        << std::setw(14) << DEBYE             << " % DEBYE"             << '\n'
        << std::setw(14) << RAD_SIM           << " % RAD_SIM"           << '\n'
        << std::setw(14) << SIM_VOLUME        << " % SIM_VOLUME"        << '\n'
        << std::setw(14) << SOUND_SPEED       << " % SOUND_SPEED"       << '\n'
        << std::setw(14) << MASS_DUST         << " % MASS_DUST"         << '\n'
        << std::setw(14) << SUPER_ION_MULT    << " % SUPER_ION_MULT"    << '\n'
        << std::setw(14) << CHARGE_ION        << " % CHARGE_ION"        << '\n'
        << std::setw(14) << MASS_ION          << " % MASS_ION"          << '\n'
        << std::setw(14) << INV_DEBYE         << " % INV_DEBYE"         << '\n'
        << std::setw(14) << SOFT_RAD_SQRD     << " % SOFT_RAD_SQRD"     << '\n'
        << std::setw(14) << RAD_SIM_SQRD      << " % RAD_SIM_SQRD"      << '\n'
        << std::setw(14) << HALF_TIME_STEP    << " % HALF_TIME_STEP"    << '\n'
        << std::setw(14) << ION_ION_ACC_MULT  << " % ION_ION_ACC_MULT"  << '\n'
        << std::setw(14) << ION_DUST_ACC_MULT << " % ION_DUST_ACC_MULT" << '\n'
        << std::setw(14) << RAD_DUST_SQRD     << " % RAD_DUST_SQRD"     << '\n'
        << std::setw(14) << EXTERN_ELC_MULT   << " % EXTERN_ELC_MULT"   << '\n';
        
    paramOutFile.flush();
	
	/*************************
	     Dust Parameters
	*************************/

	// pointer for dust positions
	float3* posDust = NULL;
	// pointer for dust charges;
	float* chargeDust = NULL;

	// counts the number of dust particles
	int tempNumDust = 0;

	// amount of memory required for the dust positions 
	int memFloat3Dust = 0;
	int memFloatDust = 0;

	
	// temporary holder for lines in the file
	std::string line;

	// skip the first line
	std::getline(dustParamFile, line);

	// count the remaining lines in the file
	while (std::getline(dustParamFile, line))
	{
		tempNumDust++;
	}

	// save the number of dust particles 
	const int NUM_DUST = tempNumDust;
	
	// if there is at least one dust particle
	if (NUM_DUST > 0) {
			
		// amount of memory required for the dust variables  
		memFloat3Dust = NUM_DUST * sizeof(float3);
		memFloatDust  = NUM_DUST * sizeof(float);

		// allocate memory for the dust variables 
		posDust = (float3*)malloc(memFloat3Dust);
		chargeDust = (float*)malloc(memFloatDust);
		
		// clear the end of file error flag
		dustParamFile.clear();
		
		// seek to the beginning  of the file 
		dustParamFile.seekg(0, std::ios::beg);

		// skip the first line of the file
		std::getline(dustParamFile, line);

		// loop over the remaining lines in the file
		// saving the dust positions
		for (int i = 0; i < NUM_DUST; i++)
		{
			// skip the first entry in each line
			dustParamFile >> line;
			
			// save the dust positions
			dustParamFile >> posDust[i].x;
			dustParamFile >> posDust[i].y;
			dustParamFile >> posDust[i].z;
			
			// save the dust charge 
			dustParamFile >> chargeDust[i];
		}
	}
	
	
	// input dust positions are in terms of the Debye length 
	// convert to meters and dust charges are in terms of 
	// the electron charge, convert to coulombs
	for (int i = 0; i < NUM_DUST; i++)
	{
		posDust[i].x *= DEBYE;
		posDust[i].y *= DEBYE;
		posDust[i].z *= DEBYE;
		
		chargeDust[i] *= CHARGE_ELC;
	}

	// check if any of the dust particles are 
	// outside of the simulation sphere
	for (int i = 0; i < NUM_DUST; i++)
	{
		if (
			   (posDust[i].x*posDust[i].x +
				posDust[i].y*posDust[i].y +
				posDust[i].z*posDust[i].z  ) > RAD_SIM_SQRD
			)
		{
			fprintf(stderr, "ERROR: Dust out of simulation\n");
			fatalError();
		}
	}

	if (debugMode)
	{
		debugFile << "-- Dust Positions --" << std::endl;
		debugFile << "NUM_DUST: " << NUM_DUST << std::endl;
		for (int i = 0; i < NUM_DUST; i++)
		{
			debugFile << "X: " << posDust[i].x <<
				        " Y: " << posDust[i].y <<
			        	" Z: " << posDust[i].z << 
				        " Q: " << chargeDust[i] << std::endl;
				
		}
		debugFile << std::endl;
        
        debugFile.flush();
	} 

	/*************************
	   time step parameters
	*************************/

	// the number of commands in the file
	int numCommands = 0;

	// array for holding the time step commands
	int* commands;

	// loop over all the commands in the file to find the number of commands
	while (getline(timestepFile, line)) {
		numCommands++;
	}

	// allocate memory for the commands
	commands = (int*)malloc(numCommands * sizeof(int));

	// clear the end of file error flag
	timestepFile.clear();
		
	// seek to the beginning of the file 
	timestepFile.seekg(0, std::ios::beg);

	// loop over all the commands and save them to the commands array
	for (int i = 0; i < numCommands; i++) {

		// get the next command
		timestepFile >> line;

		// convert the command to an int
		if (line == "CD-leapfrog") {
			commands[i] = 1;
			
		} else if (line == "CD-ion-ion-acc") {
			commands[i] = 2;
			
		} else if (line == "CD-ion-dust-acc") {
			if (NUM_DUST > 0){
				commands[i] = 3;
			} else {
				fprintf(stderr, "ERROR: cannot 'CD-ion-dust-acc'");
                fprintf(stderr, " without a dust particle");
				fatalError();
			}
			
		} else if (line == "CD-extrn-ion-acc") {
			commands[i] = 4;
			
		} else if (line == "CP-ion-pos") {
			commands[i] = 5;
			
		} else if (line == "TR-ion-pos") {
			commands[i] = 6;
			
		} else if (line == "CP-ion-acc") {
			commands[i] = 7;
			
		} else if (line == "TR-ion-acc") {
			commands[i] = 8;
			
		} else if (line == "CP-ion-vel") {
			commands[i] = 9;
			
		} else if (line == "TR-ion-vel") {
			commands[i] = 10;
			
		} else if (line == "CD-ion-sphere-bounds") {
			commands[i] = 11;
			
		} else if (line == "CD-ion-dust-bounds") {
			if (NUM_DUST > 0){	
				commands[i] = 12;
			} else {
				fprintf(stderr, "ERROR: cannot 'CD-ion-dust-bounds' ");
                fprintf(stderr, "without a dust particle");
				fatalError();
			}
			
		} else if (line == "CD-inject-ions") {
			commands[i] = 13;
			
		} else if (line == "CD-reset-ion-bounds") {
			commands[i] = 14;
			
		} else if (line == "CH-charge-dust") {
			if (NUM_DUST > 0){	
				commands[i] = 15;
			} else {
				fprintf(stderr, "ERROR: cannot 'CH-charge-dust'");
                fprintf(stderr, " without a dust particle");
				fatalError();
			}
			
		} else if (line == "CP-ion-bounds") {
				commands[i] = 16;
				
		} else if (line == "PS-charge-dust") {
			if (NUM_DUST > 0){	
				commands[i] = 17;
			} else {
				fprintf(stderr, "ERROR: cannot 'PS-charge-dust'");
                fprintf(stderr, " without a dust particle");
				fatalError();
			}
			
		} else if (line == "CP-charge-dust") {
			if (NUM_DUST > 0){	
				commands[i] = 18;
			} else {
				fprintf(stderr, "ERROR: cannot 'CP-charge-dust'");
                fprintf(stderr, " without a dust particle");
				fatalError();
			}
			
		} else if (line == "TR-dust-charge") {
			if (NUM_DUST > 0){	
				commands[i] = 19;
			} else {
				fprintf(stderr, "ERROR: cannot 'TR-dust-charge'");
				fprintf(stderr, " without a dust particle");
                fatalError();
			}
			
		} else if (line == "CH-ion-dust-current") {
			if (NUM_DUST > 0){	
				commands[i] = 20;
			} else {
				fprintf(stderr, "ERROR: cannot 'CH-ion-dust-current'");
                fprintf(stderr, " without a dust particle");
				fatalError();
			}
			
		} else if (line == "TR-ion-current") {
			if (NUM_DUST > 0){	
				commands[i] = 21;
			} else {
				fprintf(stderr, "ERROR: cannot 'TR-ion-current'");
                fprintf(stderr, " without a dust particle");
				fatalError();
			}
			
		} else {
			// if the command does not exist give an error message
			fprintf(stderr, "ERROR on line number %d in file %s\n", 
                __LINE__, __FILE__);
			fprintf(stderr, "Command \"%s\" does not exist\n\n", line.c_str());

			// terminate the program 
			fatalError();
		}
	}

	if (debugMode) {

		debugFile << "-- Time Step Commands --" << std::endl;

		debugFile << "Commands: " << '\n'
            << "1:  CD-leapfrog" << '\n'
            << "2:  CD-ion-ion-acc" << '\n'
            << "3:  CD-ion-dust-acc"  << '\n'
            << "4:  CD-extern-ion-acc" << '\n'
            << "5:  CP-ion-pos" << '\n'
            << "6:  TR-ion-pos" << '\n'
            << "7:  CP-ion-acc" << '\n'
            << "8:  TR-ion-acc" << '\n'
            << "9:  CP-ion-vel" << '\n'
            << "10: TR-ion-vel" << '\n'
            << "11: CD-ion-sphere-bounds" << '\n'
            << "12: CD-ion-dust-bounds" << '\n'
            << "13: CD-inject-ions" << '\n'
            << "14: CD-reset-ion-bounds" << '\n'
            << "15: CH-charge-dust" << '\n'
            << "16: CP-ion-bounds" << '\n'
            << "17: PS-charge-dust" << '\n'
            << "18: CP-charge-dust" << '\n'
            << "19: TR-dust-charge" << '\n'
            << "20: CH-ion-dust-current" << '\n'
            << "21: TR-ion-current" << '\n';
		debugFile << "--------------------" << std::endl;

		debugFile << "Number of commands: " << numCommands << std::endl;

		for (int i = 0; i < numCommands; i++) {
			debugFile << commands[i] << std::endl;
		}

		debugFile << "--------------------" << std::endl << std::endl;
        
        debugFile.flush();

	} 
    
	/**************************
     initialize host variables
	***************************/

	// number of blocks per grid for ions
	int blocksPerGridIon = (NUM_ION + 1) / DIM_BLOCK;

	// memory size for float3 type ion data arrays  
	int memFloat3Ion = NUM_ION * sizeof(float3);

	// allocate memory for the ion positions 
	float3* posIon = (float3*)malloc(memFloat3Ion);
    
    // allocate memory for the ion velocities  
	float3* velIon = (float3*)malloc(memFloat3Ion);
    
    // allocate memory for the ion accelerations 
	float3* accIon = (float3*)malloc(memFloat3Ion);

	// allocate memory for the ion bounds flag
	int* boundsIon = (int*)malloc(NUM_ION * sizeof(int));

	// set all ions to in-bounds
	for (int i = 0; i < NUM_ION; i++){
		boundsIon[i] = 0;
	}
	
	// allocate memory for the ion current to each dust particle
	int* ionCurrent = new int[NUM_DUST];
	
	// set initial currents to 0
	for (int i = 0; i < NUM_DUST; i++){
		ionCurrent[i] = 0;
	}
		
	// seed the random number generator 
	srand (time(NULL));
	
    // numbers used when calculating random positions and velocities
	int number = 1000;
	float randNum;
    
    // holds the distance of each ion from the center of the simulation sphere
	float dist;
	
	// loop over all the ions and initialize their velocity, acceleration,
    // and position
	for (int i = 0; i < NUM_ION; i++)
	{

        // give the ion a random position 
		randNum = (((rand() % (number*2)) - number) / (float)number);
		posIon[i].x = randNum * RAD_SIM;
		randNum = (((rand() % (number*2)) - number) / (float)number);
		posIon[i].y = randNum * RAD_SIM;
		randNum = (((rand() % (number*2)) - number) / (float)number);
		posIon[i].z = randNum * RAD_SIM;

        // calculate the distance from the ion to the center of the 
        // simulation sphere
		dist = posIon[i].x * posIon[i].x + 
		       posIon[i].y * posIon[i].y + 
			   posIon[i].z * posIon[i].z;

        // while the ion is outside of the simulation sphere, give it 
        // a new random position.
		while (dist > RAD_SIM * RAD_SIM){
			randNum = (((rand() % (number*2)) - number) / (float)number);
			posIon[i].x = randNum * RAD_SIM;
			randNum = (((rand() % (number*2)) - number) / (float)number);
			posIon[i].y = randNum * RAD_SIM;
			randNum = (((rand() % (number*2)) - number) / (float)number);
			posIon[i].z = randNum * RAD_SIM;

            // recalculate the distance to the center of the simulation
			dist = posIon[i].x * posIon[i].x + 
				   posIon[i].y * posIon[i].y + 
			       posIon[i].z * posIon[i].z;
		}
		
		// give the ion an initial random velocity
		randNum = (((rand() % (number*2)) - number) / (float)number);
		velIon[i].x = SOUND_SPEED * randNum;
		randNum = (((rand() % (number*2)) - number) / (float)number);
		velIon[i].y = SOUND_SPEED * randNum;
		randNum = ((rand() % (number*2)) / (float)number) + 2*MACH;
		velIon[i].z = - SOUND_SPEED * randNum;
		
		// set the initial acceleration to 0
		accIon[i].x = 0;
		accIon[i].y = 0;
		accIon[i].z = 0;
	}	

	if (debugMode)
	{
        
        debugFile << "-- Basic Memory Sizes --" << '\n'
            << "float  " << sizeof(float) << '\n'
            << "int    " << sizeof(int) << '\n'
            << "float3 " << sizeof(float3) << '\n' << '\n';
            
        debugFile << "-- Host Memory Use --" << '\n'
            << "velIon     " << sizeof(*velIon) * NUM_ION << '\n'
            << "posIon     " << sizeof(*posIon) * NUM_ION << '\n'
            << "accIon     " << sizeof(*accIon) * NUM_ION << '\n'
            << "boundsIon  " << sizeof(*boundsIon) * NUM_ION << '\n'
            << "ionCurrent " << sizeof(*ionCurrent) * NUM_DUST << '\n' << '\n';
        
		debugFile << "-- Initial Host Variables --" << std::endl;
        
		debugFile << "First 20 ion positions: " << std::endl;
		for (int i = 0; i < 20; i++)
		{
			debugFile << "X: " << posIon[i].x <<
				        " Y: " << posIon[i].y <<
				        " Z: " << posIon[i].z << std::endl;
		}
        
		debugFile << std::endl << "Last 20 ion positions: " << std::endl;
		for (int i = 1; i <= 20; i++)
		{
			int ID = NUM_ION - i;

			debugFile << "X: "  << posIon[ID].x
					  << " Y: " << posIon[ID].y
				      << " Z: " << posIon[ID].z
				      << std::endl;
		}
        
        debugFile << std::endl << "First 20 ion velocities: " << std::endl;
		for (int i = 0; i < 20; i++)
		{
			debugFile << "X: " << velIon[i].x <<
				        " Y: " << velIon[i].y <<
				        " Z: " << velIon[i].z << std::endl;
		}
        
		debugFile << std::endl << "Last 20 ion velocities: " << std::endl;
		for (int i = 1; i <= 20; i++)
		{
			int ID = NUM_ION - i;

			debugFile << "X: "  << velIon[ID].x
					  << " Y: " << velIon[ID].y
				      << " Z: " << velIon[ID].z
				      << std::endl;
		}
        
		debugFile << std::endl;
        
        debugFile.flush();
	} 
	
	/**************************
    initialize device variables
	***************************/
	
	// variable to hold cuda status 
	cudaError_t cudaStatus;

	// create device int pointers 
    int* d_NUM_DIV_QTH;
	int* d_NUM_DIV_VEL;
	int* d_boundsIon;
	int* d_NUM_ION;
	int* d_NUM_DUST;
    
    // create device float pointers 
	float* d_INV_DEBYE;
	float* d_RAD_DUST_SQRD;
	float* d_SOFT_RAD_SQRD;
	float* d_RAD_SIM;
	float* d_RAD_SIM_SQRD;
	float* d_HALF_TIME_STEP;
	float* d_ION_ION_ACC_MULT;
	float* d_ION_DUST_ACC_MULT;
	float* d_EXTERN_ELC_MULT;
	float* d_QCOM;
	float* d_TEMP_ION;
	float* d_DRIFT_VEL_ION;
	float* d_VCOM;
	float* d_GCOM;
	float* d_TEMP_ELC;
	float* d_SOUND_SPEED;
	float* d_PI;
	float* d_MACH;
	float* d_chargeDust;
	float* d_MASS_SINGLE_ION;
	float* d_BOLTZMANN;

    // create device float3 pointers 
    float3* d_posIon;
	float3* d_velIon;
	float3* d_accIon;
	float3* d_posDust;
    
    // create device curandState_t pointer
    curandState_t* randStates = NULL;

    // temporary holder for memory sizes
    int memSize;
    
    // total memory allocated on the GPU
    int totalGPUmem = 0;
    
    // amount of memory to allocate
    memSize = sizeof(float);
    // add to the total memory allocated on the GPU
    totalGPUmem += memSize;
    // allocate memory on the GPU for the ion mass    
	cudaStatus = cudaMalloc(&d_MASS_SINGLE_ION, sizeof(float));
	// check if the allocation was successful
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "cudaMalloc failed: d_MASS_SINGLE_ION\n");
        // terminate the program
        fatalError();
	}
	
    // amount of memory to allocate
    memSize = sizeof(float);
    // add to the total memory allocated on the GPU
    totalGPUmem += memSize;
	// allocate memory on the GPU for the Boltzmann constant
	cudaStatus = cudaMalloc(&d_BOLTZMANN, memSize);
    // check if the allocation was successful
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "cudaMalloc failed: d_BOLTZMANN\n");
        // terminate the program
        fatalError();
	}
    
    // amount of memory to allocate
    memSize = memFloatDust;
    // add to the total memory allocated on the GPU
    totalGPUmem += memSize;			
	// allocat memory on the GPU for the dust charges
	cudaStatus = cudaMalloc(&d_chargeDust, memSize);
    // check if the allocation was successful
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "cudaMalloc failed: d_chargeDust\n");
        // terminate the program
        fatalError();
	}
    
    // amount of memory to allocate
    memSize = sizeof(float);
    // add to the total memory allocated on the GPU
    totalGPUmem += memSize;		
	// allocate memory on the GPU for the mach number
	cudaStatus = cudaMalloc(&d_MACH, memSize);
    // check if the allocation was successful
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "cudaMalloc failed: d_MACH\n");
        // terminate the program
        fatalError();
	}
	
    // amount of memory to allocate
    memSize = sizeof(float);
    // add to the total memory allocated on the GPU
    totalGPUmem += memSize;		
	// allocate memory on the GPU for the sound speed
	cudaStatus = cudaMalloc(&d_SOUND_SPEED, memSize);
    // check if the allocation was successful
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "cudaMalloc failed: d_SOUND_SPEED\n");
        // terminate the program
        fatalError();
	}

    // amount of memory to allocate
    memSize = sizeof(float);
    // add to the total memory allocated on the GPU
    totalGPUmem += memSize;		
	// allocate memory on the GPU for the electron temperature
	cudaStatus = cudaMalloc(&d_TEMP_ELC, memSize);
    // check if the allocation was successful
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "cudaMalloc failed: d_TEMP_ELC\n");
        // terminate the program
        fatalError();
	}
	
    // amount of memory to allocate
    memSize = sizeof(float);
    // add to the total memory allocated on the GPU
    totalGPUmem += memSize;	
	// allocate memory on the GPU for PI
	cudaStatus = cudaMalloc(&d_PI, memSize);
    // check if the allocation was successful
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "cudaMalloc failed: d_PI\n");
        // terminate the program
        fatalError();
	}
	
    // amount of memory to allocate
    memSize = sizeof(float);
    // add to the total memory allocated on the GPU
    totalGPUmem += memSize;	
	// allocate memory on the GPU for the ion temperature 
	cudaStatus = cudaMalloc(&d_TEMP_ION, memSize);
    // check if the allocation was successful
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "cudaMalloc failed: d_TEMP_ION\n");
        // terminate the program
        fatalError();
	}
	
    // amount of memory to allocate
    memSize = sizeof(float);
    // add to the total memory allocated on the GPU
    totalGPUmem += memSize;	
	// allocate memory on the GPU for the ion drift velocity
	cudaStatus = cudaMalloc(&d_DRIFT_VEL_ION, memSize);
    // check if the allocation was successful
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "cudaMalloc failed: d_DRIFT_VEL_ION\n");
        // terminate the program
        fatalError();
	}
	
    // amount of memory to allocate
    memSize = sizeof(int);
    // add to the total memory allocated on the GPU
    totalGPUmem += memSize;	
	// allocate GPU memory for d_NUM_DIV_QTH which is the length of d_QCOM
	cudaStatus = cudaMalloc(&d_NUM_DIV_QTH, memSize);
    // check if the allocation was successful
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "cudaMalloc failed: d_NUM_DIV_QTH\n");
        // terminate the program
        fatalError();
	}
	
    // amount of memory to allocate
    memSize = sizeof(int);
    // add to the total memory allocated on the GPU
    totalGPUmem += memSize;	
	// allocate GPU memory for d_NUM_DIV_VEL which is the length of d_VCOM
	cudaStatus = cudaMalloc(&d_NUM_DIV_VEL, memSize);
    // check if the allocation was successful
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "cudaMalloc failed: d_NUM_DIV_VEL\n");
        // terminate the program
        fatalError();
	}
	
    // amount of memory to allocate
    memSize = NUM_DIV_QTH * sizeof(float);
    // add to the total memory allocated on the GPU
    totalGPUmem += memSize;	
	// allocate GPU memory for Qcom which is used in 
	// the Piel 2017 ion injection method
	cudaStatus = cudaMalloc(&d_QCOM, memSize);
    // check if the allocation was successful
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "cudaMalloc failed: d_QCOM\n");
        // terminate the program
        fatalError();
	}

    // amount of memory to allocate
    memSize = NUM_DIV_VEL * sizeof(float);
    // add to the total memory allocated on the GPU
    totalGPUmem += memSize;		
	// allocate GPU memory for Vcom which is used in
	// the Piel 2017 ion injection method
	cudaStatus = cudaMalloc(&d_VCOM, memSize);
    // check if the allocation was successful
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "cudaMalloc failed: d_VCOM\n");
        // terminate the program
        fatalError();
	}

    // amount of memory to allocate
    memSize = NUM_DIV_QTH * NUM_DIV_VEL * sizeof(float);
    // add to the total memory allocated on the GPU
    totalGPUmem += memSize;	
	// allocate GPU memory for Gcom which is used in
	// the Piel 2017 ion injection method
	cudaStatus = cudaMalloc(&d_GCOM, memSize);
    // check if the allocation was successful
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "cudaMalloc failed: d_GCOM\n");
        // terminate the program
        fatalError();
	}
	
    // amount of memory to allocate
    memSize = NUM_ION * sizeof(int);
    // add to the total memory allocated on the GPU
    totalGPUmem += memSize;	
	// allocate GPU memory for an out of bounds flag for the ions
	cudaStatus = cudaMalloc(&d_boundsIon, memSize);
    // check if the allocation was successful
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "cudaMalloc failed: d_boundsIon\n");
        // terminate the program
        fatalError();
	}
	
    // amount of memory to allocate
    memSize = sizeof(float);
    // add to the total memory allocated on the GPU
    totalGPUmem += memSize;	
	// allocate GPU memory for the external electric field 
	// multiplier for calculating the acceleration
	cudaStatus = cudaMalloc(&d_EXTERN_ELC_MULT, memSize);
    // check if the allocation was successful
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "cudaMalloc failed: d_EXTERN_ELC_MULT\n");
        // terminate the program
        fatalError();
	}
	
    // amount of memory to allocate
    memSize = sizeof(float);
    // add to the total memory allocated on the GPU
    totalGPUmem += memSize;	
	// allocate GPU memory for the dust particle radius squared
	cudaStatus = cudaMalloc(&d_RAD_DUST_SQRD, memSize);
    // check if the allocation was successful
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "cudaMalloc failed: d_RAD_DUST_SQRD\n");
        // terminate the program
        fatalError();
	}

    // amount of memory to allocate
    memSize = sizeof(int);
    // add to the total memory allocated on the GPU
    totalGPUmem += memSize;	
	// allocate GPU memory for the number of dust particles
	cudaStatus = cudaMalloc(&d_NUM_DUST, memSize);
    // check if the allocation was successful
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "cudaMalloc failed: d_NUM_DUST\n");
        // terminate the program
        fatalError();
	}

    // amount of memory to allocate
    memSize = sizeof(int);
    // add to the total memory allocated on the GPU
    totalGPUmem += memSize;	
	// allocate GPU memory for the number of ions
	cudaStatus = cudaMalloc(&d_NUM_ION, memSize);
    // check if the allocation was successful
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "cudaMalloc failed: d_NUM_ION\n");
        // terminate the program
        fatalError();
	}

    // amount of memory to allocate
    memSize = sizeof(float);
    // add to the total memory allocated on the GPU
    totalGPUmem += memSize;	
	// allocate GPU memory for the inverse debye
	cudaStatus = cudaMalloc(&d_INV_DEBYE, memSize);
    // check if the allocation was successful
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "cudaMalloc failed: d_INV_DEBYE\n");
        // terminate the program
        fatalError();
	}

    // amount of memory to allocate
    memSize = sizeof(float);
    // add to the total memory allocated on the GPU
    totalGPUmem += memSize;	
	// allocate GPU memory for the softening radius squared
	cudaStatus = cudaMalloc(&d_SOFT_RAD_SQRD, memSize);
    // check if the allocation was successful
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "cudaMalloc failed: d_SOFT_RAD_SQRD\n");
        // terminate the program
        fatalError();
	}

    // amount of memory to allocate
    memSize = sizeof(float);
    // add to the total memory allocated on the GPU
    totalGPUmem += memSize;	
	// allocate GPU memory for the simulation radius
	cudaStatus = cudaMalloc(&d_RAD_SIM, memSize);
    // check if the allocation was successful
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "cudaMalloc failed: d_RAD_SIM\n");
        // terminate the program
        fatalError();
	}
    
    // amount of memory to allocate
    memSize = sizeof(float);
    // add to the total memory allocated on the GPU
    totalGPUmem += memSize;	
	// allocate GPU memory for the simulation radius squared
	cudaStatus = cudaMalloc(&d_RAD_SIM_SQRD, memSize);
    // check if the allocation was successful
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "cudaMalloc failed: d_RAD_SIM_SQRD\n");
        // terminate the program
        fatalError();
	}

    // amount of memory to allocate
    memSize = sizeof(float);
    // add to the total memory allocated on the GPU
    totalGPUmem += memSize;	
	// allocate GPU memory for the half time step
	cudaStatus = cudaMalloc(&d_HALF_TIME_STEP, memSize);
    // check if the allocation was successful
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "cudaMalloc failed: d_HALF_TIME_STEP\n");
        // terminate the program
        fatalError();
	}

    // amount of memory to allocate
    memSize = sizeof(float);
    // add to the total memory allocated on the GPU
    totalGPUmem += memSize;	
	// allocate GPU memory for the ion ion acceleration multiplier 
	cudaStatus = cudaMalloc(&d_ION_ION_ACC_MULT, memSize);
    // check if the allocation was successful
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "cudaMalloc failed: d_ION_ION_ACC_MULT\n");
        // terminate the program
        fatalError();
	}

    // amount of memory to allocate
    memSize = sizeof(float);
    // add to the total memory allocated on the GPU
    totalGPUmem += memSize;	
	// allocate GPU memory for the ion dust acceleration multiplier 
	cudaStatus = cudaMalloc(&d_ION_DUST_ACC_MULT, memSize);
    // check if the allocation was successful
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "cudaMalloc failed: d_ION_DUST_ACC_MULT\n");
        // terminate the program
        fatalError();
	}

    // amount of memory to allocate
    memSize = memFloat3Dust;
    // add to the total memory allocated on the GPU
    totalGPUmem += memSize;	
	// allocate GPU memory for the dust positions
	cudaStatus = cudaMalloc(&d_posDust, memSize);
    // check if the allocation was successful
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "cudaMalloc failed: d_posDust\n");
        // terminate the program
        fatalError();
	}

    // amount of memory to allocate
    memSize = memFloat3Ion;
    // add to the total memory allocated on the GPU
    totalGPUmem += memSize;	
	// allocate GPU memory for the ion positions
	cudaStatus = cudaMalloc(&d_posIon, memSize);
    // check if the allocation was successful
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "cudaMalloc failed: d_posIon\n");
        // terminate the program
        fatalError();
	}

    // amount of memory to allocate
    memSize = memFloat3Ion;
    // add to the total memory allocated on the GPU
    totalGPUmem += memSize;	
	// allocate GPU memory for the ion velocities 
	cudaStatus = cudaMalloc(&d_velIon, memSize);
    // check if the allocation was successful
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "cudaMalloc failed: d_velIon\n");
        // terminate the program
        fatalError();
	}

    // amount of memory to allocate
    memSize = memFloat3Ion;
    // add to the total memory allocated on the GPU
    totalGPUmem += memSize;	
	// allocate GPU memory for the ion accelerations
	cudaStatus = cudaMalloc(&d_accIon, memSize);
    // check if the allocation was successful
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "cudaMalloc failed: d_accIon\n");
        // terminate the program
        fatalError();
	}

    // amount of memory to allocate
    memSize = blocksPerGridIon * DIM_BLOCK * sizeof(curandState_t);
    // add to the total memory allocated on the GPU
    totalGPUmem += memSize;	
	// allocate GPU memory for random states
	cudaStatus = cudaMalloc(&randStates, memSize);
	// check if the allocation was successful
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "cudaMalloc failed: randStates\n");
        // terminate the program
        fatalError();
	}

	// copy the external electric acceleration multiplier to the GPU
	cudaStatus = cudaMemcpy(d_EXTERN_ELC_MULT, &EXTERN_ELC_MULT,
		sizeof(float), cudaMemcpyHostToDevice);
    // check if the memory copy was successful
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "cudaMemcpy failed: d_EXTERN_ELC_MULT\n");
        // terminate the program
        fatalError();
	}
	
	// copy pi to the GPU
	cudaStatus = cudaMemcpy(d_PI, &PI, sizeof(float), cudaMemcpyHostToDevice);
	// check if the memory copy was successful
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "cudaMemcpy failed: d_PI\n");
        // terminate the program
        fatalError();
	}

	// copy the mass of a single ion to the GPU
	cudaStatus = cudaMemcpy(d_MASS_SINGLE_ION, &MASS_SINGLE_ION,
		sizeof(float), cudaMemcpyHostToDevice);
	// check if the memory copy was successful
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "cudaMemcpy failed: d_MASS_SINGLE_ION\n");
        // terminate the program
        fatalError();
	}
	
	// copy the boltzmann constant to the GPU
	cudaStatus = cudaMemcpy(d_BOLTZMANN, &BOLTZMANN,
		sizeof(float), cudaMemcpyHostToDevice);
	// check if the memory copy was successful
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "cudaMemcpy failed: d_BOLTZMANN\n");
        // terminate the program
        fatalError();
	}
	// copy the ion temperature to the GPU
	cudaStatus = cudaMemcpy(d_TEMP_ION, &TEMP_ION,
		sizeof(float), cudaMemcpyHostToDevice);
	// check if the memory copy was successful
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "cudaMemcpy failed: d_TEMP_ION\n");
        // terminate the program
        fatalError();
	}
	
	// copy the ion drift velocity to the GPU
	cudaStatus = cudaMemcpy(d_DRIFT_VEL_ION, &DRIFT_VEL_ION,
		sizeof(float), cudaMemcpyHostToDevice);
	// check if the memory copy was successful
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "cudaMemcpy failed: d_DRIFT_VEL_ION\n");
        // terminate the program
        fatalError();
	}
	
	// copy the dust radius squared to the GPU
	cudaStatus = cudaMemcpy(d_RAD_DUST_SQRD, &RAD_DUST_SQRD,
		sizeof(float), cudaMemcpyHostToDevice);
	// check if the memory copy was successful
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "cudaMemcpy failed: d_RAD_DUST_SQRD\n");
        // terminate the program
        fatalError();
	}

	// copy the inverse debye to the GPU
	cudaStatus = cudaMemcpy(d_INV_DEBYE, &INV_DEBYE, 
		sizeof(float), cudaMemcpyHostToDevice);
	// check if the memory copy was successful
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "cudaMemcpy failed: d_INV_DEBYE\n");
        // terminate the program
        fatalError();
	}
	
	// copy the mach number to the GPU
	cudaStatus = cudaMemcpy(d_MACH, &MACH, 
        sizeof(float), cudaMemcpyHostToDevice);
	// check if the memory copy was successful
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "cudaMemcpy failed: d_MACH\n");
        // terminate the program
        fatalError();
	}
	
	// copy the softening radius squared to the GPU
	cudaStatus = cudaMemcpy(d_SOFT_RAD_SQRD, &SOFT_RAD_SQRD, 
		sizeof(float), cudaMemcpyHostToDevice);
	// check if the memory copy was successful
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "cudaMemcpy failed: d_SOFT_RAD_SQRD\n");
        // terminate the program
        fatalError();
	}
    
	// copy the simulation radius to the GPU
	cudaStatus = cudaMemcpy(d_RAD_SIM, &RAD_SIM, 
        sizeof(float), cudaMemcpyHostToDevice);
	// check if the memory copy was successful
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "cudaMemcpy failed: d_RAD_SIM\n");
        // terminate the program
        fatalError();
	}
	
	// copy the simulation radius squared to the GPU
	cudaStatus = cudaMemcpy(d_RAD_SIM_SQRD, &RAD_SIM_SQRD, 
		sizeof(float), cudaMemcpyHostToDevice);
	// check if the memory copy was successful
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "cudaMemcpy failed: d_RAD_SIM_SQRD\n");
        // terminate the program
        fatalError();
	}

	// copy the dust charges to the GPU
	cudaStatus = cudaMemcpy(d_chargeDust, chargeDust, 
		memFloatDust, cudaMemcpyHostToDevice);
	// check if the memory copy was successful
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "cudaMemcpy failed: d_chargeDust\n");
        // terminate the program
        fatalError();
	}
	
	// copy the half time step to the GPU
	cudaStatus = cudaMemcpy(d_HALF_TIME_STEP, &HALF_TIME_STEP, 
		sizeof(float), cudaMemcpyHostToDevice);
	// check if the memory copy was successful
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "cudaMemcpy failed: d_HALF_TIME_STEP\n");
        // terminate the program
        fatalError();
	}
	
	// copy the ion ion acceleration multiplier to the GPU
	cudaStatus = cudaMemcpy(d_ION_ION_ACC_MULT, &ION_ION_ACC_MULT, 
		sizeof(float), cudaMemcpyHostToDevice);
	// check if the memory copy was successful
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "cudaMemcpy failed: d_ION_ION_ACC_MULT\n");
        // terminate the program
        fatalError();
	}

	// copy the ion dust acceleration multiplier to the GPU
	cudaStatus = cudaMemcpy(d_ION_DUST_ACC_MULT, &ION_DUST_ACC_MULT,
		sizeof(float), cudaMemcpyHostToDevice);
	// check if the memory copy was successful
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "cudaMemcpy failed: d_ION_DUST_ACC_MULT\n");
        // terminate the program
        fatalError();
	}

	// copy the number of dust particles to the GPU
	cudaStatus = cudaMemcpy(d_NUM_DUST, &NUM_DUST, sizeof(int), 
		cudaMemcpyHostToDevice);
	// check if the memory copy was successful
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "cudaMemcpy failed: d_NUM_DUST\n");
        // terminate the program
        fatalError();
	}

	// copy the number of ions to the GPU
	cudaStatus = cudaMemcpy(d_NUM_ION, &NUM_ION, sizeof(int), 
		cudaMemcpyHostToDevice);
	// check if the memory copy was successful
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "cudaMemcpy failed: d_NUM_ION\n");
        // terminate the program
        fatalError();
	}

	// copy the number of divisions  in d_QCOM to the GPU
	cudaStatus = cudaMemcpy(d_NUM_DIV_QTH, &NUM_DIV_QTH, sizeof(int), 
		cudaMemcpyHostToDevice);
	// check if the memory copy was successful
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "cudaMemcpy failed: d_NUM_DIV_QTH\n");
        // terminate the program
        fatalError();
	}

	// copy the electron temperature to the GPU
	cudaStatus = cudaMemcpy(d_TEMP_ELC, &TEMP_ELC, sizeof(float), 
		cudaMemcpyHostToDevice);
	// check if the memory copy was successful
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "cudaMemcpy failed: d_TEMP_ELC\n");
        // terminate the program
        fatalError();
	}

	// copy the sound speed to the GPU
	cudaStatus = cudaMemcpy(d_SOUND_SPEED, &SOUND_SPEED, sizeof(float), 
		cudaMemcpyHostToDevice);
	// check if the memory copy was successful
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "cudaMemcpy failed: d_SOUND_SPEED\n");
        // terminate the program
        fatalError();
	}
	
	// copy the number of divisions in d_VCOM to the GPU
	cudaStatus = cudaMemcpy(d_NUM_DIV_VEL, &NUM_DIV_VEL, sizeof(int), 
		cudaMemcpyHostToDevice);
	// check if the memory copy was successful
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "cudaMemcpy failed: d_NUM_DIV_VEL\n");
        // terminate the program
        fatalError();
	}
		
	// copy the dust positions to the GPU
	cudaStatus = cudaMemcpy(d_posDust, posDust, 
        memFloat3Dust, cudaMemcpyHostToDevice);
	// check if the memory copy was successful
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "cudaMemcpy failed: d_posDust\n");
        // terminate the program
        fatalError();
	}

	// copy the ion positions to the GPU
	cudaStatus = cudaMemcpy(d_posIon, posIon, 
        memFloat3Ion, cudaMemcpyHostToDevice);
	// check if the memory copy was successful
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "cudaMemcpy failed: d_posIon\n");
        // terminate the program
        fatalError();
	}

	// copy the ion velocities to the GPU
	cudaStatus = cudaMemcpy(d_velIon, velIon, 
        memFloat3Ion, cudaMemcpyHostToDevice);
	// check if the memory copy was successful
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "cudaMemcpy failed: d_velIon\n");
        // terminate the program
        fatalError();
	}

	// copy the ion accelerations to the GPU
	cudaStatus = cudaMemcpy(d_accIon, accIon, 
        memFloat3Ion, cudaMemcpyHostToDevice);
	// check if the memory copy was successful
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "cudaMemcpy failed: d_accIon\n");
        // terminate the program
        fatalError();
	}

	// generate all of the random states on the GPU
	init <<< DIM_BLOCK * blocksPerGridIon, 1 >>> (time(0), randStates);

	// Check for any errors launching the init kernel
	cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "kernel launch failed: init\n");
        // terminate the program
        fatalError();
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns 
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "Kernel failed: init");
        fprintf(stderr, "Error code: %d\n", cudaStatus);
        // terminate the program
        fatalError();
	}
	
	// initialize variables needed for injecting ions with the Piel 2017 method
	initInjectIonPiel(
		NUM_DIV_QTH, 
		NUM_DIV_VEL, 
		TEMP_ELC, 
		TEMP_ION, 
		DRIFT_VEL_ION, 
		MACH,
		MASS_SINGLE_ION,
		BOLTZMANN,
		PI,
		d_QCOM,
		d_VCOM,
		d_GCOM,
		debugMode,
		debugFile);

    if (debugMode){
        debugFile << '\n' << "-- GPU Allocated Memory --" << '\n'
            << "total memory allocated " << totalGPUmem << '\n' << '\n';
            
        debugFile.flush();
    }
        
	/*************************
            time step
	*************************/

	// synchronize threads and check for errors before entering timestep
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaStatus));
        // terminate the program
        fatalError();
	}
    
    // time step
    for (int i = 1; i <= NUM_TIME_STEP; i++)
	{

        // print the time step number to the status file 
		statusFile << i << ": ";

        // loop over all of the commands for each time step
		for (int j = 0; j < numCommands; j++) {
			
			// perform a leapfrog integration
			if (commands[j] == 1) {
                
                // print command number to status file 
				statusFile << "1 ";

				// perform a leapfrog integration for the ions 
				leapfrog <<< blocksPerGridIon, DIM_BLOCK >>>
					(d_posIon, d_velIon, d_accIon, d_HALF_TIME_STEP);

				// check for any errors launching the kernel
				cudaStatus = cudaGetLastError();
				if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "ERROR on line number %d in file %s\n", 
                        __LINE__, __FILE__);
                    fprintf(stderr, "kernel launch failed: leapfrog\n");
					fprintf(stderr, "Error code : %s\n\n", cudaStatus);
                    // terminate the program
                    fatalError();
				}

				// synchronize threads and check for errors
				cudaStatus = cudaDeviceSynchronize();
				if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "ERROR on line number %d in file %s\n", 
                        __LINE__, __FILE__);
                    fprintf(stderr, "Kernel failed: leapfrog");
					fprintf(stderr, "Error code: %d\n", cudaStatus);
                    // terminate the program
                    fatalError();
				}                
			}

			// calculate the acceleration due to ion-ion interactions
			else if (commands[j] == 2){
				
                // print the command number to the status file 
				statusFile << "2 ";

				// calculate the forces between all ions
				calcIonIonAcc 
                    <<< blocksPerGridIon, 
                        DIM_BLOCK, 
                        sizeof(float3) * DIM_BLOCK >>>
                       (d_posIon, 
                        d_accIon, 
                        d_NUM_ION, 
                        d_SOFT_RAD_SQRD, 
                        d_ION_ION_ACC_MULT, 
                        d_INV_DEBYE);

                // check for any errors launching the kernel
				cudaStatus = cudaGetLastError();
				if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "ERROR on line number %d in file %s\n", 
                        __LINE__, __FILE__);
                    fprintf(stderr, "kernel launch failed: calcIonIonAcc\n");
					fprintf(stderr, "Error code : %s\n\n", cudaStatus);
                    // terminate the program
                    fatalError();
				}

				// synchronize threads and check for errors
				cudaStatus = cudaDeviceSynchronize();
				if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "ERROR on line number %d in file %s\n", 
                        __LINE__, __FILE__);
                    fprintf(stderr, "Kernel failed: calcIonIonAcc");
					fprintf(stderr, "Error code: %d\n", cudaStatus);
                    // terminate the program
                    fatalError();
				} 
			}

			// calculate the acceleration due to ion-dust interactions
			else if (commands[j] == 3){
				
                // print the command number to the status file 
				statusFile << "3 ";

				// calculate ion dust accelerations
				calcIonDustAcc <<< blocksPerGridIon, DIM_BLOCK >>> 
                       (d_posIon, 
                        d_accIon, 
                        d_posDust,
                        d_NUM_ION,
                        d_NUM_DUST, 
                        d_SOFT_RAD_SQRD, 
                        d_ION_DUST_ACC_MULT, 
                        d_INV_DEBYE, 
                        d_chargeDust);

                // check for any errors launching the kernel
				cudaStatus = cudaGetLastError();
				if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "ERROR on line number %d in file %s\n", 
                        __LINE__, __FILE__);
                    fprintf(stderr, 
                        "Kernel launch failed: calcIonDustAcc\n");
					fprintf(stderr, "Error code : %s\n\n", cudaStatus);
                    // terminate the program
                    fatalError();
				}

				// synchronize threads and check for errors
				cudaStatus = cudaDeviceSynchronize();
				if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "ERROR on line number %d in file %s\n", 
                        __LINE__, __FILE__);
                    fprintf(stderr, "Kernel failed: calcIonDustAcc");
					fprintf(stderr, "Error code: %d\n", cudaStatus);
                    // terminate the program
                    fatalError();
				} 
			}
			
			// calculate the ion accelerations due to the ions outside of 
            // the simulation sphere
			else if (commands[j] == 4) {
				
                // print the command number to the status file 
				statusFile << "4 ";

				// calculate the forces between all ions
				calcExtrnElcAcc <<< blocksPerGridIon, DIM_BLOCK >>>
					(d_accIon, d_posIon, d_EXTERN_ELC_MULT, d_INV_DEBYE);

                // check for any errors launching the kernel
				cudaStatus = cudaGetLastError();
				if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "ERROR on line number %d in file %s\n", 
                        __LINE__, __FILE__);
                    fprintf(stderr, 
                        "Kernel launch failed: calcExtrnElcAcc\n");
					fprintf(stderr, "Error code : %s\n\n", cudaStatus);
                    // terminate the program
                    fatalError();
				}

				// synchronize threads and check for errors
				cudaStatus = cudaDeviceSynchronize();
				if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "ERROR on line number %d in file %s\n", 
                        __LINE__, __FILE__);
                    fprintf(stderr, "Kernel failed: calcExtrnElcAcc");
					fprintf(stderr, "Error code: %d\n", cudaStatus);
                    // terminate the program
                    fatalError();
				} 				
			}
			
			// copy ion positions to the host
			else if (commands[j] == 5) {
				
                // print the command number to the status file 
				statusFile << "5 ";

				// copy ion positions to host
				cudaStatus = cudaMemcpy(posIon, d_posIon, 
                    memFloat3Ion, cudaMemcpyDeviceToHost);
                    
                // check if the memory copy was successful
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "ERROR on line number %d in file %s\n", 
                        __LINE__, __FILE__);
                    fprintf(stderr, "cudaMemcpy failed: d_posIon\n");
                    // terminate the program
                    fatalError();
                }
			}

			// print the position of an ion to the trace file
			else if (commands[j] == 6) {
				
                // print the command number to the status file 
				statusFile << "6 ";

				// print the position of the specified ion to the trace file
				traceFile << posIon[ionTraceIndex].x;
				traceFile << ", " << posIon[ionTraceIndex].y;
				traceFile << ", " << posIon[ionTraceIndex].z << std::endl;
			}


			// copy the ion accelerations to the host
			else if (commands[j] == 7) {
                
                // print the command number to the status file 
				statusFile << "7 ";

				// copy ion positions to host
				cudaStatus = cudaMemcpy(accIon, d_accIon, 
                    memFloat3Ion, cudaMemcpyDeviceToHost);
                    
                // check if the memory copy was successful
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "ERROR on line number %d in file %s\n", 
                        __LINE__, __FILE__);
                    fprintf(stderr, "cudaMemcpy failed: accIon\n");
                    // terminate the program
                    fatalError();
                }
			}

			// print the acceleration of an ion to the trace file 
			else if (commands[j] == 8) {
                
                // print the command number to the status file 
				statusFile << "8 ";

				// print the acceleration of the specified ion to the trace file
				traceFile << accIon[ionTraceIndex].x;
				traceFile << ", " << accIon[ionTraceIndex].y;
				traceFile << ", " << accIon[ionTraceIndex].z << std::endl;
			}

			// copy the ion velocities to the host
			else if (commands[j] == 9) {
				
				statusFile << "9 ";

				// copy ion velocities to host
				cudaStatus = cudaMemcpy(velIon, d_velIon, 
                    memFloat3Ion, cudaMemcpyDeviceToHost);
                    
                // check if the memory copy was successful
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "ERROR on line number %d in file %s\n", 
                        __LINE__, __FILE__);
                    fprintf(stderr, "cudaMemcpy failed: d_velIon\n");
                    // terminate the program
                    fatalError();
                }
			}

			// print the velocity of an ion to the trace file 
			else if (commands[j] == 10) {
				
                // print the command number to the status file 
				statusFile << "10 ";

				// print the velocity of the specified ion to the trace file
				traceFile << velIon[ionTraceIndex].x;
				traceFile << ", " << velIon[ionTraceIndex].y;
				traceFile << ", " << velIon[ionTraceIndex].z << std::endl;
			}
			
			// check if any ions are outside of the simulation sphere
			else if (commands[j] == 11){
				
                // print the command number to the status file 
				statusFile << "11 ";
				
				// check if any ions are outside of the simulation sphere
				checkIonSphereBounds <<< blocksPerGridIon, DIM_BLOCK >>> 
                       (d_posIon, d_boundsIon, d_RAD_SIM_SQRD);

                // check for any errors launching the kernel
				cudaStatus = cudaGetLastError();
				if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "ERROR on line number %d in file %s\n", 
                        __LINE__, __FILE__);
                    fprintf(stderr, 
                        "Kernel launch failed: checkIonSphereBounds\n");
					fprintf(stderr, "Error code : %s\n\n", cudaStatus);
                    // terminate the program
                    fatalError();
				}

				// synchronize threads and check for errors
				cudaStatus = cudaDeviceSynchronize();
				if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "ERROR on line number %d in file %s\n", 
                        __LINE__, __FILE__);
                    fprintf(stderr, "Kernel failed: checkIonSphereBounds");
					fprintf(stderr, "Error code: %d\n", cudaStatus);
                    // terminate the program
                    fatalError();
				}
			}
			
			// check if any ions are inside of a dust particle
			else if (commands[j] == 12) {
                
                // print the command number to the status file 
				statusFile << "12 ";
				
				// check if any ions are inside a dust particle 
				checkIonDustBounds <<< blocksPerGridIon, DIM_BLOCK >>> 
                       (d_posIon,
                        d_boundsIon,
                        d_RAD_DUST_SQRD,
                        d_NUM_DUST,
                        d_posDust);

                // check for any errors launching the kernel
				cudaStatus = cudaGetLastError();
				if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "ERROR on line number %d in file %s\n", 
                        __LINE__, __FILE__);
                    fprintf(stderr, 
                        "Kernel launch failed: checkIonSphereBounds\n");
					fprintf(stderr, "Error code : %s\n\n", cudaStatus);
                    // terminate the program
                    fatalError();
				}

				// synchronize threads and check for errors
				cudaStatus = cudaDeviceSynchronize();
				if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "ERROR on line number %d in file %s\n", 
                        __LINE__, __FILE__);
                    fprintf(stderr, "Kernel failed: checkIonSphereBounds");
					fprintf(stderr, "Error code: %d\n", cudaStatus);
                    // terminate the program
                    fatalError();
				}
			}
			
			// inject ions as in Piel 2017
			else if (commands[j] == 13){
                
                // print the command number to the status file 
				statusFile << "13 ";
				
				// inject ions into the simulation sphere 
				injectIonPiel <<< blocksPerGridIon, DIM_BLOCK >>> 
                       (d_posIon,
                        d_velIon,
                        d_accIon,
                        randStates,
                        d_RAD_SIM,
                        d_boundsIon,
                        d_GCOM,
                        d_QCOM,
                        d_VCOM,
                        d_NUM_DIV_QTH,
                        d_NUM_DIV_VEL,
                        d_SOUND_SPEED,
                        d_TEMP_ION,
                        d_PI,
                        d_TEMP_ELC,
                        d_MACH,
                        d_MASS_SINGLE_ION,
                        d_BOLTZMANN);

	            // check for any errors launching the kernel
				cudaStatus = cudaGetLastError();
				if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "ERROR on line number %d in file %s\n", 
                        __LINE__, __FILE__);
                    fprintf(stderr, 
                        "Kernel launch failed: checkIonSphereBounds\n");
					fprintf(stderr, "Error code : %s\n\n", cudaStatus);
                    // terminate the program
                    fatalError();
				}

				// synchronize threads and check for errors
				cudaStatus = cudaDeviceSynchronize();
				if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "ERROR on line number %d in file %s\n", 
                        __LINE__, __FILE__);
                    fprintf(stderr, "Kernel failed: checkIonSphereBounds");
					fprintf(stderr, "Error code: %d\n", cudaStatus);
                    // terminate the program
                    fatalError();
				}
			}
			
			// reset the ion bounds flag to 0
			else if (commands[j] == 14){
                
                // print the command number to the status file 
				statusFile << "14 ";
				
				// reset the ion bounds flag to 0 
				resetIonBounds <<< blocksPerGridIon, DIM_BLOCK >>>
                    (d_boundsIon);

	            // check for any errors launching the kernel
				cudaStatus = cudaGetLastError();
				if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "ERROR on line number %d in file %s\n", 
                        __LINE__, __FILE__);
                    fprintf(stderr, 
                        "Kernel launch failed: checkIonSphereBounds\n");
					fprintf(stderr, "Error code : %s\n\n", cudaStatus);
                    // terminate the program
                    fatalError();
				}

				// synchronize threads and check for errors
				cudaStatus = cudaDeviceSynchronize();
				if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "ERROR on line number %d in file %s\n", 
                        __LINE__, __FILE__);
                    fprintf(stderr, "Kernel failed: checkIonSphereBounds");
					fprintf(stderr, "Error code: %d\n", cudaStatus);
                    // terminate the program
                    fatalError();
				}
			}

			// update the charge on the dust grains 
			else if (commands[j] == 15) {
				// not implemented  
                //for (int i = 0; i < NUM_DUST; i++){
                //    chargeDust[i] += htifi;
                //}
			}
			
			// copy d_ionBounds to the host
			else if (commands[j] == 16){
				
                // print the command number to the status file 
				statusFile << "16 ";
				
				// copy ion bounds to host
				cudaStatus = cudaMemcpy(boundsIon, d_boundsIon, 
                    NUM_ION * sizeof(int), cudaMemcpyDeviceToHost);
                    
                // check if the memory copy was successful
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "ERROR on line number %d in file %s\n", 
                        __LINE__, __FILE__);
                    fprintf(stderr, "cudaMemcpy failed: d_boundsIon\n");
                    // terminate the program
                    fatalError();
                }
			}
			
			// copy the dust charge to the device 
			else if (commands[j] == 17){
				
                // print the command number to the status file 
				statusFile << "17 ";
				
				// copy the dust charge to the GPU
				cudaStatus = cudaMemcpy(d_chargeDust, chargeDust, 
                    memFloatDust, cudaMemcpyHostToDevice);
                    
                // check if the memory copy was successful
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "ERROR on line number %d in file %s\n", 
                        __LINE__, __FILE__);
                    fprintf(stderr, "cudaMemcpy failed: d_chargeDust\n");
                    // terminate the program
                    fatalError();
                }
			}
			
			// copy dust charge to the host
			else if (commands[j] == 18){
				
                // print the command number to the status file 
				statusFile << "18 ";
				
				// copy dust charge to host
				cudaStatus = cudaMemcpy(chargeDust, d_chargeDust, 
                    memFloatDust, cudaMemcpyDeviceToHost);
                    
                // check if the memory copy was successful
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "ERROR on line number %d in file %s\n", 
                        __LINE__, __FILE__);
                    fprintf(stderr, "cudaMemcpy failed: d_chargeDust\n");
                    // terminate the program
                    fatalError();
                }
			}
			
			// print the dust charge to the trace file 
			else if (commands[j] == 19) {
                
                // print the command number to the status file 
				statusFile << "19 ";

				// print all the dust charges to the trace file
				for (int k = 0; k < NUM_DUST; k++){
					traceFile << chargeDust[k];
					traceFile << ", ";
				}
				traceFile << std::endl;
			}

			// calculate the ion currents to the dust particles 
			else if (commands[j] == 20) {
				
                // print the command number to the status file 
                statusFile << "20 ";
                
				// set initial currents to 0
				for (int i = 0; i < NUM_DUST; i++){
					ionCurrent[i] = 0;
				}
				
				// loop over all of the ion bounds 
				for (int i = 0; i < NUM_ION; i++){
					// if the ion was collected by a dust particle 
					if (boundsIon[i] > 0){
						// increase the current to that dust particle by 1
						ionCurrent[boundsIon[i] - 1] += 1;
					}
				}
			}
			
			// print the ion current to a dust particle to the trace file 
			else if (commands[j] == 21) {
                
                // print the command number to the status file 
                statusFile << "21 ";
                
                // print the ion current to the first dust particle to 
                // the trace file 
				traceFile << ionCurrent[0] << std::endl;
			}
				
			// if the command number does not exist throw an error
			else {
				
				// output an error message
				fprintf(stderr, "ERROR on line number %d in file %s\n", 
                    __LINE__, __FILE__);
				fprintf(stderr, "Command number %d does not exist\n\n", 
                    commands[j]);

				// terminate the program 
				fatalError();
			}
		}

		statusFile << "|" << std::endl;

	} // end time step
	
	if (debugMode)
	{
		// print the index of the traced ion to the debuging file
		debugFile << "Single ion trace index: " << ionTraceIndex << "\n\n";
	}

    /***********************
           save data 
	***********************/
	
    // copy ion positions to host
	cudaStatus = cudaMemcpy(posIon, d_posIon, 
        memFloat3Ion, cudaMemcpyDeviceToHost);
    // check if the memory copy was successful
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
        fprintf(stderr, "cudaMemcpy failed: d_posIon\n");
        // terminate the program
        fatalError();
    }
    
	// copy ion velocities to host
	cudaStatus = cudaMemcpy(velIon, d_velIon, 
        memFloat3Ion, cudaMemcpyDeviceToHost);
    // check if the memory copy was successful
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
        fprintf(stderr, "cudaMemcpy failed: d_velIon\n");
        // terminate the program
        fatalError();
    }
	
	// copy dust charges to the host 
	cudaStatus = cudaMemcpy(chargeDust, d_chargeDust, 
        memFloatDust, cudaMemcpyDeviceToHost);
    // check if the memory copy was successful
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
        fprintf(stderr, "cudaMemcpy failed: d_chargeDust\n");
        // terminate the program
        fatalError();
    }

	// synchronize threads and check for errors
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
        fprintf(stderr, "Unknown CUDA failure");
        fprintf(stderr, "Error code: %d\n", cudaStatus);
        // terminate the program
        fatalError();
	}
    
	// print final ion positions to the ionPosFile	
    // loop over all of the positions 
	for (int i = 0; i < NUM_ION; i++) {
		// print the ion position
		ionPosFile << posIon[i].x;
		ionPosFile << ", " << posIon[i].y;
		ionPosFile << ", " << posIon[i].z << std::endl;
	}

	// print the final dust charges to the dustChargeFile
    // loop over all of the dust particles 
	for (int i = 0; i < NUM_DUST; i++) {
		// print the dust charge 
		dustChargeFile << chargeDust[i] << std::endl;
	}

	// print the final dust positions to the dustPosFile
    // loop over all of the dust particles 
	for (int i = 0; i < NUM_DUST; i++) {
		// print the dust positions 
		dustPosFile << posDust[i].x << ", ";
		dustPosFile << posDust[i].y << ", ";
		dustPosFile << posDust[i].z << std::endl;
	}

	if (debugMode)
	{

		debugFile << "-- Final Ion Positions: First 20 Ions --" << std::endl;
		for (int i = 0; i < 20; i++)
		{
			debugFile << "#: "  << i 
					  << " X: " << posIon[i].x
					  << " Y: " << posIon[i].y
					  << " Z: " << posIon[i].z
					  << std::endl;
		}
        
		debugFile << '\n' << "-- Final Ion Positions: Last 20 Ions --" << '\n';
		for (int i = 1; i <= 20; i++)
		{
			int ID = NUM_ION - i;

			debugFile << "#: "  << ID
				      << " X: " << posIon[ID].x
				      << " Y: " << posIon[ID].y
				      << " Z: " << posIon[ID].z
				      << std::endl;
		}
        
		debugFile << '\n' 
                  << "-- Final Ion Velocities: First 20 Ions --" << '\n';
		for (int i = 0; i < 20; i++)
		{
			debugFile << "#: " << i
				<< " X: " << velIon[i].x
				<< " Y: " << velIon[i].y
				<< " Z: " << velIon[i].z
				<< std::endl;
		}
        
		debugFile << '\n' 
                  << "-- Final Ion Velocities: Last 20 Ions --" << '\n';
		for (int i = 1; i <= 20; i++)
		{
			int ID = NUM_ION - i;

			debugFile << "#: " << ID
				<< " X: " << velIon[ID].x
				<< " Y: " << velIon[ID].y
				<< " Z: " << velIon[ID].z
				<< std::endl;
		}
        
		debugFile << '\n' 
                  << "-- Final Ion Accelerations: First 20 Ions --" << '\n';
		for (int i = 0; i < 20; i++)
		{
			debugFile << "#: " << i
				<< " X: " << accIon[i].x
				<< " Y: " << accIon[i].y
				<< " Z: " << accIon[i].z
				<< std::endl;
		}
        
		debugFile << '\n' 
                  << "-- Final Ion Accelerations: Last 20 Ions --" << '\n';
		for (int i = 1; i <= 20; i++)
		{
			int ID = NUM_ION - i;

			debugFile << "#: " << ID
				<< " X: " << accIon[ID].x
				<< " Y: " << accIon[ID].y
				<< " Z: " << accIon[ID].z
				<< std::endl;
		}
		debugFile << std::endl;
	}

	/*************************
		  check device 
		  "constants"
	*************************/

	// temporary host pointers to copy device constants to
	float* floatTestVal = (float*)malloc(sizeof(float));
	int* intTestVal = (int*)malloc(sizeof(int));

	// copy the number of divisions  in d_QCOM to the host
	cudaStatus = cudaMemcpy(intTestVal, d_NUM_DIV_QTH, 
		sizeof(int), cudaMemcpyDeviceToHost);
    // check if the memory copy was successful
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
        fprintf(stderr, "cudaMemcpy failed: d_NUM_DIV_QTH\n");
        // terminate the program
        fatalError();
    }
	// check if the device d_NUM_DIV_QTH changed 
	if ((*intTestVal - NUM_DIV_QTH) != 0) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "Const Device Value Changed: d_NUM_DIV_QTH\n");
		fprintf(stderr, "from %u", NUM_DIV_QTH);
		fprintf(stderr, " to %u\n", *intTestVal);
        // terminate the program
        fatalError();
	}
	
	// copy the number of divisions  in d_VCOM to the host
	cudaStatus = cudaMemcpy(intTestVal, d_NUM_DIV_VEL, 
		sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
        fprintf(stderr, "cudaMemcpy failed: d_NUM_DIV_VEL\n");
        // terminate the program
        fatalError();
    }
	// check if the device d_NUM_DIV_VEL changed 
	if ((*intTestVal - NUM_DIV_VEL) != 0) {
		fprintf(stderr, "Const Device Value Changed: d_NUM_DIV_VEL\n");
		fprintf(stderr, "from %u", NUM_DIV_VEL);
		fprintf(stderr, " to %u\n", *intTestVal);
        // terminate the program
        fatalError();
	}
	
	// copy the number of ions to the host
	cudaStatus = cudaMemcpy(intTestVal, d_NUM_ION, 
		sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
        fprintf(stderr, "cudaMemcpy failed: d_NUM_ION\n");
        // terminate the program
        fatalError();
    }
	// check if the device d_NUM_ION changed 
	if ((*intTestVal - NUM_ION) != 0) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "Const Device Value Changed: d_NUM_ION\n");
		fprintf(stderr, "from %u", NUM_ION);
		fprintf(stderr, " to %u\n", *intTestVal);
        // terminate the program
        fatalError();
	}

	// copy the number of dust particles to the host
	cudaStatus = cudaMemcpy(intTestVal, d_NUM_DUST, 
		sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
        fprintf(stderr, "cudaMemcpy failed: d_NUM_DUST\n");
        // terminate the program
        fatalError();
    }
	// check if the device d_NUM_DUST changed 
	if ((*intTestVal - NUM_DUST) != 0) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "Const Device Value Changed: d_NUM_DUST\n");
		fprintf(stderr, "from %u", NUM_DUST);
		fprintf(stderr, " to %u\n", *intTestVal);
        // terminate the program
        fatalError();
	}
	
	// copy the dust radii squared to the host
	cudaStatus = cudaMemcpy(floatTestVal, d_RAD_DUST_SQRD,
		sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
        fprintf(stderr, "cudaMemcpy failed: d_RAD_DUST_SQRD\n");
        // terminate the program
        fatalError();
    }
	// check if the device d_RAD_DUST_SQRD changed 
	if ((*floatTestVal - RAD_DUST_SQRD) != 0) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "Const Device Value Changed: d_RAD_DUST_SQRD\n");
		fprintf(stderr, "from %f", RAD_DUST_SQRD);
		fprintf(stderr, " to %f\n", *floatTestVal);
        // terminate the program
        fatalError();
	}

	// copy the external electric field multiplier to the host
	cudaStatus = cudaMemcpy(floatTestVal, d_EXTERN_ELC_MULT,
		sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
        fprintf(stderr, "cudaMemcpy failed: d_EXTERN_ELC_MULT\n");
        // terminate the program
        fatalError();
    }
	// check if the device d_EXTERN_ELC_MULT changed 
	if ((*floatTestVal - EXTERN_ELC_MULT) != 0) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "Const Device Value Changed: d_EXTERN_ELC_MULT\n");
		fprintf(stderr, "from %f", EXTERN_ELC_MULT);
		fprintf(stderr, " to %f\n", *floatTestVal);
        // terminate the program
        fatalError();
	}

	// copy the PI to the host
	cudaStatus = cudaMemcpy(floatTestVal, d_PI, 
		sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
        fprintf(stderr, "cudaMemcpy failed: d_PI\n");
        // terminate the program
        fatalError();
    }
	// check if the device d_PI changed 
	if ((*floatTestVal - PI) != 0) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "Const Device Value Changed: d_PI\n");
		fprintf(stderr, "from %f", PI);
		fprintf(stderr, " to %f\n", *floatTestVal);
        // terminate the program
        fatalError();
	}

	// copy the sound speed to the host
	cudaStatus = cudaMemcpy(floatTestVal, d_SOUND_SPEED, 
		sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
        fprintf(stderr, "cudaMemcpy failed: d_SOUND_SPEED\n");
        // terminate the program
        fatalError();
    }
	// check if the device d_SOUND_SPEED changed 
	if ((*floatTestVal - SOUND_SPEED) != 0) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "Const Device Value Changed: d_SOUND_SPEED\n");
		fprintf(stderr, "from %f", SOUND_SPEED);
		fprintf(stderr, " to %f\n", *floatTestVal);
        // terminate the program
        fatalError();
	}	
	
	// copy the electron temperature to the host
	cudaStatus = cudaMemcpy(floatTestVal, d_TEMP_ELC, 
		sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
        fprintf(stderr, "cudaMemcpy failed: d_TEMP_ELC\n");
        // terminate the program
        fatalError();
    }
	// check if the device d_TEMP_ELC changed 
	if ((*floatTestVal - TEMP_ELC) != 0) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "Const Device Value Changed: d_TEMP_ELC\n");
		fprintf(stderr, "from %f", TEMP_ELC);
		fprintf(stderr, " to %f\n", *floatTestVal);
        // terminate the program
        fatalError();
	}
	
	// copy the ion drift velocity to the host
	cudaStatus = cudaMemcpy(floatTestVal, d_DRIFT_VEL_ION, 
		sizeof(float), cudaMemcpyDeviceToHost);
   if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
        fprintf(stderr, "cudaMemcpy failed: d_DRIFT_VEL_ION\n");
        // terminate the program
        fatalError();
    }
	// check if the device d_DRIFT_VEL_ION changed 
	if ((*floatTestVal - DRIFT_VEL_ION) != 0) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "Const Device Value Changed: d_DRIFT_VEL_ION\n");
		fprintf(stderr, "from %f", DRIFT_VEL_ION);
		fprintf(stderr, " to %f\n", *floatTestVal);
        // terminate the program
        fatalError();
	}
	
	// copy the ion temperature to the host
	cudaStatus = cudaMemcpy(floatTestVal, d_TEMP_ION, 
		sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
        fprintf(stderr, "cudaMemcpy failed: d_TEMP_ION\n");
        // terminate the program
        fatalError();
    }
	// check if the device d_TEMP_ION changed 
	if ((*floatTestVal - TEMP_ION) != 0) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "Const Device Value Changed: d_TEMP_ION\n");
		fprintf(stderr, "from %i", TEMP_ION);
		fprintf(stderr, " to %i\n", *floatTestVal);
        // terminate the program
        fatalError();
	}
	
	// copy the ion ion acceleration multiplier to the host
	cudaStatus = cudaMemcpy(floatTestVal, d_ION_ION_ACC_MULT, 
		sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
        fprintf(stderr, "cudaMemcpy failed: d_ION_ION_ACC_MULT\n");
        // terminate the program
        fatalError();
    }
	// check if the device d_ION_ION_ACC_MULT changed 
	if ((*floatTestVal - ION_ION_ACC_MULT) != 0) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "Const Device Value Changed: d_ION_ION_ACC_MULT\n");
		fprintf(stderr, "from %f", ION_ION_ACC_MULT);
		fprintf(stderr, " to %f\n", *floatTestVal);
        // terminate the program
        fatalError();
	}

	// copy the mach number to the host
	cudaStatus = cudaMemcpy(floatTestVal, d_MACH, 
		sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
        fprintf(stderr, "cudaMemcpy failed: d_MACH\n");
        // terminate the program
        fatalError();
    }
	// check if the device d_MACH changed 
	if ((*floatTestVal - MACH) != 0) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "Const Device Value Changed: d_MACH\n");
		fprintf(stderr, "from %f", MACH);
		fprintf(stderr, " to %f\n", *floatTestVal);
        // terminate the program
        fatalError();
	}
	
	// copy the ion dust acceleration multiplier to the host
	cudaStatus = cudaMemcpy(floatTestVal, d_ION_DUST_ACC_MULT,
		sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
        fprintf(stderr, "cudaMemcpy failed: d_ION_DUST_ACC_MULT\n");
        // terminate the program
        fatalError();
    }
	// check if the host device d_ION_DUST_ACC_MULT changed 
	if ((*floatTestVal - ION_DUST_ACC_MULT) != 0) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "Const Device Value Changed: d_ION_DUST_ACC_MULT\n");
		fprintf(stderr, "from %f", ION_DUST_ACC_MULT);
		fprintf(stderr, " to %f\n", *floatTestVal);
        // terminate the program
        fatalError();
	}

	// copy the half time step to the host
	cudaStatus = cudaMemcpy(floatTestVal, d_HALF_TIME_STEP, 
		sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
        fprintf(stderr, "cudaMemcpy failed: d_HALF_TIME_STEP\n");
        // terminate the program
        fatalError();
    }
	//check if the device d_HALF_TIME_STEP changed 
	if ((*floatTestVal - HALF_TIME_STEP) != 0) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "Const Device Value Changed: d_HALF_TIME_STEP");
		fprintf(stderr, "from %f", HALF_TIME_STEP);
		fprintf(stderr, " to %f\n", *floatTestVal);
        // terminate the program
        fatalError();
	}

	// copy the squared simulation radius to the host
	cudaStatus = cudaMemcpy(floatTestVal, d_RAD_SIM_SQRD, 
		sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
        fprintf(stderr, "cudaMemcpy failed: d_RAD_SIM_SQRD\n");
        // terminate the program
        fatalError();
    }
	// check if the device d_RAD_SIM_SQRD changed 
	if ((*floatTestVal - RAD_SIM_SQRD) != 0) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "Const Device Value Changed: d_RAD_SIM_SQRD\n");
		fprintf(stderr, "from %f", RAD_SIM_SQRD);
		fprintf(stderr, " to %f\n", *floatTestVal);
        // terminate the program
        fatalError();
	}
	
	// copy the simulation radius to the device
	cudaStatus = cudaMemcpy(floatTestVal, d_RAD_SIM, 
        sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
        fprintf(stderr, "cudaMemcpy failed: d_RAD_SIM\n");
        // terminate the program
        fatalError();
    }
	//check if the device d_RAD_SIM changed 
	if ((*floatTestVal - RAD_SIM) != 0) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "Const Device Value Changed: d_RAD_SIM");
		fprintf(stderr, "from %f", RAD_SIM);
		fprintf(stderr, " to %f\n", *floatTestVal);
        // terminate the program
        fatalError();
	}

	// copy the inverse debye to the host
	cudaStatus = cudaMemcpy(floatTestVal, d_INV_DEBYE, 
        sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
        fprintf(stderr, "cudaMemcpy failed: d_INV_DEBYE\n");
        // terminate the program
        fatalError();
    }
	// check if the device d_INV_DEBYE changed 
	if ((*floatTestVal - INV_DEBYE) != 0) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "Const Device Value Changed: d_INV_DEBYE");
		fprintf(stderr, "from %f", INV_DEBYE);
		fprintf(stderr, " to %f\n", *floatTestVal);
        // terminate the program
        fatalError();
	}
	
	// copy the boltzmann number to the host
	cudaStatus = cudaMemcpy(floatTestVal, d_BOLTZMANN, 
		sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
        fprintf(stderr, "cudaMemcpy failed: d_BOLTZMANN\n");
        // terminate the program
        fatalError();
    }
	// check if the device d_BOLTZMANN
	if ((*floatTestVal - BOLTZMANN) != 0) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "Const Device Value Changed: d_BOLTZMANN\n");
		fprintf(stderr, "from %f", BOLTZMANN);
		fprintf(stderr, " to %f\n", *floatTestVal);
        // terminate the program
        fatalError();
	}
	
	// copy the ion mass to the host
	cudaStatus = cudaMemcpy(floatTestVal, d_MASS_SINGLE_ION, 
		sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
        fprintf(stderr, "cudaMemcpy failed: d_MASS_SINGLE_ION\n");
        // terminate the program
        fatalError();
    }
	// check if the device d_MASS_SINGLE_ION changed 
	if ((*floatTestVal - MASS_SINGLE_ION) != 0)
	{
        fprintf(stderr, "ERROR on line number %d in file %s\n", 
            __LINE__, __FILE__);
		fprintf(stderr, "Const Device Value Changed: d_MASS_SINGLE_ION\n");
		fprintf(stderr, "from %f", MASS_SINGLE_ION);
		fprintf(stderr, " to %f\n", *floatTestVal);
        // terminate the program
        fatalError();
	}
	
	/**********************
	      free memory 
	**********************/
	
	free(intTestVal);
	free(floatTestVal);
    free(posDust);
    free(chargeDust);
    free(commands);
    free(posIon);
    free(velIon);
    free(accIon);
    free(boundsIon);
	delete[] ionCurrent;
	
	/*************************
	      close files
	*************************/

	// close all opened files 
	paramFile.close();
	dustParamFile.close();
	timestepFile.close();
	debugFile.close();
	debugSpecificFile.close();
	traceFile.close();
	statusFile.close();
	ionPosFile.close();
	dustPosFile.close();
	dustChargeFile.close();
	paramOutFile.close();
	
	// exit
	return 0;
} 