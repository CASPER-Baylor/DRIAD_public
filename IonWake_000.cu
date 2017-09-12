
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
*		Last Contribution: 8/26/2017
*
* Description:
*	Handles the execution of the IonWake simulation. Provides a user interface in 
*	the form of input and output files.  Handles memory allocation and declaration 
*	for non-function specific variables on both the CPU host and GPU device. Includes 
*	a modular time-step for rapid development and testing of various time step 
*	schemes. For descriptions of the scope of the IonWake simulation as well as user interface, 
*	program input and output, and time-step options please see the respective 
*	sections of the README file. 
*
* Output:
*	Determined by user settings. For a complete description of all available program 
*	output please see the README file.
*
* Input:
*	Determined by user settings. For a complete description of all available program
*	input please see the README file.
*
* Implementation :
*	User input parameters are read in from input text files.  Each parameter file 
*	is processed serially. The function getUsserParams() saves all of the parameters 
*	in a file into a parameter array.  The values in the parameter array are then 
*	saved to the appropriate variables.  All parameter variables are constants.  
*	After reading in all the parameters, constant parameter dependent values are 
*	calculated, such as the Debye length.  Then host side memory is allocated.  
*	Device pointers are created and memory is allocated on the device. Initial values 
*	and constants are copied to the device.  Copying values to device variables 
*	created with the __constant__ declspec using cudaMemcpyToSymbol() resulted in 
*	the device constants evaluating to 0. Because of this “constant” variables on the 
*	device are held in global write accessible device memory. These “constant” 
*	variables follow the naming conventions for constant variables and at the end of 
*	the program are checked against the host constants and return an error if the 
*	values are different.  Two sets of random states are initialized on the device 
*	for use with curand() in device kernels.  One set contains as many states as the 
*	maximum number of blocks and the other contains as many as the maximum number of 
*	threads per block. The two sets of states are intended to be used in conjunction 
*	so that each thread in a kernel launch has a unique random number without having 
*	to save as many random states as threads.  All threads are synchronized and the 
*	program checks for any CUDA errors before entering the time step. The exact 
*	structure of the time-step depends on the user time-step settings. The general 
*	structure is: replace particles that have left the simulation region, calculate 
*	accelerations/forces, integrate, then save any tracked data.  Other than tracked 
*	data all calculations are performed on the device. Because of this there are no 
*	memory transfers between device calculations.  This leads to decrease program 
*	runtime as there are no memory transfer latencies, but any data that needs to 
*	be handled on the host during the time-step needs to be specifically transferred 
*	from the device to the host.  Threads are synchronized between time-step 
*	calculations that need to be calculated synchronously.  After the time-step, 
*	device data is copied to the host to be processed for output.  The precise 
*	processing and output is determined by the user output settings. Though each 
*	output generally follows the structure of: data selection, data processing, data 
*	output/saving.  Data selection is often done inline, while processing and 
*	output/saving are outsourced to functions.  All steps can be performed by the host 
*	or device.          
*
*	The program is permeated with output and error handling. The precise behavior of 
*	these sections is dependent upon user settings. These sections are delimited 
*	by // <option name> //. For example: // DEBUGGING //       
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
		   File Names
	*************************/

	// directory name where data ouput is saved
	std::string dataDirName = argv[2];
	// directory name where program inputs are read from
	std::string inputDirName = argv[1];
	// read in comand line argument for the run name
	std::string runName = argv[3];

	// create file names for input files
	std::string paramListDebugFileName = inputDirName + "param-list-debug.txt";
	std::string paramListFileName = inputDirName + "param-list.txt";
	std::string dustPosFileName = inputDirName + "dust-pos.txt";
	std::string timestepFileName = inputDirName + "timestep.txt";

	// create file names for output files
	std::string debugFileName = dataDirName + runName + "_debug-file.txt";
	std::string ionPosTraceName = dataDirName + "ionPosTrace.txt";
	std::string statusFileName = dataDirName + runName + "_status-file.txt";
	std::string numDenPlotName = dataDirName + runName + "_num-den-plot.bmp";

	/*************************
			Debugging
	*************************/

	// number of user defined parameters
	const int NUM_DEBUG_PARAMS = 8;

	// allocate memory for user parameters
	int* intParams = (int*)malloc(NUM_DEBUG_PARAMS * sizeof(int));

	// get user defined parameters
	getUsserParams(intParams, NUM_DEBUG_PARAMS, paramListDebugFileName.c_str());

	// assign user defined parameters
	const bool debugMode = intParams[0];
	const bool showParameters = intParams[1];
	const bool showConstants = intParams[2];
	const bool showOutputParameters = intParams[3];
	const bool showInitHostVars = intParams[4];
	const bool showFinalHostVars = intParams[5];
	const bool singleIonTraceMode = intParams[6];
	const int  ionTraceIndex = intParams[7];

	// free memory allocated for user parameters
	free(intParams);

	// create an output debugging text file
	std::ofstream debugFile;
	//create an output ion trace file
	std::ofstream ionPosTrace;

	// if debuging mode is set to true (on)
	if (debugMode)
	{
		// open the output text file
		debugFile.open(debugFileName.c_str());

		// set the output file to display 
		// 5 digits the right of the decimal 
		debugFile.precision(5);
		debugFile << std::showpoint;

		// if single ion trace is set to true (on)
		if (singleIonTraceMode)
		{
			// open the ion trace file
			ionPosTrace.open(ionPosTraceName.c_str());
		}

		// holds GPU device properties 
		cudaDeviceProp prop;
		// get GPU device properties
		cudaGetDeviceProperties(&prop, 0);

		// display GPU device properties
		debugFile << "-- Debugging: GPU Properties --" << std::endl;
		debugFile << "sharedMemPerBlock: " << prop.sharedMemPerBlock << std::endl;
		debugFile << "totalGlobalMem: " << prop.totalGlobalMem << std::endl;
		debugFile << "regsPerBlock: " << prop.regsPerBlock << std::endl;
		debugFile << "warpSize: " << prop.warpSize << std::endl;
		debugFile << "maxThreadsPerBlock: " << prop.maxThreadsPerBlock << std::endl;
		debugFile << "maxGridSize: " << prop.maxGridSize[0] << ", "
			<< prop.maxGridSize[1] << ", "
			<< prop.maxGridSize[2] << std::endl;
		debugFile << "clockRate: " << prop.clockRate << std::endl;
		debugFile << "deviceOverlap: " << prop.deviceOverlap << std::endl;
		debugFile << "multiProcessorCount: " << prop.multiProcessorCount << std::endl;
		debugFile << "kernelExecTimeoutEnabled: " << prop.kernelExecTimeoutEnabled << std::endl;
		debugFile << "integrated: " << prop.integrated << std::endl;
		debugFile << "canMapHostMemory: " << prop.canMapHostMemory << std::endl;
		debugFile << "computeMode: " << prop.computeMode << std::endl;
		debugFile << "concurrentKernels: " << prop.concurrentKernels << std::endl;
		debugFile << "ECCEnabled: " << prop.ECCEnabled << std::endl;
		debugFile << "pciBusID: " << prop.pciBusID << std::endl;
		debugFile << "pciDeviceID: " << prop.pciDeviceID << std::endl;
		debugFile << "tccDriver: " << prop.tccDriver << std::endl << std::endl;
	}

	/*************************
			Constants
	*************************/

	// electron charge (Q)
	const float CHARGE_ELC = -1.602177e-19;

	// permittivity of free pace in a vacuum (F/m)
	const float PERM_FREE_SPACE = 8.854e-12;

	// Boltzmann Constant (Kgm^2)/(K*s^2)
	const float BOLTZMANN = 1.380649e-23;

	// Pi
	const float PI = 3.141593;

	// Number of threads oer block
	// has a limit of 1024 and should be 
	// a multiple of warp size
	const unsigned int DIM_BLOCK = 256;

	// number of threads in a warp
	const int WARP_SIZE = 32;

	// DEBUGGING // 
	if (debugMode && showConstants)
	{
		debugFile << "-- Debugging: Constants --" << std::endl;
		debugFile << "CHARGE_ELC: " << CHARGE_ELC << std::endl;
		debugFile << "PERM_FREE_SPACE: " << PERM_FREE_SPACE << std::endl;
		debugFile << "BOLTZMANN: " << BOLTZMANN << std::endl;
		debugFile << "PI: " << PI << std::endl;
		debugFile << "DIM_BLOCK: " << DIM_BLOCK << std::endl;
		debugFile << "WARP_SIZE: " << WARP_SIZE << std::endl;
		debugFile << std::endl;
	}

	/*************************
		Output Parameters
	*************************/

	// !!! will want an output parameter file !!!

	// the number of grid cells along each side of
	// output statistic such as ion density and 
	// electric potential. Each grid cell can be
	// represented as a pixel. 
	const short int GRID_RES = 600;

	// DEBUGGING //
	if (debugMode && showOutputParameters)
	{
		debugFile << "-- Debugging: Output Parameters --" << std::endl;
		debugFile << "GRID_RES:" << GRID_RES << std::endl;
		debugFile << std::endl;
	} // DEBUGGING //

	/*************************
			Parameters
	*************************/

	// number of user defined parameters
	const int NUM_USSER_PARAMS = 14;

	// allocate memory for user parameters
	float* floatParams = (float*)malloc(NUM_USSER_PARAMS * sizeof(float));

	// get user defined parameters
	getUsserParams(floatParams, NUM_USSER_PARAMS,
		paramListFileName.c_str());

	// assign user defined parameters
	const unsigned int  NUM_ION = static_cast<int>(floatParams[0] / DIM_BLOCK) * DIM_BLOCK;
	const float DEN_FAR_PLASMA = floatParams[1];
	const unsigned short int TEMP_ELC = floatParams[2];
	const short int TEMP_ION = floatParams[3];
	const short int DEN_DUST = floatParams[4];
	const float MASS_ION = floatParams[5];
	const float MACH = floatParams[6];
	const float SOFT_RAD = floatParams[7];
	const float RAD_DUST = floatParams[8];
	const float CHARGE_DUST = floatParams[9] * CHARGE_ELC;
	const float CHARGE_ION = floatParams[10] * CHARGE_ELC;
	const float TIME_STEP = floatParams[11];
	const short int NUM_TIME_STEP = floatParams[12];
	const float RAD_SIM_DEBYE = floatParams[13];

	// free memory allocated for user parameters
	free(floatParams);

	// Debye length
	const float DEBYE =
		sqrt(
		(PERM_FREE_SPACE * BOLTZMANN * TEMP_ELC) /
			(DEN_FAR_PLASMA * CHARGE_ELC * CHARGE_ELC)
		);

	// sound speed of the plasma
	const float SOUND_SPEED =
		sqrt(2 * BOLTZMANN * TEMP_ION / MASS_ION);

	// dust particle mass
	// assumes spherical particle 
	const float MASS_DUST =
		DEN_DUST * (4 / 3) * PI * RAD_DUST * RAD_DUST * RAD_DUST;

	// radius of the spherical simulation volume (m)
	const float RAD_SIM = RAD_SIM_DEBYE * DEBYE;

	// inverse debye
	const float INV_DEBYE = 1 / DEBYE;

	// soft radius squared
	const float SOFT_RAD_SQRD = SOFT_RAD * SOFT_RAD;

	// simulation radius squared
	const float RAD_SIM_SQRD = RAD_SIM * RAD_SIM;

	// half of a time step
	const float HALF_TIME_STEP = TIME_STEP / 2;

	// a constant multiplier for acceleration due to Ion Ion forces
	const float ION_ION_ACC_MULT = (CHARGE_ION * CHARGE_ION) /
		(4 * PI * PERM_FREE_SPACE * MASS_ION);

	// a constant multiplier for acceleration due to Ion Dust forces
	const float ION_DUST_ACC_MULT = (CHARGE_ION * CHARGE_DUST) /
		(4 * PI * PERM_FREE_SPACE * MASS_ION);

	// dust radius squared
	const float RAD_DUST_SQRD = RAD_DUST * RAD_DUST;

	// a constant multiplier for acceleration due to the 
	// electric field due to plasma outside of the simulation
	const float EXTERN_ELC_MULT =
		((RAD_SIM / DEBYE) + 1)*
		exp(-RAD_SIM / DEBYE)*
		(CHARGE_ION * CHARGE_ION * DEN_FAR_PLASMA * DEBYE * DEBYE) /
		(PERM_FREE_SPACE * MASS_ION);

	// DEBUGGING //
	if (debugMode && showParameters)
	{
		debugFile << "-- Debugging: Parameters --" << std::endl;
		debugFile << "NUM_ION: " << NUM_ION << std::endl;
		debugFile << "DEN_FAR_PLASMA: " << DEN_FAR_PLASMA << std::endl;
		debugFile << "TEMP_ELC: " << TEMP_ELC << std::endl;
		debugFile << "TEMP_ION: " << TEMP_ION << std::endl;
		debugFile << "DEN_DUST: " << DEN_DUST << std::endl;
		debugFile << "MASS_ION: " << MASS_ION << std::endl;
		debugFile << "MACH: " << MACH << std::endl;
		debugFile << "SOFT_RAD: " << SOFT_RAD << std::endl;
		debugFile << "RAD_DUST: " << RAD_DUST << std::endl;
		debugFile << "CHARGE_DUST: " << CHARGE_DUST << std::endl;
		debugFile << "CHARGE_ION: " << CHARGE_ION << std::endl;
		debugFile << "TIME_STEP: " << TIME_STEP << std::endl;
		debugFile << "NUM_TIME_STEP: " << NUM_TIME_STEP << std::endl;
		debugFile << "RAD_SIM_DEBYE: " << RAD_SIM_DEBYE << std::endl;
		debugFile << "DEBYE: " << DEBYE << std::endl;
		debugFile << "SOUND_SPEED: " << SOUND_SPEED << std::endl;
		debugFile << "MASS_DUST: " << MASS_DUST << std::endl;
		debugFile << "RAD_SIM: " << RAD_SIM << std::endl;
		debugFile << "NUM_ION: " << NUM_ION << std::endl;
		debugFile << "INV_DEBYE: " << INV_DEBYE << std::endl;
		debugFile << "SOFT_RAD_SQRD: " << SOFT_RAD_SQRD << std::endl;
		debugFile << "RAD_SIM_SQRD: " << RAD_SIM_SQRD << std::endl;
		debugFile << "HALF_TIME_STEP: " << HALF_TIME_STEP << std::endl;
		debugFile << "ION_ION_ACC_MULT: " << ION_ION_ACC_MULT << std::endl;
		debugFile << "ION_DUST_ACC_MULT: " << ION_ION_ACC_MULT << std::endl;
		debugFile << "RAD_DUST_SQRD: " << RAD_DUST_SQRD << std::endl;
		debugFile << "EXTERN_ELC_MULT: " << EXTERN_ELC_MULT << std::endl;
		debugFile << std::endl;
	}  // DEBUGGING //

	/*************************
	   Get Dust Positions
	*************************/

	// pointer for dust positions
	float3* posDust = NULL;

	// counts the number of dust particles
	int tempNumDust = 0;

	// amount of memory required for the dust positions 
	int memFloat3Dust = 0;

	// open the file containing dust positions 
	std::ifstream dustPosFile;
	dustPosFile.open(dustPosFileName.c_str());

	// check if the file opened 
	if (!dustPosFile)
	{
		fprintf(stderr, "ERROR: file not open (IonWake_000.cu) [1]\n");
	}
	else
	{
		// temporary holder for lines in the file
		std::string line;

		// skip the first line
		std::getline(dustPosFile, line);

		// count the remaining lines in the file
		while (std::getline(dustPosFile, line))
		{
			tempNumDust++;
		}

		// amount of memory required for the dust positions 
		memFloat3Dust = tempNumDust * sizeof(float3);

		// close and re-open the file so as to start pulling 
		// data from the top of the file
		dustPosFile.close();
		dustPosFile.open(dustPosFileName.c_str());

		// check if the file was re-opened
		if (!dustPosFile)
		{
			fprintf(stderr, "ERROR: file not open in IonWake.000 - %d\n", dustPosFileName.c_str());
		}
		else
		{
			// skip the first line of the file
			std::getline(dustPosFile, line);

			// alocate memory for the dust positions 
			posDust = (float3*)malloc(memFloat3Dust);

			// loop over the remaining lines in the file
			// saving the dust positions
			for (int i = 0; i < tempNumDust; i++)
			{
				// skip the first entry in each line
				dustPosFile >> line;
				// save the dust positions
				dustPosFile >> posDust[i].x;
				dustPosFile >> posDust[i].y;
				dustPosFile >> posDust[i].z;
			}
		}
	}

	// if there are no dust particles read in from
	// dustPos.txt then place one dust particle 
	// at {0, 0, 0}
	if (tempNumDust == 0)
	{
		// set number of dust particles to 1
		tempNumDust = 1;

		// amount of memory required for float3 dust data
		memFloat3Dust = tempNumDust * sizeof(float3);

		// alocate memory for dust position
		posDust = (float3*)malloc(memFloat3Dust);

		// set dust position to (0,0,0)
		posDust[0].x = 0;
		posDust[0].y = 0;
		posDust[0].z = 0;
	}

	// save the number of dust particles 
	const int NUM_DUST = tempNumDust;

	// input dust positions are in terms of the Debye length 
	// convert to meters
	for (int i = 0; i < NUM_DUST; i++)
	{
		posDust[i].x *= DEBYE;
		posDust[i].y *= DEBYE;
		posDust[i].z *= DEBYE;
	}

	// check if any of the dust particles are 
	// outside of the simulation buble
	for (int i = 0; i < NUM_DUST; i++)
	{
		if (
			(posDust[i].x*posDust[i].x
				+ posDust[i].y*posDust[i].y
				+ posDust[i].z*posDust[i].z) > RAD_SIM_SQRD
			)
		{
			fprintf(stderr, "ERROR: Dust out of simulation\n");
		}
	}

	// DEBUGGING //
	if (debugMode)
	{
		debugFile << "-- Dust Positions --" << std::endl;
		debugFile << "NUM_DUST: " << NUM_DUST << std::endl;
		for (int i = 0; i < NUM_DUST; i++)
		{
			debugFile << "X: " << posDust[i].x <<
				" Y: " << posDust[i].y <<
				" Z: " << posDust[i].z << std::endl;
		}
		debugFile << std::endl;
	} // DEBUGGING //

	/*************************
		 Get Time Step
		  Parameters
	*************************/

	// open the file containing instructions for the 
	// structure of the time step
	std::ifstream timestepFile(timestepFileName.c_str());

	// the number of commands in the file
	int numCommands = 0;

	// array for holding the time step commands
	int* commands;

	// check if the file opened 
	if (!timestepFile) {

		// output an error message
		fprintf(stderr, "ERROR on line number %d in file %s\n", __LINE__, __FILE__);
		fprintf(stderr, "File not open\n\n");

		// terminate the program 
		fatalError();

	}
	else {

		// a holder for lines from the file
		std::string line;

		// loop over all of the commands in the file
		// to find the number of commands
		while (getline(timestepFile, line)) {
			numCommands++;
		}

		// close the file
		timestepFile.close();

		// allocate memory for the commands
		commands = (int*)malloc(numCommands * sizeof(int));

		// re-open the file
		timestepFile.open(timestepFileName.c_str());

		// loop over all of the commands and save 
		// them to the commands array
		for (int i = 0; i < numCommands; i++) {

			// get the next command
			timestepFile >> line;

			// convert the command to an int
			if (line == "leapfrog") {
				commands[i] = 1;
			} else if (line == "ion-ion-acc") {
				commands[i] = 2;
			} else if (line == "ion-dust-acc") {
				commands[i] = 3;
			} else if (line == "sphere-ion-bounds") {
				commands[i] = 4;
			} else if (line == "extrn-elc-acc") {
				commands[i] = 5;
			} else if (line == "copy-ion-pos") {
				commands[i] = 6;
			} else if (line == "save-pos-trace") {
				commands[i] = 7;
			} else {
				// if the command does not exist give an error message
				fprintf(stderr, "ERROR on line number %d in file %s\n", __LINE__, __FILE__);
				fprintf(stderr, "Command \"%s\" does not exist\n\n", line.c_str());

				// terminate the program 
				fatalError();
			}
		}
	}

	// DEBUGGING //
	if (debugMode) {

		debugFile << "-- Time Step Commands --" << std::endl;

		debugFile << "Commands: " << std::endl;
		debugFile << "1: leapfrog" << std::endl;
		debugFile << "2: ion-ion-acc" << std::endl;
		debugFile << "3: ion-dust-acc" << std::endl;
		debugFile << "4: sphere-ion-bounds" << std::endl;

		debugFile << "--------------------" << std::endl;

		debugFile << "Number of commands: " << numCommands << std::endl;

		for (int i = 0; i < numCommands; i++) {
			debugFile << commands[i] << std::endl;
		}

		debugFile << "--------------------" << std::endl;

	} // DEBUGGING //

	/*************************
		Initialize Initial
		  Host Variables 
	*************************/

	// number of blocks per grid for Ions
	int blocksPerGridIon = (NUM_ION + 1) / DIM_BLOCK;

	// memory size for float type ion data arrays 
	unsigned long long int memFloatIon = NUM_ION * sizeof(float);

	// memory size for float3 type ion data arrays  
	unsigned long long int memFloat3Ion = NUM_ION * sizeof(float3);

	// allocate memory 
	float3* posIon = (float3*)malloc(memFloat3Ion);
	float3* velIon = (float3*)malloc(memFloat3Ion);
	float3* accIon = (float3*)malloc(memFloat3Ion);

	// initialize ion data arrays
	for (unsigned long long int i = 0; i < NUM_ION; i++)
	{
		// set all ions out of bounds
		// then when the bounds are checked in
		// the time step all of the ions are 
		// given a new position
		posIon[i].x = RAD_SIM + 1;
		posIon[i].y = RAD_SIM + 1;
		posIon[i].z = RAD_SIM + 1;

		// set an initial velocity
		velIon[i].x = 0;
		velIon[i].y = 0;
		velIon[i].z = 0;

		// set an initial acceleration
		accIon[i].x = 0;
		accIon[i].y = 0;
		accIon[i].z = 0;
	}	

	// DEBUGGING //
	if (debugMode && showInitHostVars)
	{
		debugFile << "-- Initial Host Variables --" << std::endl;
		debugFile << "memFloatIon: " << memFloatIon << std::endl;
		debugFile << "memFloat3Ion: " << memFloat3Ion << std::endl;
		debugFile << "shared memory: " << sizeof(float3) << std::endl;
		debugFile << std::endl;
		debugFile << "First 20 ion poisitions: " << std::endl;
		for (int i = 0; i < 20; i++)
		{
			debugFile << "X: " << posIon[i].x <<
				        " Y: " << posIon[i].y <<
				        " Z: " << posIon[i].z << std::endl;
		}
		debugFile << std::endl << "Last 20 ion poisitions" << std::endl;
		for (int i = 1; i <= 20; i++)
		{
			int ID = NUM_ION - i;

			debugFile << "X: "  << posIon[ID].x
					  << " Y: " << posIon[ID].y
				      << " Z: " << posIon[ID].z
				      << std::endl;
		}
		debugFile << std::endl;
	} // DEBUGGING //
	
	/*************************
		Initialize Initial
		 Device Variables
	*************************/

	// variable to hold cuda status 
	cudaError_t cudaStatus;

	// create device pointers 
	float3* d_posIon;
	float3* d_velIon;
	float3* d_accIon;
	float3* d_posDust;
	curandState_t* d_statesBlock = NULL;
	curandState_t* d_statesThread = NULL;
	float* d_INV_DEBYE;
	float* d_RAD_DUST_SQRD;
	float* d_SOFT_RAD_SQRD;
	float* d_RAD_SIM;
	float* d_RAD_SIM_SQRD;
	float* d_HALF_TIME_STEP;
	float* d_ION_ION_ACC_MULT;
	float* d_ION_DUST_ACC_MULT;
	float* d_EXTERN_ELC_MULT;
	unsigned int* d_NUM_ION;
	unsigned int* d_NUM_DUST;


	// allocate GPU memory for the external electric field 
	// multiplier for calculating the acceleration
	cudaStatus = cudaMalloc(&d_EXTERN_ELC_MULT, sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed: d_EXTERN_ELC_MULT\n");
	}
	
	// allocate GPU memory for the dust particle radii squared
	cudaStatus = cudaMalloc(&d_RAD_DUST_SQRD, sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed: d_RAD_DUST_SQRD\n");
	}

	// allocate GPU memory for the number of dust particles
	cudaStatus = cudaMalloc(&d_NUM_DUST, sizeof(unsigned int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed: d_NUM_DUST\n");
	}

	// allocate GPU memory for the number of ions
	cudaStatus = cudaMalloc(&d_NUM_ION, sizeof(unsigned int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed: d_NUM_ION\n");
	}

	// allocate GPU memory for the inverse debye
	cudaStatus = cudaMalloc(&d_INV_DEBYE, sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed: d_INV_DEBYE\n");
	}

	// allocate GPU memory for the softening radius 
	cudaStatus = cudaMalloc(&d_SOFT_RAD_SQRD, sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed: d_SOFT_RAD_SQRD\n");
	}

	// allocate GPU memory for the simulation radius
	cudaStatus = cudaMalloc(&d_RAD_SIM, sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed: d_RAD_SIM\n");
	}

	// allocate GPU memory for the sumulation radius squared
	cudaStatus = cudaMalloc(&d_RAD_SIM_SQRD, sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed: d_RAD_SIM_SQRD\n");
	}

	// allocate GPU memory for the half time step
	cudaStatus = cudaMalloc(&d_HALF_TIME_STEP, sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed: d_HALF_TIME_STEP\n");
	}

	// allocate GPU memory for the ion ion acceleration multiplier 
	cudaStatus = cudaMalloc(&d_ION_ION_ACC_MULT, sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed: d_ION_ION_ACC_MULT\n");
	}

	// allocate GPU memory for the ion ion acceleration multiplier 
	cudaStatus = cudaMalloc(&d_ION_DUST_ACC_MULT, sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed: d_ION_DUST_ACC_MULT\n");
	}

	// allocate GPU memory for the dust positions
	cudaStatus = cudaMalloc(&d_posDust, memFloat3Dust);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed: d_posDust\n");
	}

	// allocate GPU memory for the ion positions
	cudaStatus = cudaMalloc(&d_posIon, memFloat3Ion);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed: d_posIon\n");
	}

	// allocate GPU memory for the ion velocities 
	cudaStatus = cudaMalloc(&d_velIon, memFloat3Ion);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed: d_velIon\n");
	}

	// allocate GPU memory for the ion accelerations
	cudaStatus = cudaMalloc(&d_accIon, memFloat3Ion);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed: d_accIon\n");
	}

	// allocate GPU memory for block-wise random states
	cudaStatus = cudaMalloc(&d_statesBlock, DIM_BLOCK * sizeof(curandState_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed: d_statesBlock\n");
	}

	// allocate GPU memory for thread-wise random states
	cudaStatus = cudaMalloc(&d_statesThread, DIM_BLOCK * sizeof(curandState_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed: d_statesThread\n");
	}

	// copy the external electric acceleration multiplier 
	// value to the GPU
	cudaStatus = cudaMemcpy(d_EXTERN_ELC_MULT, &EXTERN_ELC_MULT,
		sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed: d_EXTERN_ELC_MULT\n");
	}

	// copy dust radii squared value to the GPU
	cudaStatus = cudaMemcpy(d_RAD_DUST_SQRD, &RAD_DUST_SQRD,
		sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed: d_RAD_DUST_SQRD\n");
	}

	// copy inverse debye value to the GPU
	cudaStatus = cudaMemcpy(d_INV_DEBYE, &INV_DEBYE, 
		sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed: d_INV_DEBYE\n");
	}
	
	// copy softening radius squared  value to the GPU
	cudaStatus = cudaMemcpy(d_SOFT_RAD_SQRD, &SOFT_RAD_SQRD, 
		sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed: d_SOFT_RAD_SQRD\n");
	}
	
	// copy simulation radius value to the GPU
	cudaStatus = cudaMemcpy(d_RAD_SIM, &RAD_SIM, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed: d_RAD_SIM\n");
	}
	
	// copy simulation radius squared value to the GPU
	cudaStatus = cudaMemcpy(d_RAD_SIM_SQRD, &RAD_SIM_SQRD, 
		sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed: d_RAD_SIM_SQRD\n");
	}
	
	// copy half time step value to the GPU
	cudaStatus = cudaMemcpy(d_HALF_TIME_STEP, &HALF_TIME_STEP, 
		sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed: d_HALF_TIME_STEP\n");
	}
	
	// copy ion ion acceleration multiplier value to the GPU
	cudaStatus = cudaMemcpy(d_ION_ION_ACC_MULT, &ION_ION_ACC_MULT, 
		sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed: d_ION_ION_ACC_MULT\n");
	}

	// copy ion ion acceleration multiplier value to the GPU
	cudaStatus = cudaMemcpy(d_ION_DUST_ACC_MULT, &ION_DUST_ACC_MULT,
		sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed: d_ION_DUST_ACC_MULT\n");
	}

	// copy the number of dust particles to the GPU
	cudaStatus = cudaMemcpy(d_NUM_DUST, &NUM_DUST, sizeof(unsigned int), 
		cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed: d_NUM_DUST\n");
	}

	// copy the number of ions to the GPU
	cudaStatus = cudaMemcpy(d_NUM_ION, &NUM_ION, sizeof(unsigned int), 
		cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed: d_NUM_ION\n");
	}

	// copy dust possitions to the GPU
	cudaStatus = cudaMemcpy(d_posDust, posDust, memFloat3Dust, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed: d_posDust\n");
	}

	// copy ion possitions to the GPU
	cudaStatus = cudaMemcpy(d_posIon, posIon, memFloat3Ion, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed: d_posIon\n");
	}

	// copy ion velocities to the GPU
	cudaStatus = cudaMemcpy(d_velIon, velIon, memFloat3Ion, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed: d_velIon\n");
	}

	// copy ion accelerations to the GPU
	cudaStatus = cudaMemcpy(d_accIon, accIon, memFloat3Ion, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed: d_accIon\n");
	}

	// invoke the GPU to initialize all the random states
	init <<< DIM_BLOCK, 1 >>> (time(0), d_statesThread);

	// invoke the GPU to initialize all the random states
	init <<< blocksPerGridIon, 1 >> > (time(0), d_statesBlock);

	// Check for any errors launching the init kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "init kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns 
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, 
			"cudaDeviceSynchronize returned error code %d after launching init kernel!\n"
			, cudaStatus);
	}

	/*************************
			Time Step
	*************************/

	// create and open an output text file
	std::ofstream statusFile;
	statusFile.open(statusFileName.c_str());

	// Syncronize threads and check for errors before entering timestep
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Error before time step kernel launches: %s\n", 
			cudaGetErrorString(cudaStatus));
	}


	// Parameters that turn on and off
	// parts of the timestep for debuging.
	// Should be removed before a production 
	// release
	bool checkBounds  = true,
		ionIonForce   = true,
		ionDustForce  = true,
		extrnElcForce = false,
		integrate     = true;

	statusFile << "-- Time Step Parameters --" << std::endl;
	statusFile << "checkBounds: " << checkBounds << std::endl;
	statusFile << "ionIonForce: " << ionIonForce << std::endl;
	statusFile << "ionDustForce: " << ionDustForce << std::endl;
	statusFile << "extrnElcForce: " << extrnElcForce << std::endl;
	statusFile << "integrate: " << integrate << std::endl << std::endl;
	statusFile << "-- Time Step --" << std::endl;
	/*
		Durring timestep all calculations are 
		completed on the device so no memory 
		transfers to the host are completed 
		to reduce calculation time. Because of 
		this data manipulation on the host 
		will cause undefined errors.
		cudaDeviceSynchronize is called before 
		each kernel to prevent asynchronus 
		exicution of linear steps.
	*/
	for (int i = 1; i <= NUM_TIME_STEP; i++)
	{

		statusFile << i << ": ";

		for (int j = 0; j < numCommands; j++) {
			
			// perform a leapfrog integration
			if (commands[j] == 1)
			{
				statusFile << "1 ";

				// use the new accelerations to update the positions and velocities
				// of all the ions
				stepForward << < blocksPerGridIon, DIM_BLOCK >> >
					(d_posIon, d_velIon, d_accIon, d_HALF_TIME_STEP);

				// Check for any errors launching the kernel
				cudaStatus = cudaGetLastError();
				if (cudaStatus != cudaSuccess) {
					fprintf(stderr, "stepForward launch failed: %s\n\n", cudaGetErrorString(cudaStatus));
				}

				// Syncronize threads and check for errors
				cudaStatus = cudaDeviceSynchronize();
				if (cudaStatus != cudaSuccess) {
					fprintf(stderr, "cudaDeviceSynchronize returned error code: %d\n", cudaStatus);
					fprintf(stderr, "Location: after stepForward on timestep %d\n\n", i);
				}

			}

			// calculate the acceleration due to ion-ion interactions
			else if (commands[j] == 2)
			{
				statusFile << "2 ";

				// calculate the forces between all ions
				calcIonIonForces << < blocksPerGridIon, DIM_BLOCK, sizeof(float3)*DIM_BLOCK >> >
					(d_posIon, d_accIon, d_NUM_ION, d_SOFT_RAD_SQRD, d_ION_ION_ACC_MULT, d_INV_DEBYE);

				// Check for any errors launching the kernel
				cudaStatus = cudaGetLastError();
				if (cudaStatus != cudaSuccess) {
					fprintf(stderr, "calcIonIonForce launch failed: %s\n\n",
						cudaGetErrorString(cudaStatus));
				}

				// Syncronize threads and check for errors
				cudaStatus = cudaDeviceSynchronize();
				if (cudaStatus != cudaSuccess) {
					fprintf(stderr, "cudaDeviceSynchronize returned error code: %d\n", cudaStatus);
					fprintf(stderr, "Location: after calcIonIonForce on timestep %d\n\n", i);
				}
			}

			// calculate the acceleration due to ion-dust interactions
			else if (commands[j] == 3)
			{
				statusFile << "3 ";

				// calculate ion dust accelerations
				calcIonDustForces << < blocksPerGridIon, DIM_BLOCK >> > (d_posIon, d_accIon,
					d_NUM_ION, d_SOFT_RAD_SQRD, d_ION_DUST_ACC_MULT, d_INV_DEBYE,
					d_NUM_DUST, d_posDust);

				// Check for any errors launching the kernel
				cudaStatus = cudaGetLastError();
				if (cudaStatus != cudaSuccess) {
					fprintf(stderr, "calcIonDustForce launch failed: %s\n\n",
						cudaGetErrorString(cudaStatus));
				}

				// Syncronize threads and check for errors
				cudaStatus = cudaDeviceSynchronize();
				if (cudaStatus != cudaSuccess) {
					fprintf(stderr, "cudaDeviceSynchronize returned error code: %d\n", cudaStatus);
					fprintf(stderr, "Location: after calcIonDustForce on timestep %d\n\n", i);
				}
			}

			// check ion bounds for spherical simulation region
			else if (commands[j] == 4)
			{
				statusFile << "4 ";

				// check for ions that have left the simulation region
				// and give them a new position and velocity
				replaceOutOfBoundsIons << < blocksPerGridIon, DIM_BLOCK >> >
					(d_posIon, d_velIon, d_statesThread, d_statesBlock, d_RAD_SIM_SQRD, d_RAD_SIM,
						d_NUM_ION, d_NUM_DUST, d_posDust, d_RAD_DUST_SQRD);

				// Check for any errors launching the kernel
				cudaStatus = cudaGetLastError();
				if (cudaStatus != cudaSuccess) {
					fprintf(stderr, "replaceOutOfBoundsIons launch failed: %s\n\n",
						cudaGetErrorString(cudaStatus));
				}

				// Syncronize threads and check for errors
				cudaStatus = cudaDeviceSynchronize();
				if (cudaStatus != cudaSuccess) {
					fprintf(stderr, "cudaDeviceSynchronize returned error code: %d\n", cudaStatus);
					fprintf(stderr, "Location: after replaceOutOfBoundsIons on timestep %d\n\n", i);
				}	
			}
			
			// calculate the ion accelerations due to the electric field 
			// outside of the simulation sphere
			else if (commands[j] == 5)
			{
				statusFile << "5 ";

				// calculate the forces between all ions
				calcExtrnElcForce << < blocksPerGridIon, DIM_BLOCK >> >
					(d_accIon, d_posIon, d_EXTERN_ELC_MULT, d_INV_DEBYE);

				// Check for any errors launching the kernel
				cudaStatus = cudaGetLastError();
				if (cudaStatus != cudaSuccess) {
					fprintf(stderr, "calcExtrnForce launch failed: %s\n\n",
						cudaGetErrorString(cudaStatus));
				}
				
				// Syncronize threads and check for errors
				cudaStatus = cudaDeviceSynchronize();
				if (cudaStatus != cudaSuccess) {
					fprintf(stderr, "cudaDeviceSynchronize returned error code: %d\n", cudaStatus);
					fprintf(stderr, "Location: after calcExtrnElcForce on timestep %d\n\n", i);
				}				
			}
			
			// copy ion possitions to the host
			else if (commands[j] == 6)
			{
				statusFile << "6 ";

				// copy ion possitions to host
				cudaStatus = cudaMemcpy(posIon, d_posIon, memFloat3Ion, cudaMemcpyDeviceToHost);
				if (cudaStatus != cudaSuccess) {
					fprintf(stderr, "cudaMemcpy failed: d_posIon\n");
				}
			}

			// if the program is in debuging mode and set to trace the position
			// of a single ion then save the position of a single ion to 
			// the ionPosTrace file
			else if (debugMode && singleIonTraceMode && commands[j] == 7)
			{
				statusFile << "7 ";

				// print the position of of the specified ion to the ion trace file
				ionPosTrace << posIon[ionTraceIndex].x;
				ionPosTrace << ", " << posIon[ionTraceIndex].y;
				ionPosTrace << ", " << posIon[ionTraceIndex].z << std::endl;
			}

			// if the command number does not exist throw an error
			else
			{
				// output an error message
				fprintf(stderr, "ERROR on line number %d in file %s\n", __LINE__, __FILE__);
				fprintf(stderr, "Command number %d does not exist\n\n", commands[j]);

				// terminate the program 
				fatalError();
			}
		}

		statusFile << "|" << std::endl;

	} // end time step
	

	if (debugMode && singleIonTraceMode)
	{
		// print the index of the traced ion to the debuging file
		debugFile << "Single ion trace index: " << ionTraceIndex << std::endl << std::endl;

		// close the ion trace file
		ionPosTrace.close();
	}

	// copy ion positions to host
	cudaStatus = cudaMemcpy(posIon, d_posIon, memFloat3Ion, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed: d_posIon\n");
	}

	// copy ion velocities to host
	cudaStatus = cudaMemcpy(velIon, d_velIon, memFloat3Ion, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed: d_velIon\n");
	}

	// Syncronize threads and check for errors
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code: %d\n", cudaStatus);
		fprintf(stderr, "Location: after timestep \n\n");
	}

	// DEBUGGING //
	if (debugFile && showFinalHostVars)
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
		debugFile << std::endl << "-- Final Ion Positions: Last 20 Ions --" << std::endl;
		for (int i = 1; i <= 20; i++)
		{
			int ID = NUM_ION - i;

			debugFile << "#: "  << ID
				      << " X: " << posIon[ID].x
				      << " Y: " << posIon[ID].y
				      << " Z: " << posIon[ID].z
				      << std::endl;
		}
		debugFile << std::endl << "-- Final Ion Velocities: First 20 Ions --" << std::endl;
		for (int i = 0; i < 20; i++)
		{
			debugFile << "#: " << i
				<< " X: " << velIon[i].x
				<< " Y: " << velIon[i].y
				<< " Z: " << velIon[i].z
				<< std::endl;
		}
		debugFile << std::endl << "-- Final Ion Velocities: Last 20 Ions --" << std::endl;
		for (int i = 1; i <= 20; i++)
		{
			int ID = NUM_ION - i;

			debugFile << "#: " << ID
				<< " X: " << velIon[ID].x
				<< " Y: " << velIon[ID].y
				<< " Z: " << velIon[ID].z
				<< std::endl;
		}
		debugFile << std::endl << "-- Final Ion Accelerations: First 20 Ions --" << std::endl;
		for (int i = 0; i < 20; i++)
		{
			debugFile << "#: " << i
				<< " X: " << accIon[i].x
				<< " Y: " << accIon[i].y
				<< " Z: " << accIon[i].z
				<< std::endl;
		}
		debugFile << std::endl << "-- Final Ion Accelerations: Last 20 Ions --" << std::endl;
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
	} // DEBUGGING // 

	/*************************
		  Check Device 
		  "Constants"
	*************************/

	// temporary host pointers to copy device constants to
	float* testVal = (float*)malloc(sizeof(float));
	unsigned int* intTestVal = (unsigned int*)malloc(sizeof(unsigned int));

	// copy the number of ions to the host
	cudaStatus = cudaMemcpy(intTestVal, d_NUM_ION, 
		sizeof(unsigned int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed: d_NUM_ION\n");
	}
	// check if the host and device number of ions are the same
	if ((*intTestVal - NUM_ION) != 0)
	{
		fprintf(stderr, "Const Device Value Changed: NUM_ION\n");
		fprintf(stderr, "from %u", NUM_ION);
		fprintf(stderr, " to %u\n", *intTestVal);
	}

	// copy the number of dust particles to the host
	cudaStatus = cudaMemcpy(intTestVal, d_NUM_DUST, 
		sizeof(unsigned int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed: d_NUM_DUST\n");
	}
	// check if the host and device number of dust particles 
	// are the same
	if ((*intTestVal - NUM_DUST) != 0)
	{
		fprintf(stderr, "Const Device Value Changed: NUM_DUST\n");
		fprintf(stderr, "from %u", NUM_DUST);
		fprintf(stderr, " to %u\n", *intTestVal);
	}

	// copy the dust radii squared to the host
	cudaStatus = cudaMemcpy(testVal, d_RAD_DUST_SQRD,
		sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed: d_RAD_DUST_SQRD\n");
	}
	// check if the host and device dust radii squared
	// are the same
	if ((*testVal - RAD_DUST_SQRD) != 0)
	{
		fprintf(stderr, "Const Device Value Changed: RAD_DUST_SQRD\n");
		fprintf(stderr, "from %f", RAD_DUST_SQRD);
		fprintf(stderr, " to %f\n", *intTestVal);
	}

	// copy the dust radii squared to the host
	cudaStatus = cudaMemcpy(testVal, d_EXTERN_ELC_MULT,
		sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed: d_EXTERN_ELC_MULT\n");
	}
	// check if the host and device dust radii squared
	// are the same
	if ((*testVal - EXTERN_ELC_MULT) != 0)
	{
		fprintf(stderr, "Const Device Value Changed: EXTERN_ELC_MULT\n");
		fprintf(stderr, "from %f", EXTERN_ELC_MULT);
		fprintf(stderr, " to %f\n", *intTestVal);
	}

	// copy the ion ion acceleration multiplier to the host
	cudaStatus = cudaMemcpy(testVal, d_ION_ION_ACC_MULT, 
		sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed: d_ION_ION_ACC_MULT\n");
	}
	// check if the host and device ion ion acceleration
	// multipliers are the same
	if ((*testVal - ION_ION_ACC_MULT) != 0)
	{
		fprintf(stderr, "Const Device Value Changed: ION_ION_ACC_MULT\n");
		fprintf(stderr, "from %f", ION_ION_ACC_MULT);
		fprintf(stderr, " to %f\n", *intTestVal);
	}

	// copy the ion dust acceleration multiplier to the host
	cudaStatus = cudaMemcpy(testVal, d_ION_DUST_ACC_MULT,
		sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed: d_ION_DUST_ACC_MULT\n");
	}
	// check if the host and device ion ion acceleration
	// multipliers are the same
	if ((*testVal - ION_DUST_ACC_MULT) != 0)
	{
		fprintf(stderr, "Const Device Value Changed: ION_DUST_ACC_MULT\n");
		fprintf(stderr, "from %f", ION_DUST_ACC_MULT);
		fprintf(stderr, " to %f\n", *intTestVal);
	}

	// copy the half time step to the host
	cudaStatus = cudaMemcpy(testVal, d_HALF_TIME_STEP, 
		sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed: d_HALF_TIME_STEP\n");
	}
	//check if the host and device half time steps 
	// are the same
	if ((*testVal - HALF_TIME_STEP) != 0)
	{
		fprintf(stderr, "Const Device Value Changed: HALF_TIME_STEP");
		fprintf(stderr, "from %f", HALF_TIME_STEP);
		fprintf(stderr, " to %f\n", *intTestVal);
	}

	// copy the squared simulation radius to the host
	cudaStatus = cudaMemcpy(testVal, d_RAD_SIM_SQRD, 
		sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed: d_RAD_SIM_SQRD\n");
	}
	// check if the host and device squared sumulation radius
	// are the same
	if ((*testVal - RAD_SIM_SQRD) != 0)
	{
		fprintf(stderr, "Const Device Value Changed: RAD_SIM_SQRD\n");
		fprintf(stderr, "from %f", RAD_SIM_SQRD);
		fprintf(stderr, " to %f\n", *intTestVal);
	}
	
	// copy the simulation radius to the device
	cudaStatus = cudaMemcpy(testVal, d_RAD_SIM, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed: d_RAD_SIM\n");
	}
	//check if the device and host simulation radius
	// are the same
	if ((*testVal - RAD_SIM) != 0)
	{
		fprintf(stderr, "Const Device Value Changed: RAD_SIM");
		fprintf(stderr, "from %f", RAD_SIM);
		fprintf(stderr, " to %f\n", *intTestVal);
	}

	// copy the inverse debye to the host
	cudaStatus = cudaMemcpy(testVal, d_INV_DEBYE, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed: d_INV_DEBYE\n");
	}
	// check if the host and device inverse debye 
	// are the same
	if ((*testVal - INV_DEBYE) != 0)
	{
		fprintf(stderr, "Const Device Value Changed: d_INV_DEBYE");
		fprintf(stderr, "from %f", INV_DEBYE);
		fprintf(stderr, " to %f\n", *intTestVal);
	}
	
	/*************************
	   Reduce and Save Data
	*************************/

	// alocate memory to create matricies to pass 
	// into display functions
	float* xPosIon = (float*)malloc(memFloat3Ion);
	float* yPosIon = (float*)malloc(memFloat3Ion);

	// Declaire matricies for the x and y positions 
	for (int i = 0; i < NUM_ION; i++)
	{
		*(xPosIon + i) = posIon[i].x;
		*(yPosIon + i) = -posIon[i].y;
	}

	// calculate the number density
	int* numDenIon = getNumDen(GRID_RES, NUM_ION, xPosIon, yPosIon);

	// create a number density graphic
	 createBmp(GRID_RES, GRID_RES, numDenIon, numDenPlotName.c_str());
	 
	// close the debugging file
	debugFile.close();
	statusFile.close();

	return 0;
} 