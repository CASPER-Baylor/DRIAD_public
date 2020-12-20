// header file
#include "IonWake_000.h"
#include <iostream>


int main(int argc, char* argv[])
{

	/************************
		Pre-Processes 
	************************/

	// {{{
	
	/****** Open Files ******/
	// {{{

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
		EXIT_WITH_FATAL_ERROR;
	}

	// open input file for dust parameters
	fileName = inputDirName + "dust-params.txt";
	std::ifstream dustParamFile(fileName.c_str());
	if (!dustParamFile){
		fprintf(stderr, "ERROR on line number %d in file %s\n",
			__LINE__, __FILE__);
		fprintf(stderr, "ERROR: dustParamFile not open\n");
		EXIT_WITH_FATAL_ERROR;
	}

	// open input file for time step parameters
	fileName = inputDirName + "timestep.txt";
	std::ifstream timestepFile(fileName.c_str());
	if (!timestepFile){
		fprintf(stderr, "ERROR on line number %d in file %s\n",
			__LINE__, __FILE__);
		fprintf(stderr, "ERROR: timestepFile not open\n");
		EXIT_WITH_FATAL_ERROR;
	}

	// open input file for evolving plasma parameters
	fileName = inputDirName + "plasma_params.txt";
	std::ifstream plasmaParamFile(fileName.c_str());
	if (!paramFile){
		fprintf(stderr, "ERROR on line number %d in file %s\n",
			__LINE__, __FILE__);
		fprintf(stderr, "ERROR: plasmaParamFile not open\n");
		EXIT_WITH_FATAL_ERROR;
	}

	// open an output file for dust acc due to ion forces 
	fileName = dataDirName + runName + "_ion_on_dust_acc.txt";
	std::ofstream ionOnDustAccFile(fileName.c_str());

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

	// open an output file for holding ion velocities 
	//fileName = dataDirName + runName + "_ion-vel.txt";
	//std::ofstream ionVelFile(fileName.c_str());

	// open an output file for holding dust positions
	fileName = dataDirName + runName + "_dust-pos.txt";
	std::ofstream dustPosFile(fileName.c_str());

	// open an output file for holding dust charges
	fileName = dataDirName + runName + "_dust-charge.txt";
	std::ofstream dustChargeFile(fileName.c_str());

	// open an output file for tracing dust positions during the simulation
	fileName = dataDirName + runName + "_dust-pos-trace.txt";
	std::ofstream dustTraceFile(fileName.c_str());

	// open an output file for outputting the input parameters
	fileName = dataDirName + runName + "_params.txt";
	std::ofstream paramOutFile(fileName.c_str());
	
	// open an output file for outputting the grid data
	fileName = dataDirName + runName + "_ion-den.txt";
	std::ofstream ionDensOutFile(fileName.c_str());

	// open an output file for outputting the outside ion boundary data 
	fileName = dataDirName + runName + "_outside_potential.txt";
	std::ofstream ionPotOutsideFile(fileName.c_str());


	/****** Debugging Parameters ******/

	// turns on or off debugging output
	bool debugMode = true;

	// sets which ion to trace
	int ionTraceIndex = 60;

	// set the debugFule file to display 5 digits the right of the decimal
	debugFile.precision(5);
	debugFile << std::showpoint;

	/****** Print Device Properties ******/
	
	if (debugMode) {
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

	// }}}

	/****** Constants ******/

	// electron charge (Q)
	const float CHARGE_ELC = -1.602177e-19;

	// permittivity of free pace in a vacuum (F/m)
	const float PERM_FREE_SPACE = 8.854188e-12;

	// Boltzmann Constant (Kgm^2)/(K*s^2)
	const float BOLTZMANN = 1.380649e-23;

	// Pi
	const float PI = 3.141593;

	// Number of threads per block
	// has a limit of 1024 and should be a multiple of warp size
	const unsigned int DIM_BLOCK = 1024;
	const unsigned int DIM_BLOCK2 = 512;

	// number of threads in a warp
	const int WARP_SIZE = 32;

	// mass of an electron
	const float ELC_MASS = 9.10938356e-31;

	// Coulomb's constant
	const float COULOMB_CONST = 8.987552e9;

	// maximum depth for adaptive time step
	const int MAX_DEPTH = 8;

	if (debugMode) {
		debugFile << "-- Constants --" << '\n'
		<< "CHARGE_ELC      " << CHARGE_ELC      << '\n'
		<< "PERM_FREE_SPACE " << PERM_FREE_SPACE << '\n'
		<< "BOLTZMANN       " << BOLTZMANN       << '\n'
		<< "PI:             " << PI              << '\n'
		<< "DIM_BLOCK:      " << DIM_BLOCK       << '\n'
		<< "ELC_MASS:       " << ELC_MASS        << '\n'
		<< "COULOMB_CONST:  " << COULOMB_CONST   << '\n'
		<< "MAX_DEPTH:      " << MAX_DEPTH       << '\n'
		<< "WARP_SIZE:      " << WARP_SIZE       << '\n' << '\n';
		debugFile.flush();
	}

	/****** Parameters ******/

	// assign user defined parameters
	// GEOMETRY: 0 = Sphere, 1 = Cylinder
	const int   NUM_ION 
		= static_cast<int>(
			 getParam_106<int>( paramFile, "NUM_ION" ) / (2*DIM_BLOCK)
		  )	* (2 * DIM_BLOCK);
	float DEN_FAR_PLASMA 
		= getParam_106<float>( paramFile, "DEN_FAR_PLASMA" );
	float TEMP_ELC = getParam_106<float>( paramFile, "TEMP_ELC" );
	float TEMP_ION = getParam_106<float>( paramFile, "TEMP_ION" );
	const short DEN_DUST = getParam_106<short>( paramFile, "DEN_DUST" );
	const float MASS_SINGLE_ION 
		= getParam_106<float>( paramFile, "MASS_SINGLE_ION" );
	float MACH = getParam_106<float>( paramFile, "MACH" ); 
	const float SOFT_RAD = getParam_106<float>( paramFile, "SOFT_RAD" );
	const float RAD_DUST = getParam_106<float>( paramFile, "RAD_DUST" );
	const float M_FACTOR = getParam_106<float>( paramFile, "M_FACTOR" );
	const float CHARGE_SINGLE_ION 
		= CHARGE_ELC * getParam_106<float>( paramFile, "CHARGE_SINGLE_ION" );
	const float ION_TIME_STEP 
		= getParam_106<float>( paramFile, "ION_TIME_STEP" );
	const int NUM_TIME_STEP 
		= getParam_106<int>( paramFile, "NUM_TIME_STEP" );
	const int  GEOMETRY = getParam_106<int>( paramFile, "GEOMETRY" );
	const float RAD_SPH_DEBYE 
		= getParam_106<float>( paramFile, "RAD_SPH_DEBYE" );
	const int   NUM_DIV_VEL = getParam_106<int>( paramFile, "NUM_DIV_VEL" );
	const int   NUM_DIV_QTH = getParam_106<int>( paramFile, "NUM_DIV_QTH" );
  	const float RAD_CYL_DEBYE 
		= getParam_106<float>( paramFile, "RAD_CYL_DEBYE" );
	const float HT_CYL_DEBYE =getParam_106<float>( paramFile, "HT_CYL_DEBYE" );
	const float PRESSURE = getParam_106<float>( paramFile, "PRESSURE" );
	const float FREQ = getParam_106<double>( paramFile, "FREQ" );
	float E_FIELD = getParam_106<float>( paramFile, "E_FIELD" );
	const float OMEGA1 = getParam_106<float>( paramFile, "OMEGA1" );
	const float OMEGA2 = getParam_106<float>( paramFile, "OMEGA2" );
	const float RADIAL_CONF = getParam_106<float>( paramFile, "RADIAL_CONF" );
	const float AXIAL_CONF = getParam_106<float>( paramFile, "AXIAL_CONF" );
	const int	N_IONDT_PER_DUSTDT 
		= getParam_106<int>( paramFile, "N_IONDT_PER_DUSTDT" );
	const int	N_IONDT_PER_PLASMADT
		= getParam_106<int>( paramFile, "N_IONDT_PER_PLASMADT" );
	const int	N_PRINT_DEN_POT
		= getParam_106<int>( paramFile, "N_PRINT_DEN_POT" );
	const float GRID_FACTOR = getParam_106<float>( paramFile, "GRID_FACTOR" );
	const float GAS_TYPE = getParam_106<float>( paramFile, "GAS_TYPE" );
    const float TEMP_GAS = getParam_106<float>( paramFile, "TEMP_GAS" );
	const int	USE_GRAV = getParam_106<int>( paramFile, "USE_GRAV" );
	const float BOX_CENTER = getParam_106<float>( paramFile, "BOX_CENTER" );
	const float E_MULT = getParam_106<float>( paramFile, "E_MULT" );
	const int	LASER_PUSH = getParam_106<int>( paramFile, "LASER_PUSH" );
	const float LASER_ON = getParam_106<float>( paramFile, "LASER_ON" );
	const float LASER_OFF = getParam_106<float>( paramFile, "LASER_OFF" );
	const int	TIME_EVOL = getParam_106<int>( paramFile, "TIME_EVOL" );	
	
	// Set size of the simulation region

	// debye length (m)
	float DEBYE =
		sqrt((PERM_FREE_SPACE * BOLTZMANN * TEMP_ELC)/
		(DEN_FAR_PLASMA * CHARGE_ELC * CHARGE_ELC));

	// radius of the spherical simulation volume (m)
	const float RAD_SPH = RAD_SPH_DEBYE * DEBYE;

	// simulation radius squared (m^2)
	const float RAD_SPH_SQRD = RAD_SPH * RAD_SPH;

	// radius of the simulation cylinder (m)
	const float RAD_CYL = RAD_CYL_DEBYE * DEBYE;

	// cylinder radius squared (m^2)
	const float RAD_CYL_SQRD = RAD_CYL * RAD_CYL;

	// (half) height of the simulation cylinder (m)
	const float HT_CYL = HT_CYL_DEBYE * DEBYE;

	// variable for the volume
	float temp_volume = 0;
	if(GEOMETRY == 0) {
		// volume of the simulation sphere (m^3)
		temp_volume = (4.0 / 3.0) * PI * RAD_SPH * RAD_SPH * RAD_SPH;
	} else {
		// volume of the simulation cylinder (overwrites vol abv)
		temp_volume = PI * RAD_CYL_SQRD * 2.0* HT_CYL;
	}
	const float SIM_VOLUME = temp_volume;


	/*****  Read in evolving plasma parameters **********/
	/* These will overwrite paramters set in params.txt */

	// allocate memory for the evolving time parameters 
	int num_pts; 
	if(TIME_EVOL > 0) {num_pts = TIME_EVOL;}
	else {num_pts = 1;}

	float* evolEz = (float*)malloc(num_pts * sizeof(float));
	float* evolEr = (float*)malloc(num_pts * sizeof(float));
	float* evolTe = (float*)malloc(num_pts * sizeof(float));
	float* evolTi = (float*)malloc(num_pts * sizeof(float));
	float* evolne = (float*)malloc(num_pts * sizeof(float));
	float* evolni = (float*)malloc(num_pts * sizeof(float));
	float* evolVz = (float*)malloc(num_pts * sizeof(float));
	float* evolMach = (float*)malloc(num_pts * sizeof(float));

	// index for advancing to next evolving time
	int plasma_counter = 0;
	// fraction part of plasma evolving timestep
	float counter_part = 0;

	// temporary holder for lines in the file
	std::string line;

	// input file for evolving plasma conditions 
	if( TIME_EVOL > 0 ) {

		fileName = inputDirName + "plasma_params.txt";
		std::ifstream plasmaEvolFile(fileName.c_str());
		// check if the file opened
		bool plasma_cond_file = plasmaEvolFile.is_open();

		if( plasma_cond_file ) { // read in evolving plasma params from file 
			// seek to the beginning  of the file
			plasmaEvolFile.seekg(0, std::ios::beg);

			// skip the first line of the file
			std::getline(plasmaEvolFile, line);

			for( int i=0; i<TIME_EVOL; i++ ) {
				plasmaEvolFile >> evolEz[i];
				plasmaEvolFile >> evolEr[i];
				plasmaEvolFile >> evolTe[i];
				plasmaEvolFile >> evolTi[i];
				plasmaEvolFile >> evolne[i];
				plasmaEvolFile >> evolni[i];
				plasmaEvolFile >> evolVz[i];
				plasmaEvolFile >> evolMach[i];
			}
			plasmaEvolFile.close();
		}
		if (debugMode) {
			debugFile << "-- First and last plasma params from file --" << '\n'
			<< evolEz[0] << ", " << evolEr[0] << ", " 
			<< evolTe[0] << ", " << evolTi[0] << ", " 
			<< evolne[0] << ", " << evolni[0] << ", " 
			<< evolVz[0] << ", " << evolMach[0] << '\n' 
			<< evolEz[TIME_EVOL -1] << ", " << evolEr[TIME_EVOL-1] << ", " 
			<< evolTe[TIME_EVOL -1] << ", " << evolTi[TIME_EVOL-1] << ", " 
			<< evolne[TIME_EVOL -1] << ", " << evolni[TIME_EVOL-1] << ", " 
			<< evolVz[TIME_EVOL -1] << ", "<<evolMach[TIME_EVOL-1] << '\n' << std::endl;
		}
	}

	// for evolving conditions - radial E field acting on ions
	float E_FIELDR = 0;

	if(TIME_EVOL > 0) {
		MACH = evolMach[0];
		DEN_FAR_PLASMA = evolne[0];
		TEMP_ELC = evolTe[0];
		TEMP_ION = evolTi[0];
		E_FIELD = evolEz[0];
		E_FIELDR = evolEr[0];
	}	
	else { //TIME_EVOL == 0
		//copy the values set in the param file to the evolving variables
		// for use in the injection of ions
		evolEz[0] = E_FIELD;
		evolEr[0] = 0;
		evolTe[0] = TEMP_ELC;
		evolTi[0] = TEMP_ION;
		evolne[0] = DEN_FAR_PLASMA;
		evolni[0] = DEN_FAR_PLASMA;
		evolVz[0] = MACH*sqrt(BOLTZMANN * TEMP_ELC / MASS_SINGLE_ION);
		evolMach[0] = MACH;
	}

	//  ion debye length (m) used for dust shielding
	float DEBYE_I = 
		sqrt((PERM_FREE_SPACE * BOLTZMANN * TEMP_ION)/
		(DEN_FAR_PLASMA * CHARGE_ELC * CHARGE_ELC));

	// dust particle mass assumes spherical particle (Kg)
	const float MASS_DUST =
		DEN_DUST * (4.0 / 3.0) * PI * RAD_DUST * RAD_DUST * RAD_DUST;

	// inverse debye (1/m)
	float INV_DEBYE = 1.0 / DEBYE;

	// soft radius squared (m^2)
	const float SOFT_RAD_SQRD = SOFT_RAD * SOFT_RAD;

	// half of a time step (s)
	const float HALF_TIME_STEP = ION_TIME_STEP / 2.0;

	// dust radius squared (m^2)
	const float RAD_DUST_SQRD = RAD_DUST * RAD_DUST;

	// multiplier for super ions
	float SUPER_ION_MULT = SIM_VOLUME * DEN_FAR_PLASMA / NUM_ION;

	// charge on each super ion (C)
	float CHARGE_ION = CHARGE_SINGLE_ION * SUPER_ION_MULT;

	// mass of a super ion (Kg)
	float MASS_ION = MASS_SINGLE_ION * SUPER_ION_MULT;
	const float Q_DIV_M = CHARGE_ION / MASS_ION;									 

	// a constant multiplier for acceleration due to Ion Ion forces
	float ION_ION_ACC_MULT = COULOMB_CONST * Q_DIV_M;

	// a constant multiplier for acceleration of Ions due to Dust forces
	const float ION_DUST_ACC_MULT = COULOMB_CONST * Q_DIV_M;

	// a constant multiplier for acceleration of Dust due to Ion forces
	const float DUST_ION_ACC_MULT = COULOMB_CONST / MASS_DUST;
	
	// a constant muliplier for accleration due to Dust Dust forces
	const float DUST_DUST_ACC_MULT = COULOMB_CONST / MASS_DUST;

	// a constant multiplier for acceleration due to the
	// electric field due to plasma outside of the simulation
	float EXTERN_ELC_MULT =
		((RAD_SPH / DEBYE) + 1.0) * exp(-RAD_SPH / DEBYE) *
		(CHARGE_SINGLE_ION * DEN_FAR_PLASMA * DEBYE) *
		(Q_DIV_M) / (PERM_FREE_SPACE);

	// sound speed of the plasma (m/s)
	float SOUND_SPEED = sqrt(BOLTZMANN * TEMP_ELC / MASS_SINGLE_ION);
	float ION_SPEED = sqrt(BOLTZMANN * TEMP_ION / MASS_SINGLE_ION);

	// the drift velocity of the ions
	float DRIFT_VEL_ION = MACH * SOUND_SPEED;

	// a multiplier for ion collection radius
	float vs_sq = 8 * BOLTZMANN * TEMP_ION / PI / MASS_SINGLE_ION 
			+ DRIFT_VEL_ION * DRIFT_VEL_ION;
	float RAD_COLL_MULT = 
		2 *  Q_DIV_M * COULOMB_CONST / RAD_DUST / vs_sq;
	
	// the electron current to an uncharged dust grain
	float ELC_CURRENT_0 = 4.0 * PI * RAD_DUST_SQRD * DEN_FAR_PLASMA *
		CHARGE_ELC * sqrt((BOLTZMANN * TEMP_ELC)/(2.0 * PI * ELC_MASS));

	// the electron temperature in eV is the plasma potential for this
	// model, which excludes the electrons from the calculations
	//const float ELC_TEMP_EV = TEMP_ELC * 8.61733e-5;

	// Set collision cross sections for ion and neutral gas
	const int I_CS_RANGES = 1000000;
	float totIonCollFreq = 0;
	//float TEMP_GAS = 300;
	const float NUM_DEN_GAS = PRESSURE/BOLTZMANN/TEMP_GAS;
	
	debugSpecificFile << "I_CS_RANGES " << I_CS_RANGES << '\n';
	// allocate memory for the collision cross sections
	float* sigma_i1 = new float[I_CS_RANGES+1];
	float* sigma_i2 = new float[I_CS_RANGES+1];
	float* sigma_i_tot = new float[I_CS_RANGES+1];

	//determines a constant total collision frequency
	setIonCrossSection_105( GAS_TYPE, I_CS_RANGES, NUM_DEN_GAS,
		MASS_SINGLE_ION, sigma_i1, sigma_i2, sigma_i_tot,
		totIonCollFreq, debugMode, debugSpecificFile);

	//Number of ions to collide each time step.  Adjust for non-integer value.
	const float N1 = NUM_ION * (1.0 - exp(- totIonCollFreq * ION_TIME_STEP));
	const int N_COLL = (int)(N1);
	int n_coll = 0;
	bool exist;
	//allocate memory for the ion collision list
	int* collList = (int*)malloc(NUM_ION * sizeof(int));
	int* collID = (int*)malloc(NUM_ION * sizeof(int));
	int collision_counter = 0; 
	int dum = 0; //temp variable for ion collision list
	int set_value;
	int unset_value;
	
	// a constant multiplier for the radial dust acceleration due to
	// external confinement
	const float OMEGA_DIV_M = OMEGA1 / MASS_DUST;
	const float OMEGA2_DIV_M = OMEGA2 / MASS_DUST;
	float radialConfine = RADIAL_CONF * RAD_CYL; //limit position of dust in cyl
	float axialConfine = AXIAL_CONF * HT_CYL; //limit axial position of dust in cyl
	float dust_dt = 1e-4; //N * 500 * ION_TIME_STEP;
	float half_dust_dt = dust_dt * 0.5;	
	float dust_time = 0;
	double ionTime = 0;
	float rhoDustsq = 0; // for radial dust confinement
	float rhoDust = 0; // for radial dust confinement
	float acc = 0; //for radial dust confinement
	float adj_z = 0; //for dust confinement in z
	float ht = 0; //for adjusting dust height above electrode
	float ht2 = 0; 
	// for force from ions outside simulation
	//float rad = 0; 
	//float zsq = 0;
	//float radAcc = 0;
	//float vertAcc = 0;
	float q_div_m = 0;
	float3 deltavee;
	deltavee.x = 0;
	deltavee.y = 0;
	deltavee.z = 0;
	float mom_const = MASS_ION/MASS_DUST*dust_dt/(2*N_IONDT_PER_DUSTDT*ION_TIME_STEP);
	//Adjust the dust charge for non-zero plasma potential
	//int num = 1000; //Random number for Brownian kick
	// Damping factor for dust
	const float BETA =1.44* 4.0 /3.0 * RAD_DUST_SQRD * PRESSURE / MASS_DUST * 
		sqrt(8.0 * PI * MASS_SINGLE_ION/BOLTZMANN/TEMP_GAS);
	//Thermal bath or Brownian motion of dust
	const float SIGMA = sqrt(2.0* BETA * BOLTZMANN * TEMP_GAS/MASS_DUST/dust_dt);

	int N = N_PRINT_DEN_POT; //determines how often ion dens and potential are printed
	// Set up grid for collecting ion number density and potential
	const int RESX = 64;
	const int RESZ = static_cast<int>(HT_CYL_DEBYE/(RAD_CYL_DEBYE/1))*RESX;
	const float grid_factor = GRID_FACTOR; 
	float dx = 2.0*(RAD_CYL*grid_factor)/RESX;
	float dz = 2.0*HT_CYL*grid_factor/RESZ;
	const int NUM_GRID_PTS = RESX * RESZ;
	
	if (debugMode) {
		debugFile << "-- User Parameters --" << '\n'
		<< "NUM_ION           " << NUM_ION           << '\n'
		<< "DEN_FAR_PLASMA    " << DEN_FAR_PLASMA    << '\n'
		<< "TEMP_ELC          " << TEMP_ELC          << '\n'
		<< "TEMP_ION          " << TEMP_ION          << '\n'
		<< "TEMP_GAS   		  " << TEMP_GAS			 << '\n'
		<< "PRESSURE          " << PRESSURE          << '\n'
		<< "MACH              " << MACH              << '\n'
     	<< "GAS_TYPE          " << GAS_TYPE          << '\n'
		<< "CHARGE_SINGLE_ION " << CHARGE_SINGLE_ION << '\n'
		<< "MASS_SINGLE_ION   " << MASS_SINGLE_ION   << '\n'
		<< "SOFT_RAD          " << SOFT_RAD          << '\n'
		<< "M_FACTOR		  " << M_FACTOR 		 << '\n'
		<< "RAD_DUST          " << RAD_DUST          << '\n'
		<< "DEN_DUST          " << DEN_DUST          << '\n'
		<< "ION_TIME_STEP     " << ION_TIME_STEP     << '\n'
		<< "NUM_TIME_STEP     " << NUM_TIME_STEP     << '\n'
		<< "N_IONDT_PER_DUSTDT " << N_IONDT_PER_DUSTDT << '\n'
		<< "N_IONDT_PER_PLASMADT " << N_IONDT_PER_PLASMADT << '\n'
		<< "N_PRINT_DEN_POT   " << N_PRINT_DEN_POT 	 << '\n'
		<< "GEOMETRY          " << GEOMETRY          << '\n'
		<< "RAD_SPH_DEBYE     " << RAD_SPH_DEBYE     << '\n'
		<< "NUM_DIV_VEL       " << NUM_DIV_VEL       << '\n'
		<< "NUM_DIV_QTH       " << NUM_DIV_QTH       << '\n'
		<< "RAD_CYL_DEBYE     " << RAD_CYL_DEBYE     << '\n'
		<< "HT_CYL_DEBYE      " << HT_CYL_DEBYE      << '\n'
		<< "E_FIELD           " << E_FIELD	         << '\n'
		<< "FREQ              " << FREQ	             << '\n'
		<< "OMEGA1			  " << OMEGA1			 << '\n'
		<< "OMEGA2			  " << OMEGA2			 << '\n'
		<< "RADIAL_CONF		  "	<< RADIAL_CONF		 << '\n'
		<< "AXIAL_CONF		  "	<< AXIAL_CONF		 << '\n'
		<< "N_PRINT_DEN_POT	  " << N_PRINT_DEN_POT	 << '\n'
		<< "GRID_FACTOR	      " << GRID_FACTOR		 << '\n'
		<< "BOX_CENTER		  " << BOX_CENTER		 << '\n'
		<< "USE_GRAV	  	  " << USE_GRAV   		 << '\n'
		<< "E_MULT			  " << E_MULT			 << '\n'
		<< "LASER_PUSH		  " << LASER_PUSH		 << '\n'
		<< "LASER_ON		  " << LASER_ON			 << '\n'
		<< "LASER_OFF		  " << LASER_OFF		 << '\n'
		<< "TIME_EVOL	      " << TIME_EVOL	 << '\n'
		<< '\n';

		debugFile << "-- Derived Parameters --"  << '\n'
		<< "DEBYE         " << DEBYE         << '\n'
		<< "DEBYE_I       " << DEBYE_I       << '\n'
		<< "RAD_SPH       " << RAD_SPH       << '\n'
		<< "RAD_CYL       " << RAD_CYL       << '\n'
		<< "HT_CYL        " << HT_CYL      	 << '\n'
		<< "SIM_VOLUME    " << SIM_VOLUME    << '\n'
		<< "SOUND_SPEED   " << SOUND_SPEED   << '\n'
		<< "DRIFT_VEL_ION " << DRIFT_VEL_ION << '\n'
		<< "ELC_CURRENT_0 " << ELC_CURRENT_0 << '\n'
		<< "MASS_DUST     " << MASS_DUST     << '\n'
		<< "OMEGA_DIV_M	  " << OMEGA_DIV_M	 << '\n'
		<< "OMEGA2_DIV_M  " << OMEGA2_DIV_M	 << '\n'
		<< "E_FIELDR	  " << E_FIELDR	 	 << '\n' << '\n';

		debugFile << "-- Super Ion Parameters --"  << '\n'
		<< "SUPER_ION_MULT " << SUPER_ION_MULT << '\n'
		<< "CHARGE_ION     " << CHARGE_ION     << '\n'
		<< "MASS_ION       " << MASS_ION       << '\n'
		<< "totIonCollFreq " << totIonCollFreq << '\n'
		<< "NUM_DEN_GAS	   " << NUM_DEN_GAS	   << '\n' << '\n';

		debugFile << "-- Further Derived Parameters --"  << '\n'
		<< "INV_DEBYE         " << INV_DEBYE         << '\n'
		<< "SOFT_RAD_SQRD     " << SOFT_RAD_SQRD     << '\n'
		<< "RAD_SPH_SQRD      " << RAD_SPH_SQRD      << '\n'
		<< "RAD_CYL_SQRD      " << RAD_CYL_SQRD      << '\n'
		<< "HALF_TIME_STEP    " << HALF_TIME_STEP    << '\n'
		<< "ION_ION_ACC_MULT  " << ION_ION_ACC_MULT  << '\n'
		<< "ION_DUST_ACC_MULT " << ION_DUST_ACC_MULT << '\n'
		<< "DUST_ION_ACC_MULT " << DUST_ION_ACC_MULT << '\n'
		<< "DUST_DUST_ACC_MULT " << DUST_DUST_ACC_MULT << '\n'
		<< "RAD_COLL_MULT     " << RAD_COLL_MULT	 << '\n'
		<< "RAD_DUST_SQRD     " << RAD_DUST_SQRD     << '\n'
		<< "EXTERN_ELC_MULT   " << EXTERN_ELC_MULT   << '\n'
		<< "Q_DIV_M   	      " << Q_DIV_M	         << '\n' 
		<< "BETA			  " << BETA				 << '\n' 
		<< "SIGMA			  " << SIGMA			 << '\n'
		<< "RESX			  " << RESX				 << '\n'
		<< "RESZ			  " << RESZ				 << '\n'
		<< "NUM_GRID_PTS	  " << NUM_GRID_PTS		 << '\n'		
		<< "dx			      " << dx				 << '\n'
		<< '\n';

		debugFile << "-- Sigma Values --" << '\n';
		debugFile << "First 10 sigma_i_tot" << '\n';
		for(int i=0 ; i<10 ; i++) debugFile << sigma_i_tot[i] << " ";
		debugFile << "\nFirst 10 sigma_i1" << '\n';
		for(int i=0 ; i<10 ; i++) debugFile << sigma_i1[i] << " ";
		debugFile << "\nFirst 10 sigma_i2" << '\n';
		for(int i=0 ; i<10 ; i++) debugFile << sigma_i2[i] << " ";
		debugFile << '\n';
		
		debugFile.flush();
	}

	/****** Print Parameters ******/

	// set the output file to display 7 digits the right of the decimal
	paramOutFile.precision(7);
	paramOutFile << std::showpoint << std::left;

	// output all of the parameters such that matlab can read them in
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
	<< std::setw(14) << M_FACTOR          << " % M_FACTOR"          << '\n'
	<< std::setw(14) << CHARGE_SINGLE_ION << " % CHARGE_SINGLE_ION" << '\n'
	<< std::setw(14) << ION_TIME_STEP     << " % ION_TIME_STEP"     << '\n'
	<< std::setw(14) << NUM_TIME_STEP     << " % NUM_TIME_STEP"     << '\n'
	<< std::setw(14) << RAD_SPH_DEBYE     << " % RAD_SPH_DEBYE"     << '\n'
	<< std::setw(14) << NUM_DIV_VEL       << " % NUM_DIV_VEL"       << '\n'
	<< std::setw(14) << GEOMETRY          << " % GEOMETRY"          << '\n'
	<< std::setw(14) << RAD_CYL_DEBYE	  << " % RAD_CYL_DEBYE"	    << '\n'
	<< std::setw(14) << HT_CYL_DEBYE   	  << " % HT_CYL_DEBYE"     	<< '\n'
	<< std::setw(14) << NUM_DIV_QTH       << " % NUM_DIV_QTH"       << '\n'
	<< std::setw(14) << DEBYE             << " % DEBYE"             << '\n'
	<< std::setw(14) << DEBYE_I           << " % DEBYE_I"           << '\n'
	<< std::setw(14) << RAD_SPH           << " % RAD_SPH"           << '\n'
	<< std::setw(14) << RAD_CYL           << " % RAD_CYL"           << '\n'
	<< std::setw(14) << HT_CYL            << " % HT_CYL"           << '\n'
	<< std::setw(14) << PRESSURE          << " % PRESSURE"          << '\n'
	<< std::setw(14) << FREQ              << " % FREQ  "            << '\n'
	<< std::setw(14) << E_FIELD           << " % E_FIELD"           << '\n'
	<< std::setw(14) << OMEGA1			  << " % OMEGA1"            << '\n'
	<< std::setw(14) << OMEGA2			  << " % OMEGA2"            << '\n'
	<< std::setw(14) << RADIAL_CONF		  << " % RADIAL_CONF" 		<< '\n'
	<< std::setw(14) << AXIAL_CONF		  << " % AXIAL_CONF" 		<< '\n'
	<< std::setw(14) << N_IONDT_PER_DUSTDT << " % N_IONDT_PER_DUSTDT"  << '\n'
	<< std::setw(14) << N_IONDT_PER_PLASMADT << " % N_IONDT_PER_PLASMADT"  << '\n'
	<< std::setw(14) << N_PRINT_DEN_POT	  << " % N_PRINT_DEN_POT"   << '\n'
 	<< std::setw(14) << GAS_TYPE          << " % GAS_TYPE"          << '\n'
    << std::setw(14) << BOX_CENTER        << " % BOX_CENTER"        << '\n'
    << std::setw(14) << TEMP_GAS		  << " % TEMP_GAS"          << '\n'
	<< std::setw(14) << SIM_VOLUME        << " % SIM_VOLUME"        << '\n'
	<< std::setw(14) << SOUND_SPEED       << " % SOUND_SPEED"       << '\n'
	<< std::setw(14) << DRIFT_VEL_ION     << " % DRIFT_VEL_ION"		<< '\n'
	<< std::setw(14) << MASS_DUST         << " % MASS_DUST"         << '\n'
	<< std::setw(14) << SUPER_ION_MULT    << " % SUPER_ION_MULT"    << '\n'
	<< std::setw(14) << CHARGE_ION        << " % CHARGE_ION"        << '\n'
	<< std::setw(14) << MASS_ION          << " % MASS_ION"          << '\n'
	<< std::setw(14) << INV_DEBYE         << " % INV_DEBYE"         << '\n'
	<< std::setw(14) << SOFT_RAD_SQRD     << " % SOFT_RAD_SQRD"     << '\n'
	<< std::setw(14) << RAD_SPH_SQRD      << " % RAD_SPH_SQRD"      << '\n'
	<< std::setw(14) << RAD_CYL_SQRD      << " % RAD_CYL_SQRD"      << '\n'
	<< std::setw(14) << HALF_TIME_STEP    << " % HALF_TIME_STEP"    << '\n'
	<< std::setw(14) << ION_ION_ACC_MULT  << " % ION_ION_ACC_MULT"  << '\n'
	<< std::setw(14) << ION_DUST_ACC_MULT << " % ION_DUST_ACC_MULT" << '\n'
	<< std::setw(14) << RAD_COLL_MULT 	  << " % RAD_COLL_MULT" 	<< '\n'
	<< std::setw(14) << RAD_DUST_SQRD     << " % RAD_DUST_SQRD"     << '\n'
	<< std::setw(14) << EXTERN_ELC_MULT   << " % EXTERN_ELC_MULT"   << '\n'
	<< std::setw(14) << Q_DIV_M	      	  << " % Q_DIV_M"    		<< '\n'
	<< std::setw(14) << USE_GRAV		  << " % USE_GRAV"		 	<< '\n'
	<< std::setw(14) << LASER_PUSH		  << " % LASER_PUSH"	 	<< '\n'
	<< std::setw(14) << LASER_ON		  << " % LASER_ON"			<< '\n'
	<< std::setw(14) << LASER_OFF		  << " % LASER_OFF"		 	<< '\n'
	<< std::setw(14) << TIME_EVOL	      << " % TIME_EVOL"	 		<< '\n';
	paramOutFile.flush();

	/****** Dust Parameters ******/

	// pointer for dust positions,velocities, and accels
	float4* posDust = NULL; //4th position is dust charge
	float3* velDust = NULL;
	float3* accDust = NULL;
	float3* momIonDust = NULL;
	float3* accDust2 = NULL;

	// pointer for dust charges used in dust timestep (on host)
	float* tempCharge = NULL; 
	float* simCharge = NULL; 

	// counts the number of dust particles
	int tempNumDust = 0;

	// amount of memory required for the dust positions
	int memFloat3Dust = 0;
	int memFloat4Dust = 0;
	int memFloatDust = 0;

	// skip the first line
	std::getline(dustParamFile, line);

	// count the remaining lines in the file
	while (std::getline(dustParamFile, line)) {
		tempNumDust++;
	}

	// save the number of dust particles
	const int NUM_DUST = tempNumDust;

	// if there is at least one dust particle
	if (NUM_DUST > 0) {
		// amount of memory required for the dust variables
		memFloat3Dust = NUM_DUST * sizeof(float3);
		memFloat4Dust = NUM_DUST * sizeof(float4);
		memFloatDust  = NUM_DUST * sizeof(float);

		// allocate memory for the dust variables
		posDust = (float4*)malloc(memFloat4Dust);
		tempCharge = (float*)malloc(memFloatDust); 
		simCharge = (float*)malloc(memFloatDust); 
		velDust = (float3*)malloc(memFloat3Dust);
		accDust = (float3*)malloc(memFloat3Dust);
		momIonDust = (float3*)malloc(memFloat3Dust);
		accDust2 = (float3*)malloc(memFloat3Dust);

		// clear the end of file error flag
		dustParamFile.clear();

		// seek to the beginning  of the file
		dustParamFile.seekg(0, std::ios::beg);

		// skip the first line of the file
		std::getline(dustParamFile, line);

		// loop over the remaining lines in the file
		// saving the dust positions
		for (int i = 0; i < NUM_DUST; i++) {
			// skip the first entry in each line
			dustParamFile >> line;

			// save the dust positions
			dustParamFile >> posDust[i].x;
			dustParamFile >> posDust[i].y;
			dustParamFile >> posDust[i].z;

			dustParamFile >> velDust[i].x;
			dustParamFile >> velDust[i].y;
			dustParamFile >> velDust[i].z;

			// save the dust charge
			dustParamFile >> posDust[i].w;
		}
	}

	// set up temporary variables used for dust charge on host
	for (int i = 0; i < NUM_DUST; i++) {
		tempCharge[i] = 0;
		simCharge[i] = posDust[i].w;
	}

	// check if any of the dust particles are outside of
	// the simulation
	if(GEOMETRY == 0) {
		// loop over all of the dust grains
		for (int i = 0; i < NUM_DUST; i++) {
			// distance of the particle from the center of the sphere
			float tempDist =
				posDust[i].x*posDust[i].x +
				posDust[i].y*posDust[i].y +
				posDust[i].z*posDust[i].z;

			// check if the grain is out of the simulation sphere
			if ( tempDist > RAD_SPH_SQRD) {
				fprintf(stderr, "ERROR: Dust out of simulation\n");
				EXIT_WITH_FATAL_ERROR;
			}
		}
	} if(GEOMETRY == 1) {
		// loop over all of the dust grains
		for (int i = 0; i < NUM_DUST; i++) {
			// distance between the dust particle and the z axis
			float tempDist =
				posDust[i].x*posDust[i].x +
				posDust[i].y*posDust[i].y;

			// check if the grain is out of the simulation cylinder
			if (tempDist > RAD_CYL_SQRD || abs(posDust[i].z) > HT_CYL) {
				fprintf(stderr, "ERROR: Dust out of simulation\n");
				fprintf(stderr, "Dust grain %i pos %3.2e %3.2e %3.2e \n", i,posDust[i].x, posDust[i].y, posDust[i].z);
				EXIT_WITH_FATAL_ERROR;
			}
		}
	}

	if (debugMode) {
		debugFile << "-- First 20 Dust Positions --" << std::endl;
		debugFile << "NUM_DUST: " << NUM_DUST << std::endl;

		for (int i = 0; i < 20; i++) {
			debugFile << "X: " << posDust[i].x <<
			" Y: " << posDust[i].y <<
			" Z: " << posDust[i].z <<
			" Q: " << posDust[i].w << 
			" VX: " << velDust[i].x <<
			" VY " << velDust[i].y <<
			" VZ " << velDust[i].z << std::endl;
		}

		debugFile << std::endl;
		debugFile.flush();
	}

	// Output the number of dust to the parameter file
	paramOutFile << std::setw(14) << NUM_DUST << " % NUM_DUST\n";
	paramOutFile.flush();
	
	/****** Calculations on the Grid ******/

	// pointer for grid positions, potentials, and ion density 
	float2* gridPos = NULL;
	float* ionDensity = NULL;
	float* ionPotential = NULL;

	// amount of memory required for the grid variables
	int memFloat2Grid = NUM_GRID_PTS * sizeof(float2);
	int memFloatGrid  = NUM_GRID_PTS * sizeof(float);
	
	// allocate memory for the grid variables
	gridPos = (float2*)malloc(memFloat2Grid);
	ionDensity = (float*)malloc(memFloatGrid);
	ionPotential = (float*)malloc(memFloatGrid);
	
	//Set up grid for output number density and ion potential
	for (int z =0; z < RESZ; z++) {
		for (int x=0; x < RESX; x++) {
			gridPos[RESX* z + x].x = (-(RAD_CYL*grid_factor) + dx/2.0 + dx * x);
			//gridPos[RESX* z + x].y = 0;
			gridPos[RESX* z + x].y = (-HT_CYL*grid_factor + dz/2.0 + dz * z);
		}
	}
	
	// output all of the grid positions such that matlab can read them in
	for (int j =0; j< NUM_GRID_PTS; j++) {
		ionDensOutFile << gridPos[j].x;
		//ionDensOutFile << ", " << gridPos[j].y;
		ionDensOutFile << ", " << gridPos[j].y << std::endl;
	}
	ionDensOutFile << "" << std::endl;

	paramOutFile << std::setw(14) << NUM_GRID_PTS << " % NUM_GRID_PTS\n";
	paramOutFile << std::setw(14) << RESX << " % RESX\n";
	paramOutFile << std::setw(14) << RESZ << " % RESZ\n";
	paramOutFile.flush();
	
	// number of blocks per grid for density and potential grid points
	int blocksPerGridGrid = (NUM_GRID_PTS +1) / DIM_BLOCK;	
    /**********************/
	
    /******  Calculations for the Table Lookup for Outside Ions  *******/
	/** Integrate over the potential from ions
	 *	inside the cylinder -- these are "subtracted" from
	 *	a constant potential to get potential inside cavity. **/

	// pointer for grid positions, potential of outside ions 
	float2* GRID_POS = NULL;
	float4* GCYL_POS = NULL;
	float* Vout= NULL;

	// amount of memory required for the grid variables
	//This grid only covers half the xz-plane (r > 0)
	int NUM_GRID_PTS2 = NUM_GRID_PTS/2;
	memFloat2Grid = NUM_GRID_PTS2 * sizeof(float2);
	memFloatGrid  = NUM_GRID_PTS2 * sizeof(float);

	// allocate memory for the grid points and potential at those points 
	GRID_POS = (float2*)malloc(memFloat2Grid);
	Vout = (float*)malloc(memFloatGrid*num_pts);

	debugFile << "Memory size Vout" << memFloatGrid*num_pts  << std::endl;
	debugFile << "num_pts " << num_pts << " NUM_GRID_PTS2 " << NUM_GRID_PTS2 << "\n";

	int NUMR = RESX/2;
	float dr = RAD_CYL/(NUMR - 2);
	dz = 2.0*HT_CYL/(RESZ - 1);
	
	//Set up grid for calculating outside ion potential
	// using cylindrical symmetry, NUM_R = RESX/2
	for (int z =0; z < RESZ; z++) {
		for (int x=0; x < NUMR; x++) {
			//grid points run from -dr:dr:RAD_CYL
			GRID_POS[NUMR* z + x].x = -dr + dr * x;
			GRID_POS[NUMR* z + x].y = (-HT_CYL + dz * z);
		}
	}

	// Zero the potential at all of the grid points for all plasma conditions
	for (int p=0; p < num_pts * NUM_GRID_PTS2; p++) {
		Vout[p] = 0;
	}
	
	//output all of the grid positions such that matlab can read them in
	for (int j =0; j< NUM_GRID_PTS2; j++) {
		ionPotOutsideFile << GRID_POS[j].x;
		ionPotOutsideFile << ", " << GRID_POS[j].y << std::endl;
	}
	ionPotOutsideFile << "" << std::endl;
	
	// amount of memory required for the positions within cylinder
	int RESXc = RESX;
	int RESZc = RESZ;
	int memFloat3DGrid = RESXc * RESXc *RESZc * sizeof(float4);

	// allocate memory for the points in the cylinder
	GCYL_POS = (float4*)malloc(memFloat3DGrid);

	//Set up points within 3D cylinder -- bring in edges slightly
	// so that points don't overlap with GRID_POS
	dx = 2.0*(RAD_CYL)/(RESXc);
	float dz2 = 2.0*(HT_CYL)/(RESZc);

	float tempx;
	float tempy;
	int count = 0;

	//Set up 3d positions of points inside the cylinder
	// NUMBER OF POINTS WILL IS A MULTIPLE OF 512
	// RESZ is a multiple of 64
	// Use float4 so that memory used is a multiple of 4
	// The 4th memory position is used to tell if point is inside cylinder
	//float dr_div_debye = dx/2*INV_DEBYE;
	//float multiplier=CHARGE_SINGLE_ION*n_i*DEBYE*DEBYE/PERM_FREE_SPACE
	//					* (1-(1+dr_div_debye)*exp(-dr_div_debye));

	//float ION_OUTSIDE_MULT = COULOMB_CONST * CHARGE_ION * 
	float kq_in_box = COULOMB_CONST * CHARGE_SINGLE_ION * dx *dx *dz2;
	float TABLE_POTENTIAL_MULT = DEN_FAR_PLASMA * kq_in_box;
	for (int z =0; z < RESZc; z++) {
		for (int y=0; y < RESXc; y++) {
			for (int x=0; x < RESXc; x++) {
				tempx = -(RAD_CYL) + dx/2.0 + dx * x;
				tempy = -(RAD_CYL) + dx/2.0 + dx * y;
				//Ensure that tempx, tempy are symmetric about r=0
				//tempx = dx/2.0 + dx * x;
				//tempy = dx/2.0 + dx * y;
				//tag points inside cylinder
				if(tempx*tempx+tempy*tempy < RAD_CYL*RAD_CYL){
					GCYL_POS[count].w = 1;
				}
				else {
					GCYL_POS[count].w = 0;
				}
					GCYL_POS[count].x = tempx;
					GCYL_POS[count].y = tempy;
					GCYL_POS[count].z = -(HT_CYL) + dz2/2.0 + dz2 * z;
					count += 1;
			}
		}
	}

	//const int NUM_CYL_PTS = count;
	const int NUM_CYL_PTS = RESXc * RESXc *RESZc;
	debugFile <<  "Created cylinder positions " << std::endl;
	debugFile << NUM_CYL_PTS << std::endl << std::endl;
	//for (int j = 0; j < NUM_CYL_PTS; j++) {
	//	debugFile << GCYL_POS[j].x << ", " << GCYL_POS[j].y << ", ";
	//	debugFile << GCYL_POS[j].z << ", " << GCYL_POS[j].w << ", ";
	//	debugFile << std::endl;
	//}
	//debugFile << std::endl;

	// Need to get rid of the extra entries in GCYL_PTS -- allocated memory for
	// more than needed.  Currently just giving them a value of 0 in 4th posn.
		
	// number of blocks per grid for Table -- this is a multiple of 1024 (RESX/2*RESZ)
	const int blocksPerTable= (NUM_GRID_PTS2 +1) / DIM_BLOCK2;	
	debugFile << " blocksPerTable " << blocksPerTable << std::endl << std::endl;
	
    /**********************/
	
	/****** Time Step Parameters ******/

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

	bool MOVE_DUST = 0;
	// loop over all the commands and save them to the commands array
	for (int i = 0; i < numCommands; i++) {

		// get the next command
		timestepFile >> line;

		// convert the command to an int
		if (line == "TR-ion-pos") {
			commands[i] = 1;
		} else if (line == "TR-ion-vel") {
			commands[i] = 2;
		} else if (line == "TR-ion-acc") {
			commands[i] = 3;
		} else if (line == "CH-charge-dust") {
			if (NUM_DUST > 0){
				commands[i] = 4;
			} else {
				fprintf(stderr, "ERROR: cannot 'CH-charge-dust'");
				fprintf(stderr, " without a dust particle");
				EXIT_WITH_FATAL_ERROR;
			}
		} else if (line == "CH-move-dust") {
			if (NUM_DUST > 0){
				commands[i] = 5;
				MOVE_DUST = 1;
			} else {
				fprintf(stderr, "ERROR: cannot 'CH-move-dust'");
				fprintf(stderr, " without a dust particle");
				EXIT_WITH_FATAL_ERROR;
			}
		} else {
			// if the command does not exist give an error message
			fprintf(stderr, "ERROR on line number %d in file %s\n",
				__LINE__, __FILE__);
			fprintf(stderr, "Command \"%s\" does not exist\n\n", line.c_str());

			// terminate the program
			EXIT_WITH_FATAL_ERROR;
		}
	}

	if (debugMode) {
		debugFile << "-- Time Step Commands --" << std::endl;
		debugFile << "Commands: " << '\n'
		<< "1:  TR-ion-pos" << '\n'
		<< "2:  TR-ion-vel" << '\n'
		<< "3:  TR-ion-acc"  << '\n'
		<< "4:  CH-charge-dust" << '\n'
		<< "5:  CH-move-dust" << '\n';
		debugFile << "--------------------" << std::endl;
		debugFile << "Number of commands: " << numCommands << std::endl;

		for (int i = 0; i < numCommands; i++) {
			debugFile << "i: " << i << " | " << commands[i] << std::endl;
		}

		debugFile << "--------------------" << std::endl << std::endl;
		debugFile.flush();
	}

	/****** Initialize Host Variables ******/

	// holds electron current to a dust grain in the time step in C
	float elcCurrent = 0;
    //float delta_q = 0; //change in charge in one time step

	// holds a dust grain potential in the time step in V
	float dustPotential = 0;

	// number of blocks per grid for ions
	int blocksPerGridIon = (NUM_ION + 1) / DIM_BLOCK;

	// memory size for float4 type ion data arrays
	int memFloat4Ion = NUM_ION * sizeof(float4);

	// allocate memory for the ion positions
	float4* posIon = (float4*)malloc(memFloat4Ion);

	// allocate memory for the ion velocities
	float4* velIon = (float4*)malloc(memFloat4Ion);

	// allocate memory for the ion accelerations
	float4* accIon = (float4*)malloc(memFloat4Ion);

	// allocate memory for ion accel due to dust
	float4* accIonDust = (float4*)malloc(memFloat4Ion);

	// allocate memory for dust accel due to ion
	float4* accDustIon = (float4*)malloc(memFloat4Ion * NUM_DUST);

	// allocate memory for the ion bounds flag
	int* boundsIon = (int*)malloc(NUM_ION * sizeof(int));

	// allocate memory for the ion adaptive timestep depth
	int* m = (int*)malloc(NUM_ION * sizeof(int));

	// allocate memory for the timestep factor for adaptive timestep
	int* timeStepFactor = (int*)malloc(NUM_ION * sizeof(int));

	// allocate memory for the closest dust particle
	float* minDistDust = (float*)malloc(NUM_ION * sizeof(float));

	// set all ions to in-bounds, minDistDust to large number
	for (int i = 0; i < NUM_ION; i++) {
		boundsIon[i] = 0;
		minDistDust[i] = 1000;
		m[i] = 0;
		timeStepFactor[i] = 1;
	}

	// allocate memory for the ion current to each dust particle
	//int* ionCurrent = new int[NUM_DUST];
	float* ionCurrent = new float[NUM_DUST];

	// set initial currents to 0
	for (int i = 0; i < NUM_DUST; i++) {
		ionCurrent[i] = 0.0;
	}

	// seed the random number generator
	srand (time(NULL));

	// numbers used when calculating random positions and velocities
	int number = 1000;
	float randNum;

	// holds the distance of each ion from the center of the simulation sphere
	float dist;
	
	// set direction of the axial electric field
	int E_direction;

	// allocate variables used in dust-dust forces
	float3 distdd;
	float distSquared;
	float linForce;
		
	// initialize the dust velocities, accelerations and momentum transfer 
	for (int i = 0; i < NUM_DUST; i++)
	{
		accDust[i].x = OMEGA_DIV_M * simCharge[i] * posDust[i].x;
		accDust[i].y = OMEGA_DIV_M * simCharge[i] * posDust[i].y;
		//polarity switching
		accDust[i].z = simCharge[i] / MASS_DUST * E_FIELD;
		if(USE_GRAV ==1) {
			accDust[i].z -= 9.81;
		}
		momIonDust[i].x = 0;
		momIonDust[i].y = 0;
		momIonDust[i].z = 0;
	}


	// attempt to open input file for initial ion positions and velocities
	fileName = inputDirName + "init-ions.txt";
	std::ifstream ionInitFile(fileName.c_str());
	// check if the file opened
	bool init_ions_from_file = ionInitFile.is_open();

	if( init_ions_from_file ) { // initialize ion data from file

		for( int i=0 ; i<NUM_ION ; i++ ) {
			ionInitFile >> 	posIon[i].x;
			ionInitFile >> 	posIon[i].y;
			ionInitFile >> 	posIon[i].z;

			// read in the charge
			ionInitFile >>  posIon[i].w;

			ionInitFile >>  velIon[i].x;
			ionInitFile >>  velIon[i].y;
			ionInitFile >>  velIon[i].z;
			// initialize velIon.w to zero
			velIon[i].w = 0.0;
		}

		ionInitFile.close();

	} else { // as no initial ion data was given, initialize ion data with 
			 //zeros or random values
		// loop over all the ions and initialize their velocity, acceleration,
		// and position
		for (int i = 0; i < NUM_ION; i++) {
			//set the charge on the ion
			posIon[i].w = CHARGE_SINGLE_ION*SUPER_ION_MULT;
			//set the position of the ion
			if(GEOMETRY == 0) {
				// give the ion a random position
				randNum = (((rand() % (number*2)) - number) / (float)number);
				posIon[i].x = randNum * RAD_SPH;
				randNum = (((rand() % (number*2)) - number) / (float)number);
				posIon[i].y = randNum * RAD_SPH;
				randNum = (((rand() % (number*2)) - number) / (float)number);
				posIon[i].z = randNum * RAD_SPH;

				// calculate the distance from the ion to the center of the
				// simulation sphere
				dist = posIon[i].x * posIon[i].x +
				posIon[i].y * posIon[i].y +
				posIon[i].z * posIon[i].z;

				// while the ion is outside of the simulation sphere, give it
				// a new random position.
				while (dist > RAD_SPH * RAD_SPH) {
					randNum = (((rand() % (number*2)) - number) / (float)number);
					posIon[i].x = randNum * RAD_CYL;
					randNum = (((rand() % (number*2)) - number) / (float)number);
					posIon[i].y = randNum * RAD_CYL;
					randNum = (((rand() % (number*2)) - number) / (float)number);
					posIon[i].z = randNum * RAD_CYL;

					// recalculate the distance to the center of the simulation
					dist = posIon[i].x * posIon[i].x +
					posIon[i].y * posIon[i].y +
					posIon[i].z * posIon[i].z;
				}
			} else if(GEOMETRY == 1) {
				// give the ion a random position
				randNum = (((rand() % (number*2)) - number) / (float)number);
				posIon[i].x = randNum * RAD_CYL;
				randNum = (((rand() % (number*2)) - number) / (float)number);
				posIon[i].y = randNum * RAD_CYL;
				randNum = (((rand() % (number*2)) - number) / (float)number);
				posIon[i].z = randNum * HT_CYL;

				// calculate the distance from the ion to the center of the
				// simulation cylinder
				dist = posIon[i].x * posIon[i].x + posIon[i].y * posIon[i].y;

				// while the ion is outside of the simulation cylinder, give it
				// a new random position.
				while (dist > RAD_CYL * RAD_CYL){
					randNum = (((rand() % (number*2)) - number) / (float)number);
					posIon[i].x = randNum * RAD_CYL;
					randNum = (((rand() % (number*2)) - number) / (float)number);
					posIon[i].y = randNum * RAD_CYL;

					// recalculate the distance to the center of the simulation
					dist = posIon[i].x * posIon[i].x +
					posIon[i].y * posIon[i].y;
				}
			}

			// give the ion an initial random velocity
			randNum = (((rand() % (number*2)) - number) / (float)number);
			velIon[i].x = ION_SPEED * randNum;
			randNum = (((rand() % (number*2)) - number) / (float)number);
			velIon[i].y = ION_SPEED * randNum;
			randNum = ((rand() % (number*2)) / (float)number) + 2.0*MACH;
			velIon[i].z = - ION_SPEED * randNum;
			velIon[i].w = 0.0;
			
		}
		
	}

	for( int i=0 ; i<NUM_ION ; i++) {
		// set the initial acceleration to 0
		accIon[i].x = 0;
		accIon[i].y = 0;
		accIon[i].z = 0;
		accIon[i].w = 0;

		// set the initial IonDust acceleration to 0
		accIonDust[i].x = 0;
		accIonDust[i].y = 0;
		accIonDust[i].z = 0;
		accIonDust[i].w = 0;

	}

	if (debugMode) {

		debugFile << "-- Initialized From File --" << '\n'
		<< "Ions From File: " << init_ions_from_file << '\n' << '\n';

		debugFile << "-- Basic Memory Sizes --" << '\n'
		<< "float  " << sizeof(float) << '\n'
		<< "int    " << sizeof(int) << '\n'
		<< "float3 " << sizeof(float3) << '\n'
		<< "float4 " << sizeof(float4) << '\n' << '\n';

		debugFile << "-- Host Memory Use --" << '\n'
		<< "velIon  " 		  << sizeof(*velIon) * NUM_ION << '\n'
		<< "posIon  " 		  << sizeof(*posIon) * NUM_ION << '\n'
		<< "accIon  " 		  << sizeof(*accIon) * NUM_ION << '\n'
		<< "accIonDust  " 	  << sizeof(*accIonDust) * NUM_ION << '\n'
		<< "accDustIon  " 	  << sizeof(*accDustIon) * NUM_ION*NUM_DUST<< '\n'
		<< "boundsIon  " 	  << sizeof(*boundsIon) * NUM_ION << '\n'
		<< "m  " 			  << sizeof(*m) * NUM_ION << '\n'
		<< "timeStepFactor  " << sizeof(*timeStepFactor) * NUM_ION << '\n'
		<< "minDistDust  "    << sizeof(*minDistDust) * NUM_ION << '\n'
		<< "ionCurrent  "     << sizeof(*ionCurrent) * NUM_DUST << '\n'
		<< '\n';

		debugFile << "-- Initial Host Variables --" << std::endl;
		debugFile << "First 10 ion positions: " << std::endl;
		for (int i = 0; i < 10; i++) {
			debugFile << "X: " << posIon[i].x <<
			" Y: " << posIon[i].y <<
			" Z: " << posIon[i].z << std::endl;
		}

		debugFile << std::endl << "Last 10 ion positions: " << std::endl;
		for (int i = 1; i <= 10; i++) {
			int ID = NUM_ION - i;
			debugFile << "X: "  << posIon[ID].x
			<< " Y: " << posIon[ID].y
			<< " Z: " << posIon[ID].z
			<< std::endl;
		}

		debugFile << std::endl << "First 10 ion velocities: " << std::endl;
		for (int i = 0; i < 10; i++) {
			debugFile << "X: " << velIon[i].x <<
			" Y: " << velIon[i].y <<
			" Z: " << velIon[i].z << std::endl;
		}

		debugFile << std::endl << "Last 10 ion velocities: " << std::endl;
		for (int i = 1; i <= 10; i++) {
			int ID = NUM_ION - i;
			debugFile << "X: "  << velIon[ID].x
			<< " Y: " << velIon[ID].y
			<< " Z: " << velIon[ID].z
			<< std::endl;
		}

		debugFile << std::endl;
		debugFile.flush();
	}

	/****** Initialize Device Variables ******/
	
	bool print = false;
	roadBlock_104(statusFile, __LINE__, __FILE__, "before variables", print);

	// create constant device variables
	constCUDAvar<float> d_COULOMB_CONST(&COULOMB_CONST, 1);
	constCUDAvar<float> d_ION_ION_ACC_MULT(&ION_ION_ACC_MULT, 1);
	constCUDAvar<int> d_NUM_DIV_QTH(&NUM_DIV_QTH, 1);
	constCUDAvar<int> d_NUM_DIV_VEL(&NUM_DIV_VEL, 1);
	constCUDAvar<int> d_NUM_ION(&NUM_ION, 1);
	constCUDAvar<int> d_NUM_DUST(&NUM_DUST, 1);
	constCUDAvar<float> d_RAD_DUST(&RAD_DUST, 1);
	constCUDAvar<float> d_RAD_DUST_SQRD(&RAD_DUST_SQRD, 1);
	constCUDAvar<float> d_SOFT_RAD_SQRD(&SOFT_RAD_SQRD, 1);
	constCUDAvar<float> d_M_FACTOR(&M_FACTOR, 1);
	constCUDAvar<float> d_RAD_SPH(&RAD_SPH, 1);
	constCUDAvar<float> d_RAD_SPH_SQRD(&RAD_SPH_SQRD, 1);
	constCUDAvar<float> d_RAD_CYL(&RAD_CYL, 1);
	constCUDAvar<float> d_RAD_CYL_SQRD(&RAD_CYL_SQRD, 1);
	constCUDAvar<float> d_HT_CYL(&HT_CYL, 1);
	constCUDAvar<float> d_ION_TIME_STEP(&ION_TIME_STEP, 1);
	constCUDAvar<float> d_HALF_TIME_STEP(&HALF_TIME_STEP, 1);
	constCUDAvar<float> d_Q_DIV_M(&Q_DIV_M, 1);
	constCUDAvar<float> d_TEMP_GAS(&TEMP_GAS, 1);
	constCUDAvar<float> d_PI(&PI, 1);
	constCUDAvar<float> d_MASS_SINGLE_ION(&MASS_SINGLE_ION, 1);
	constCUDAvar<float> d_BOLTZMANN(&BOLTZMANN, 1);
	constCUDAvar<float> d_ION_DUST_ACC_MULT(&ION_DUST_ACC_MULT, 1);
	constCUDAvar<float> d_DUST_ION_ACC_MULT(&DUST_ION_ACC_MULT, 1);
	constCUDAvar<int> d_MAX_DEPTH(&MAX_DEPTH, 1);
	constCUDAvar<int> d_I_CS_RANGES(&I_CS_RANGES, 1);
	constCUDAvar<float> d_TOT_ION_COLL_FREQ(&totIonCollFreq, 1);
	constCUDAvar<int> d_NUM_CYL_PTS(&NUM_CYL_PTS, 1);
	constCUDAvar<int> d_NUMR(&NUMR, 1);
	constCUDAvar<int> d_RESZ(&RESZ, 1);
	constCUDAvar<int> d_NUM_GRID_PTS2(&NUM_GRID_PTS2, 1);
	constCUDAvar<float> d_dr(&dr, 1);
	constCUDAvar<float> d_dz(&dz, 1);
	constCUDAvar<float> d_kq_in_box(&kq_in_box, 1);

	// create device pointers
	CUDAvar<int> d_boundsIon(boundsIon, NUM_ION);
	CUDAvar<int> d_m(m, NUM_ION);
	CUDAvar<int> d_timeStepFactor(timeStepFactor, NUM_ION);
	CUDAvar<float> d_QCOM(NUM_DIV_QTH*num_pts);
	CUDAvar<float> d_VCOM(NUM_DIV_VEL*num_pts);
	CUDAvar<float> d_GCOM(NUM_DIV_QTH * NUM_DIV_VEL * num_pts);
	CUDAvar<float4> d_posIon(posIon, NUM_ION);
	CUDAvar<float4> d_velIon(velIon, NUM_ION);
	CUDAvar<float4> d_accIon(accIon, NUM_ION);
	CUDAvar<float4> d_accIonDust(accIonDust, NUM_ION);
	CUDAvar<float4> d_posDust(posDust, NUM_DUST);
	CUDAvar<float> d_minDistDust(minDistDust, NUM_ION);
	CUDAvar<float4> d_accDustIon(accDustIon, NUM_DUST * NUM_ION);
	CUDAvar<float2> d_gridPos(gridPos, NUM_GRID_PTS);
	CUDAvar<float> d_ionPotential(ionPotential, NUM_GRID_PTS);
	CUDAvar<float> d_ionDensity(ionDensity, NUM_GRID_PTS);
	CUDAvar<float> d_SIGMA_I1(sigma_i1, I_CS_RANGES+1);
	CUDAvar<float> d_SIGMA_I2(sigma_i2, I_CS_RANGES+1);
	CUDAvar<float> d_SIGMA_I_TOT(sigma_i_tot, I_CS_RANGES+1);
	CUDAvar<int> d_collList(collList, NUM_ION);
	CUDAvar<int> d_collision_counter(&collision_counter, 1);
	CUDAvar<float2> d_GRID_POS(GRID_POS, NUM_GRID_PTS2);
	CUDAvar<float4> d_GCYL_POS(GCYL_POS, NUM_CYL_PTS);
	CUDAvar<float>	d_Vout(Vout, NUM_GRID_PTS2*num_pts);
	CUDAvar<curandState_t> randStates(NUM_ION);

	float hardDist = 0;
	CUDAvar<float> d_hardDist(&hardDist, 1);
	CUDAvar<float> d_linForce(&linForce, 1);
	d_hardDist.hostToDev();
	d_linForce.hostToDev();

	// Copy over values
	d_boundsIon.hostToDev();
	d_m.hostToDev();
	d_timeStepFactor.hostToDev();
	d_QCOM.hostToDev();
	d_VCOM.hostToDev();
	d_GCOM.hostToDev();
	d_posIon.hostToDev();
	d_velIon.hostToDev();
	d_accIon.hostToDev();
	d_accIonDust.hostToDev();
	d_posDust.hostToDev();
	d_minDistDust.hostToDev();
	d_accDustIon.hostToDev();
	d_gridPos.hostToDev();
	d_ionPotential.hostToDev();
	d_ionDensity.hostToDev();
	d_SIGMA_I1.hostToDev();
	d_SIGMA_I2.hostToDev();
	d_SIGMA_I_TOT.hostToDev();
	d_collList.hostToDev();
	d_collision_counter.hostToDev();
	d_GRID_POS.hostToDev();
	d_GCYL_POS.hostToDev();


	debugFile << "Calc'ing evolving params" << std::endl;

	// Initialize evolving parameters for time-dependent plasma conditions
	if(TIME_EVOL >0) {
		TEMP_ELC = evolTe[plasma_counter];
		TEMP_ION = evolTi[plasma_counter];
		DEN_FAR_PLASMA = evolni[plasma_counter];
		MACH = evolMach[plasma_counter];
		E_FIELD = evolEz[plasma_counter];
		DRIFT_VEL_ION = evolVz[plasma_counter];
		E_FIELDR = evolEr[plasma_counter];

		DEBYE = sqrt((PERM_FREE_SPACE * BOLTZMANN * TEMP_ELC)/
			(evolne[plasma_counter] * CHARGE_ELC * CHARGE_ELC));
		INV_DEBYE = 1.0 / DEBYE;
		SUPER_ION_MULT = SIM_VOLUME * evolni[plasma_counter] / NUM_ION;
		CHARGE_ION = CHARGE_SINGLE_ION * SUPER_ION_MULT;
		MASS_ION = MASS_SINGLE_ION * SUPER_ION_MULT;
		ION_ION_ACC_MULT = COULOMB_CONST * Q_DIV_M;
		SOUND_SPEED = sqrt(BOLTZMANN * TEMP_ELC / MASS_SINGLE_ION);
		vs_sq = 8 * BOLTZMANN * TEMP_ION / PI / MASS_SINGLE_ION 
			+ DRIFT_VEL_ION * DRIFT_VEL_ION;
		RAD_COLL_MULT = 
			2 * Q_DIV_M * COULOMB_CONST / RAD_DUST / vs_sq;
		ELC_CURRENT_0 = 4.0 * PI * RAD_DUST_SQRD * evolne[plasma_counter]*
			CHARGE_ELC * sqrt((BOLTZMANN * TEMP_ELC)/(2.0 * PI * ELC_MASS));
		EXTERN_ELC_MULT =
       		((RAD_SPH / DEBYE) + 1.0) * exp(-RAD_SPH / DEBYE) *
       		(CHARGE_SINGLE_ION * DEN_FAR_PLASMA * DEBYE) *
       		(Q_DIV_M) / (PERM_FREE_SPACE);
		debugFile << plasma_counter << ", " << CHARGE_ION << std::endl;
	}

	//Create CUDA device pointers for the evolving parameters
	CUDAvar<float> d_DEN_FAR_PLASMA(&DEN_FAR_PLASMA, 1);
	CUDAvar<float> d_INV_DEBYE(&INV_DEBYE, 1);
	CUDAvar<float> d_E_FIELD(&E_FIELD, 1);
	CUDAvar<float> d_E_FIELDR(&E_FIELDR, 1);
	CUDAvar<float> d_TEMP_ION(&TEMP_ION, 1);
	CUDAvar<float> d_CHARGE_ION(&CHARGE_ION, 1);
	CUDAvar<float> d_DRIFT_VEL_ION(&DRIFT_VEL_ION, 1);
	CUDAvar<float> d_SOUND_SPEED(&SOUND_SPEED, 1);
	CUDAvar<float> d_RAD_COLL_MULT(&RAD_COLL_MULT, 1);
	CUDAvar<float> d_EXTERN_ELC_MULT(&EXTERN_ELC_MULT, 1);
	CUDAvar<float> d_TEMP_ELC(&TEMP_ELC, 1);
	CUDAvar<float> d_MACH(&MACH, 1);
	CUDAvar<float> d_TABLE_POTENTIAL_MULT(&TABLE_POTENTIAL_MULT, 1);

	// copy evolving variables to the device
	d_DEN_FAR_PLASMA.hostToDev();
	d_INV_DEBYE.hostToDev();
	d_E_FIELD.hostToDev();
	d_E_FIELDR.hostToDev();
	d_TEMP_ION.hostToDev();
	d_CHARGE_ION.hostToDev();
	d_DRIFT_VEL_ION.hostToDev();
	d_SOUND_SPEED.hostToDev();
	d_RAD_COLL_MULT.hostToDev();
	d_EXTERN_ELC_MULT.hostToDev();
	d_TEMP_ELC.hostToDev();
	d_MACH.hostToDev();

	roadBlock_104(statusFile, __LINE__, __FILE__, "before init_101", print);

	for (int p = 0; p < num_pts; p++) {
	
		DEBYE = sqrt((PERM_FREE_SPACE * BOLTZMANN * evolTe[p])/
			(evolne[p] * CHARGE_ELC * CHARGE_ELC));
		INV_DEBYE = 1.0 / DEBYE;
		TABLE_POTENTIAL_MULT = evolni[p] * kq_in_box; 
		d_INV_DEBYE.hostToDev();
		d_TABLE_POTENTIAL_MULT.hostToDev();

		//Set the radial electric field for boundary condition
		boundaryEField_101<<<blocksPerTable, DIM_BLOCK2, sizeof(float4) * DIM_BLOCK2>>>
			(d_GRID_POS.getDevPtr(),
			d_GCYL_POS.getDevPtr(),
			d_NUM_CYL_PTS.getDevPtr(),
			d_NUM_GRID_PTS2.getDevPtr(),
			d_INV_DEBYE.getDevPtr(),
			d_TABLE_POTENTIAL_MULT.getDevPtr(),
			d_Vout.getDevPtr(), p);

		roadBlock_104( statusFile, __LINE__, __FILE__, "boundaryEField_101", print);	
	}
	
	//Reset the DEBYE length to first condition -- 
	DEBYE = sqrt((PERM_FREE_SPACE * BOLTZMANN * evolTe[0])/
		(evolne[0] * CHARGE_ELC * CHARGE_ELC));
	INV_DEBYE = 1.0 / DEBYE;
	d_INV_DEBYE.hostToDev();

	// Copy the table to the host 
	d_Vout.devToHost();

	for (int p = 0; p < num_pts; p++) {
		//output potential at the grid positions such that matlab can read them in
		for (int j =0; j< NUM_GRID_PTS2; j++) {
			ionPotOutsideFile << Vout[p*NUM_GRID_PTS2 + j] << std::endl;
		}
		ionPotOutsideFile << "" << std::endl;
	}

	//Set the potential and density of ions on the grid to zero
	zeroIonDensityPotential_102 <<<blocksPerGridGrid, DIM_BLOCK >>>
		(d_ionPotential.getDevPtr(),
		 d_ionDensity.getDevPtr());

	roadBlock_104(  statusFile, __LINE__, __FILE__, "zeroIonDensityPotential", print);

	// zero the ionDustAcc
	zeroDustIonAcc_103<<<blocksPerGridIon, DIM_BLOCK >>>
		(d_accDustIon.getDevPtr(),
		d_NUM_DUST.getDevPtr(),
		d_NUM_ION.getDevPtr());

   	roadBlock_104(  statusFile, __LINE__, __FILE__, "zeroDustIonAcc", print);

	roadBlock_104(statusFile, __LINE__, __FILE__, "before init_101", print);

	// generate all of the random states on the GPU
	init_101 <<< DIM_BLOCK * blocksPerGridIon, 1 >>> (time(0), randStates.getDevPtr());

	roadBlock_104(statusFile, __LINE__, __FILE__, "init_101", print);

	// initialize variables needed for injecting ions with the Piel 2017 method
	if(GEOMETRY == 0) {
		initInjectIonSphere_101(
			NUM_DIV_QTH,
			NUM_DIV_VEL,
			TEMP_ELC,
			TEMP_ION,
			DRIFT_VEL_ION,
			MACH,
			MASS_SINGLE_ION,
			BOLTZMANN,
			PI,
			d_QCOM.getDevPtr(),
			d_VCOM.getDevPtr(),
			d_GCOM.getDevPtr(),
			debugMode,
			debugFile);
	} else if(GEOMETRY == 1) {
		initInjectIonCylinder_101(
			NUM_DIV_QTH,
			NUM_DIV_VEL,
			TIME_EVOL,
			RAD_CYL,
			HT_CYL,
			evolTe,
			evolTi,
			evolVz,	
			evolMach,
			MASS_SINGLE_ION,
			BOLTZMANN,
			PI,
			d_QCOM.getDevPtr(),
			d_VCOM.getDevPtr(),
			d_GCOM.getDevPtr(),
			debugMode,
			debugFile);

	}

	roadBlock_104( statusFile, __LINE__, __FILE__, "Pause before timestep", print);	

	//debugFile << "out of initInjectIonCyl" << std::endl;

	/*************************
		Time Step
	*************************/

	/****** Init acc. and Kick for 1/2 Step ******/

	//First make sure that no ions are inside dust
	checkIonDustBounds_101 <<< blocksPerGridIon, DIM_BLOCK >>> (
		d_posIon.getDevPtr(), // {{{
		d_boundsIon.getDevPtr(), // <-->
		d_RAD_DUST_SQRD.getDevPtr(),
		d_NUM_DUST.getDevPtr(),
		d_posDust.getDevPtr()); 

	roadBlock_104(statusFile, __LINE__, __FILE__, "checkIonBounds_101", print);

	//polarity switching of electric field
	int xac = 0;

	if(E_FIELD > 0){
		xac = 0;}
	else{
		xac = 1;}

	// inject ions on the boundary of the simulation
	if(GEOMETRY == 0) {
		injectIonSphere_101 <<< blocksPerGridIon, DIM_BLOCK >>> (
			d_posIon.getDevPtr(), 
			d_velIon.getDevPtr(),
			d_accIon.getDevPtr(),
			randStates.getDevPtr(),
			d_RAD_SPH.getDevPtr(),
			d_boundsIon.getDevPtr(),
			d_GCOM.getDevPtr(),
			d_QCOM.getDevPtr(),
			d_VCOM.getDevPtr(),
			d_NUM_DIV_QTH.getDevPtr(),
			d_NUM_DIV_VEL.getDevPtr(),
			d_SOUND_SPEED.getDevPtr(),
			d_TEMP_ION.getDevPtr(),
			d_PI.getDevPtr(),
			d_TEMP_ELC.getDevPtr(),
			d_MACH.getDevPtr(),
			d_MASS_SINGLE_ION.getDevPtr(),
			d_BOLTZMANN.getDevPtr(),
			d_CHARGE_ION.getDevPtr(),
			xac);

		roadBlock_104( statusFile, __LINE__, __FILE__, "injectIonSphere_101", print);

	} else if(GEOMETRY == 1) {
		// fraction of plasma timestep
		counter_part = 0;

		injectIonCylinder_101 <<< blocksPerGridIon, DIM_BLOCK >>> (
			d_posIon.getDevPtr(), // -->
			d_velIon.getDevPtr(), // -->
			d_accIon.getDevPtr(), // -->
			randStates.getDevPtr(),
			d_RAD_CYL.getDevPtr(),
			d_HT_CYL.getDevPtr(),
			d_boundsIon.getDevPtr(), // <-- 
			d_GCOM.getDevPtr(),
			d_QCOM.getDevPtr(),
			d_VCOM.getDevPtr(),
			d_NUM_DIV_QTH.getDevPtr(),
			d_NUM_DIV_VEL.getDevPtr(),
			d_SOUND_SPEED.getDevPtr(),
			d_TEMP_ION.getDevPtr(),
			d_PI.getDevPtr(),
			d_TEMP_ELC.getDevPtr(),
			d_MACH.getDevPtr(),
			d_MASS_SINGLE_ION.getDevPtr(),
			d_BOLTZMANN.getDevPtr(),
			d_CHARGE_ION.getDevPtr(),
			plasma_counter,counter_part,
			xac); // <--

		roadBlock_104( statusFile, __LINE__, __FILE__, "injectIonCylinder_101", print);
	}

	// reset the ion bounds flag to 0
	resetIonBounds_101 <<< blocksPerGridIon, DIM_BLOCK >>> (
		d_boundsIon.getDevPtr());

	roadBlock_104( statusFile, __LINE__, __FILE__, "resetIonBounds_101", print);

	if (xac == 0) {
		E_direction = -1;
	}
	else {
		E_direction = 1;
	}

	//Calculate ion-ion, ion-outside ion, and E_FIELD accelerations
	calcIonAccels_102 <<< blocksPerGridIon, DIM_BLOCK,sizeof(float4) * DIM_BLOCK >>>(
		d_posIon.getDevPtr(), // <--
		d_accIon.getDevPtr(), // <-->
		d_NUM_ION.getDevPtr(), 
		d_SOFT_RAD_SQRD.getDevPtr(),
		d_ION_ION_ACC_MULT.getDevPtr(),
		d_INV_DEBYE.getDevPtr(),
		d_Q_DIV_M.getDevPtr(),
		d_HT_CYL.getDevPtr(),
		d_Vout.getDevPtr(),
		d_NUMR.getDevPtr(),
		d_RESZ.getDevPtr(),
		d_dz.getDevPtr(),
		d_dr.getDevPtr(),
		d_E_FIELD.getDevPtr(),
		d_E_FIELDR.getDevPtr(),
		E_direction,
		plasma_counter,
		GEOMETRY,
		d_EXTERN_ELC_MULT.getDevPtr());

	roadBlock_104(statusFile, __LINE__, __FILE__, "calcIonAccels_102", print);

	//Calculate ion-ion forces
	//Ions inside the simulation region
	// calculate the acceleration due to ion-ion interactions
	//calcIonIonAcc_102 <<< blocksPerGridIon, DIM_BLOCK,sizeof(float4) * DIM_BLOCK >>>(
	//	d_posIon.getDevPtr(), // <--
	//	d_accIon.getDevPtr(), // <-->
	//	d_NUM_ION.getDevPtr(), 
	//	d_SOFT_RAD_SQRD.getDevPtr(),
	//	d_ION_ION_ACC_MULT.getDevPtr(),
	//	d_INV_DEBYE.getDevPtr());
//
//	roadBlock_104(  statusFile, __LINE__, __FILE__, "calcIonIonAcc_102", print);

	// Calculate the ion accelerations due to the ions outside of
	// the simulation cavity
	//if(GEOMETRY == 0) {
		// calculate the forces between all ions
	//	calcExtrnElcAcc_102 <<< blocksPerGridIon, DIM_BLOCK >>> (
	//		d_accIon.getDevPtr(), // {{{
	//		d_posIon.getDevPtr(),
	//		d_EXTERN_ELC_MULT.getDevPtr(),
	//		d_INV_DEBYE.getDevPtr());
//
//		roadBlock_104(statusFile, __LINE__, __FILE__, "calcExtrnElcAcc_102", print);
	//} else if(GEOMETRY == 1) {
	//	// calculate the forces from ions outside simulation region
	//	// and external electric field 
	//	calcExtrnElcAccCyl_102 <<< blocksPerGridIon, DIM_BLOCK >>> (
	//		d_accIon.getDevPtr(), // {{{
	//		d_posIon.getDevPtr(), 
	//		d_Q_DIV_M.getDevPtr(),
	//		d_HT_CYL.getDevPtr(),
	//		d_Vout.getDevPtr(),
	//		d_NUMR.getDevPtr(),
	//		d_RESZ.getDevPtr(),
	//		d_dz.getDevPtr(),
	//		d_dr.getDevPtr(),
	//		d_E_FIELD.getDevPtr(),
	//		E_direction,
	//		plasma_counter);
//
//		roadBlock_104(statusFile, __LINE__, __FILE__, "calcExtrnElcAccCyl_102", print);
//	}

	//Any other external forces acting on ions would be calc'd here
	// Kick for 1/2 a timestep -- using just ion-ion accels
	kick_100<<< blocksPerGridIon, DIM_BLOCK >>> (
		d_velIon.getDevPtr(), // {{{
		d_accIon.getDevPtr(), 
		d_HALF_TIME_STEP.getDevPtr());

	roadBlock_104(  statusFile, __LINE__, __FILE__, "kick_100", print);

	// calculate the acceleration due to ion-dust interactions
	// also save the distance to the closest dust particle for each ion
	calcIonDustAcc_102 <<< blocksPerGridIon, DIM_BLOCK >>> (
		d_posIon.getDevPtr(), // {{{
		d_accIonDust.getDevPtr(), // <-->
		d_posDust.getDevPtr(), // <--
		d_NUM_ION.getDevPtr(),
		d_NUM_DUST.getDevPtr(),
		d_SOFT_RAD_SQRD.getDevPtr(),
		d_ION_DUST_ACC_MULT.getDevPtr(),
		d_minDistDust.getDevPtr());

	roadBlock_104(  statusFile, __LINE__, __FILE__, "calcIonDustAcc_102 line 1734", print);
	
	/****** Time Step Loop ******/
	float flip_Ez = 1.0; //for changing dir of Ez with DC switch

	for (int i = 1; i <= NUM_TIME_STEP; i++)   
	//NUM_TIME_STEP is the number of dust time steps
	{
		// print the time step number to the status file
		statusFile << i << ": "<< std::endl;

		// Ion Time Step Loop
		for (int j = 1; j <= N_IONDT_PER_DUSTDT; j++){

			//Select the time step depth
			//select_100 <<< blocksPerGridIon, DIM_BLOCK >>> (
		//		d_posIon.getDevPtr(), // <--
		//		d_posDust.getDevPtr(), // <--
		//		d_velIon.getDevPtr(), // <-- (TS1: rand + 1/2 ion-ion kick )
		//		d_minDistDust.getDevPtr(), // <-- (TS1: good)
		//		d_RAD_DUST.getDevPtr(),
		//		d_ION_TIME_STEP.getDevPtr(),
		//		d_MAX_DEPTH.getDevPtr(),
		//		d_M_FACTOR.getDevPtr(), 
		//		d_NUM_DUST.getDevPtr(),
		//		d_m.getDevPtr(), // -->
		//		d_timeStepFactor.getDevPtr()); // -->
//	
//			roadBlock_104( statusFile, __LINE__, __FILE__, "select_100", print);

			//KDK using just the ion-dust acceleration for s^m iterations
 
			if(GEOMETRY == 0) {
				KDK_100 <<< blocksPerGridIon, DIM_BLOCK >>> (
					d_posIon.getDevPtr(), // {{{
					d_velIon.getDevPtr(), // <-->
					d_accIon.getDevPtr(), // <-->
					d_accIonDust.getDevPtr(), // <--
					d_boundsIon.getDevPtr(), // <-->
					d_minDistDust.getDevPtr(),
					d_M_FACTOR.getDevPtr(),
					d_ION_TIME_STEP.getDevPtr(), 
					GEOMETRY,
					d_RAD_SPH_SQRD.getDevPtr(),
					NULL,
					d_RAD_DUST.getDevPtr(),
					d_NUM_DUST.getDevPtr(),
					d_posDust.getDevPtr(), // <--
					d_NUM_ION.getDevPtr(),
					d_SOFT_RAD_SQRD.getDevPtr(),
					d_ION_DUST_ACC_MULT.getDevPtr(),
					d_RAD_COLL_MULT.getDevPtr());

				roadBlock_104(  statusFile, __LINE__, __FILE__, "KDK_100", print);
				
			} else if(GEOMETRY == 1) {
				KDK_100 <<< blocksPerGridIon, DIM_BLOCK >>> (
					d_posIon.getDevPtr(), 
					d_velIon.getDevPtr(), 
					d_accIon.getDevPtr(), 
					d_accIonDust.getDevPtr(),
					d_boundsIon.getDevPtr(),
					d_minDistDust.getDevPtr(),
					d_M_FACTOR.getDevPtr(),
					d_ION_TIME_STEP.getDevPtr(),
					GEOMETRY,
					d_RAD_CYL_SQRD.getDevPtr(),
					d_HT_CYL.getDevPtr(),
					d_RAD_DUST.getDevPtr(),
					d_NUM_DUST.getDevPtr(),
					d_posDust.getDevPtr(), 
					d_NUM_ION.getDevPtr(),
					d_SOFT_RAD_SQRD.getDevPtr(),
					d_ION_DUST_ACC_MULT.getDevPtr(),
					d_RAD_COLL_MULT.getDevPtr());

				roadBlock_104(  statusFile, __LINE__, __FILE__, "KDK_100", print);
			}

			// calc ion number density and ion potential
			calcIonDensityPotential_102 
				<<< blocksPerGridGrid, DIM_BLOCK, sizeof(float4) * DIM_BLOCK >>> (
				d_gridPos.getDevPtr(), // {{{
				 d_posIon.getDevPtr(),
				 d_COULOMB_CONST.getDevPtr(),
				 d_INV_DEBYE.getDevPtr(),
				 d_NUM_ION.getDevPtr(),
				 d_ionPotential.getDevPtr(),
				 d_ionDensity.getDevPtr());
			roadBlock_104(statusFile, __LINE__, __FILE__, "ionDensityPotential", print);

			//polarity switching of electric field
			// Need to track dust_time + ion_time
			ionTime = dust_time + (j)* ION_TIME_STEP;
			//ionTime = (i-1)*1e-6 + (j)* ION_TIME_STEP;
        	//xac = int(floor(2.0*FREQ*ionTime)) % 2;
			//traceFile << ionTime << ", " << xac << ", " << "\n";

			if(E_FIELD > 0){
				xac = 0;}
			else{
				xac = 1;}

			// inject ions on the boundary of the simulation
			if(GEOMETRY == 0) {
				// inject ions into the simulation sphere
				injectIonSphere_101 <<< blocksPerGridIon, DIM_BLOCK >>> (
					d_posIon.getDevPtr(),
					d_velIon.getDevPtr(),
					d_accIon.getDevPtr(),
					randStates.getDevPtr(),
					d_RAD_SPH.getDevPtr(),
					d_boundsIon.getDevPtr(),
					d_GCOM.getDevPtr(),
					d_QCOM.getDevPtr(),
					d_VCOM.getDevPtr(),
					d_NUM_DIV_QTH.getDevPtr(),
					d_NUM_DIV_VEL.getDevPtr(),
					d_SOUND_SPEED.getDevPtr(),
					d_TEMP_ION.getDevPtr(),
					d_PI.getDevPtr(),
					d_TEMP_ELC.getDevPtr(),
					d_MACH.getDevPtr(),
					d_MASS_SINGLE_ION.getDevPtr(),
					d_BOLTZMANN.getDevPtr(),
					d_CHARGE_ION.getDevPtr(),
					xac);
		
			roadBlock_104(statusFile, __LINE__, __FILE__, "injectIonSphere_101", print);
			} if(GEOMETRY == 1) {
				// fraction of plasma timestep
				if(TIME_EVOL>0 ) {
					counter_part = (j % N_IONDT_PER_PLASMADT)/N_IONDT_PER_PLASMADT;
				}
				else {
					counter_part = 0;
				}

				// inject ions into the simulation sphere
				injectIonCylinder_101 <<< blocksPerGridIon, DIM_BLOCK >>> (
					d_posIon.getDevPtr(), // -->
					d_velIon.getDevPtr(), // -->
					d_accIon.getDevPtr(), // -->
					randStates.getDevPtr(), 
					d_RAD_CYL.getDevPtr(),
					d_HT_CYL.getDevPtr(),
					d_boundsIon.getDevPtr(), // <--
					d_GCOM.getDevPtr(),
					d_QCOM.getDevPtr(),
					d_VCOM.getDevPtr(),
					d_NUM_DIV_QTH.getDevPtr(),
					d_NUM_DIV_VEL.getDevPtr(),
					d_SOUND_SPEED.getDevPtr(),
					d_TEMP_ION.getDevPtr(),
					d_PI.getDevPtr(),
					d_TEMP_ELC.getDevPtr(),
					d_MACH.getDevPtr(),
					d_MASS_SINGLE_ION.getDevPtr(),
					d_BOLTZMANN.getDevPtr(),
					d_CHARGE_ION.getDevPtr(),
					plasma_counter,counter_part,
					xac); // <--
		
			roadBlock_104(statusFile, __LINE__, __FILE__, "injectIonCylinder", print);
			}
	
			// Calculate the ion forces on the dust
			calcDustIonAcc_103 <<< blocksPerGridIon, DIM_BLOCK >>> (
				d_posIon.getDevPtr(), // {{{
				d_posDust.getDevPtr(), // <-->
				d_accDustIon.getDevPtr(), // <--
				d_NUM_DUST.getDevPtr(),
				d_NUM_ION.getDevPtr(),
				d_INV_DEBYE.getDevPtr(),
				d_DUST_ION_ACC_MULT.getDevPtr()); 
	
			roadBlock_104(statusFile, __LINE__, __FILE__, "calcDustIonAcc_103", print);

			//if (xac ==0) {
			//	E_direction = -1;
			//} else {
			//	E_direction = 1;
			//}
			//DEBUG -- for DC switch test, E field already flips
			E_direction = -1;

		//Calculate ion-ion, ion-outside ion, and E_FIELD accelerations
		calcIonAccels_102 <<<blocksPerGridIon, DIM_BLOCK,sizeof(float4) * DIM_BLOCK >>>(
			d_posIon.getDevPtr(), // <--
			d_accIon.getDevPtr(), // <-->
			d_NUM_ION.getDevPtr(), 
			d_SOFT_RAD_SQRD.getDevPtr(),
			d_ION_ION_ACC_MULT.getDevPtr(),
			d_INV_DEBYE.getDevPtr(),
			d_Q_DIV_M.getDevPtr(),
			d_HT_CYL.getDevPtr(),
			d_Vout.getDevPtr(),
			d_NUMR.getDevPtr(),
			d_RESZ.getDevPtr(),
			d_dz.getDevPtr(),
			d_dr.getDevPtr(),
			d_E_FIELD.getDevPtr(),
			d_E_FIELDR.getDevPtr(),
			E_direction,
			plasma_counter,
			GEOMETRY,
			d_EXTERN_ELC_MULT.getDevPtr());
	
		roadBlock_104(statusFile, __LINE__, __FILE__, "calcIonAccels_102", print);


			//Calculate ion-ion forces
			//Ions inside the simulation region
			// calculate the acceleration due to ion-ion interactions
			//calcIonIonAcc_102 
			//	<<< blocksPerGridIon, DIM_BLOCK, sizeof(float4) * DIM_BLOCK >>> (
			//	d_posIon.getDevPtr(), // {{{
			//	d_accIon.getDevPtr(), // <-->
			//	d_NUM_ION.getDevPtr(),
			//	d_SOFT_RAD_SQRD.getDevPtr(),
			//	d_ION_ION_ACC_MULT.getDevPtr(),
			//	d_INV_DEBYE.getDevPtr());
//	
			//roadBlock_104( statusFile, __LINE__, __FILE__, 
			//	"calcIonIonAcc_102 line 1906", print);
			// }}}	

			// copy ion accelerations to host
			//d_accIon.devToHost();
			// print the acc of specified ion to the trace file
			//traceFile << accIon[ionTraceIndex].x;
			//traceFile << ", " << accIon[ionTraceIndex].y;
			//traceFile << ", " << accIon[ionTraceIndex].z << std::endl;

			// Calculate the ion accelerations due to the ions outside of
			// the simulation cavity
			//if(GEOMETRY == 0) {
			//	// calculate the forces between all ions
			//	calcExtrnElcAcc_102 <<< blocksPerGridIon, DIM_BLOCK >>> (
			//		d_accIon.getDevPtr(), // {{{
			//		d_posIon.getDevPtr(), // <--
			//		d_EXTERN_ELC_MULT.getDevPtr(),
			//		d_INV_DEBYE.getDevPtr());
//	
//			roadBlock_104(statusFile, __LINE__, __FILE__, "calcExtrnElcAcc_102", print);
//			} else if(GEOMETRY == 1) {
			//	// calculate the forces between all ions outside
			//	//simulation region and external electric field
			//calcExtrnElcAccCyl_102 <<< blocksPerGridIon, DIM_BLOCK >>> (
			//	d_accIon.getDevPtr(), // {{{
			//	d_posIon.getDevPtr(), 
			//	d_Q_DIV_M.getDevPtr(),
			//	d_HT_CYL.getDevPtr(),
			//	d_Vout.getDevPtr(),
			//	d_NUMR.getDevPtr(),
			//	d_RESZ.getDevPtr(),
			//	d_dz.getDevPtr(),
			//	d_dr.getDevPtr(),
			//	d_E_FIELD.getDevPtr(),
			//	E_direction,
			//	plasma_counter);
//
//			roadBlock_104( statusFile, __LINE__, __FILE__, 
//				"calcExtrnElcAccCyl_102 line 1955", print);
//			}

		// Updates to ion velocity: collisions with ions and neutral //

		//Determine number of ions to collide
		randNum = (rand() % 100001)/100000.0;
		if (randNum < (N1 - N_COLL) ) n_coll = N_COLL+1; else n_coll = N_COLL;

		if (n_coll > NUM_ION/2) {
			set_value = 1;
			unset_value = 0;
			n_coll = NUM_ION - n_coll;
		} else {
			set_value = 0;
			unset_value = 1;
		}

		//reset collision list
		setCollisionList_105 <<< blocksPerGridIon, DIM_BLOCK >>> (
			d_collList.getDevPtr(), 
			set_value);
			
		roadBlock_104(  statusFile, __LINE__, __FILE__, "ionCollisionList_105", print);

		//copy collision list to host 
		d_collList.devToHost();

		// prepare list of ions to collide:
		// {{{
		for(int coll=0; coll < n_coll; coll++){
			collID[coll] = 0;
			do{
				dum  = (int)(rand() % NUM_ION);
				exist = false;
				for(int q=0;q<=coll-1;q++) if (collID[q]==dum) exist = true;
			} while(exist);
			collID[coll] = dum;
			collList[dum] = unset_value;
		}
		// }}}
		//debugFile << "number of ions to collide" << n_coll << std::endl;

		//copy collision list to device
		d_collList.hostToDev();
		roadBlock_104(  statusFile, __LINE__, __FILE__, "foo", print);					
		
		ionCollisions_105 <<< blocksPerGridIon, DIM_BLOCK >>> (
			d_collList.getDevPtr(),
			d_TEMP_GAS.getDevPtr(),
			d_MASS_SINGLE_ION.getDevPtr(),
			d_BOLTZMANN.getDevPtr(),
			d_I_CS_RANGES.getDevPtr(),
			d_TOT_ION_COLL_FREQ.getDevPtr(),
			d_SIGMA_I1.getDevPtr(),
			d_SIGMA_I2.getDevPtr(),
			d_SIGMA_I_TOT.getDevPtr(),
			d_velIon.getDevPtr(),
			randStates.getDevPtr(), 
			d_collision_counter.getDevPtr());

		bool print_test = false;
		roadBlock_104(statusFile, __LINE__, __FILE__, "ionCollisions_105", print_test);
		//if(print_test)
		//{ //copy ion velocities to the host
		//  d_velIon.devToHost();
		//	debugFile << "Failure in ionCollisions_105" << std::endl;
		//	for(int q = 1; q < NUM_ION; q++) {
		//		debugFile << velIon[q].x << ", " << velIon[q].y << ", " 
		//				<< velIon[q].z << std::endl;
		//	}
		//	exit(-1);
		//}

		// copy collision counter to the host 
		//d_collision_counter.devToHost();
		//debugFile << "Number ion collisions: " << collision_counter << "\n";

		//Any other external forces acting on ions would be calc'd here

			//Loop over ion  commands
			for(int c = 0; c < numCommands; c++){
				// copy ion positions to the host
				if (commands[c] == 1) {
					// {{{
					// print the command number to the status file
					statusFile << "1 ";
					statusFile.flush();
			
					// copy ion positions to host
					d_posIon.devToHost();

					// print the position of the specified ion to the trace file
					traceFile << posIon[ionTraceIndex].x;
					traceFile << ", " << posIon[ionTraceIndex].y;
					traceFile << ", " << posIon[ionTraceIndex].z;
					traceFile << ", " << posIon[ionTraceIndex].w << std::endl;
					// }}}

				// copy the ion velocities to the host
				} else if (commands[c] == 2) {
					// {{{
					statusFile << "2 ";

					// copy ion velocities to host
					d_velIon.devToHost();
	
					// print the velocity of the specified ion to the trace file
					traceFile << velIon[ionTraceIndex].x;
					traceFile << ", " << velIon[ionTraceIndex].y;
					traceFile << ", " << velIon[ionTraceIndex].z << std::endl;
					// }}}	

				// copy the ion accelerations to the host
				} else if (commands[c] == 3) {
					// {{{
					// print the command number to the status file
					statusFile << "3 ";
	
					// copy ion accelerations to host
					d_accIon.devToHost();
	
					// print the acceleration of the specified ion to the trace file
					traceFile << accIon[ionTraceIndex].x;
					traceFile << ", " << accIon[ionTraceIndex].y;
					traceFile << ", " << accIon[ionTraceIndex].z << std::endl;

				// Calculate New Dust Charge
				} else if (commands[c] == 4){
					// {{{
					// copy ion bounds to host
					d_boundsIon.devToHost();

					// copy dust positions and charge to host
					d_posDust.devToHost();

					// copy ion velocities to host
					d_velIon.devToHost();

					// calculate the ion currents to the dust particles
					// set initial currents to 0
					for (int k = 0; k < NUM_DUST; k++){
						ionCurrent[k] = 0.0;
					}

					// loop over all of the ion bounds
					for (int k = 0; k < NUM_ION; k++){
						// if the ion was collected by a dust particle
						if (boundsIon[k] > 0){
							// increase the current to that dust particle by 1
							ionCurrent[boundsIon[k] - 1] += posIon[k].w;
							// sum the ion momentum transfer to dust
							momIonDust[boundsIon[k] - 1].x += velIon[k].x;
							momIonDust[boundsIon[k] - 1].y += velIon[k].y;
							momIonDust[boundsIon[k] - 1].z += velIon[k].z;
					//debugFile << velIon[k].x << ", " << velIon[k].z << std::endl;
						}
					}

					// Update charge on dust
					for (int g = 0; g < NUM_DUST; g++) {
						// calculate the grain potential wrt plasma potential
						dustPotential =(COULOMB_CONST* posDust[g].w/ RAD_DUST); 
						//- ELC_TEMP_EV; 

						// calculate the electron current to the dust
						elcCurrent = ELC_CURRENT_0 * ION_TIME_STEP *
							exp((-1.0) * CHARGE_ELC * dustPotential /
							(BOLTZMANN * TEMP_ELC));
	
						// add current to dust charge
						posDust[g].w += elcCurrent+ionCurrent[g];

						//dustChargeFile << posDust[g].w << ", ";
						//save charge for averaging
						tempCharge[g] += posDust[g].w;
					}

					//dustChargeFile << "\n";

					// copy the dust charge (and position) to the GPU
					d_posDust.hostToDev(); 

				// Check For Erroneous Command
				} else if ( commands[c] != 5){
					// {{{
					// output an error message
					fprintf(stderr, "ERROR on line number %d in file %s\n",
						__LINE__, __FILE__);
					fprintf(stderr, "Command number %d of %d does not exist\n\n",
						commands[c], c);
	
					// terminate the program
					EXIT_WITH_FATAL_ERROR;
				}
			}

		// reset the ion bounds flag to 0
		//resetIonBounds_101 <<< blocksPerGridIon, DIM_BLOCK >>> (
		//	d_boundsIon.getDevPtr());
		//	
		//roadBlock_104(  statusFile, __LINE__, __FILE__, "resetIonBounds_101", print);
	

		// Kick for one timestep -- using just ion-ion accels
		//kick_100 <<< blocksPerGridIon, DIM_BLOCK >>> (
		//	d_velIon.getDevPtr(), // {{{
		//	d_accIon.getDevPtr(), // <-->
		//	d_ION_TIME_STEP.getDevPtr()); //lsm 1.23.18
		//	
		//roadBlock_104( statusFile, __LINE__, __FILE__, "kick_100", print);
	
		// Recalculate evolving parameters for time-dependent plasma conditions
		if(TIME_EVOL >0) {
			//advance values every Nth ion time step
			if( j % N_IONDT_PER_PLASMADT == 0) {
			// Update the plasma-counter and reset to zero if it has reached
			// the end of the values stored in the file
			plasma_counter = plasma_counter +1;
			if(plasma_counter == TIME_EVOL) { 
				plasma_counter = 0;
				flip_Ez = flip_Ez *(-1.0);
			}

			// copy variables to the host 
			d_INV_DEBYE.devToHost();
			d_E_FIELD.devToHost();
			d_E_FIELDR.devToHost();
			d_TEMP_ION.devToHost();
			d_CHARGE_ION.devToHost();
			d_DRIFT_VEL_ION.devToHost();
			d_SOUND_SPEED.devToHost();
			d_DEN_FAR_PLASMA.devToHost();
			d_RAD_COLL_MULT.devToHost();
			d_EXTERN_ELC_MULT.devToHost();
			d_TEMP_ELC.devToHost();
			d_MACH.devToHost();

			TEMP_ELC = evolTe[plasma_counter];
			TEMP_ION = evolTi[plasma_counter];
			DEN_FAR_PLASMA = evolne[plasma_counter];
			MACH = evolMach[plasma_counter];
			E_FIELD = evolEz[plasma_counter];
			E_FIELD = E_FIELD * flip_Ez;
			DRIFT_VEL_ION = evolVz[plasma_counter];
			E_FIELDR = evolEr[plasma_counter];

			DEBYE = sqrt((PERM_FREE_SPACE * BOLTZMANN * TEMP_ELC)/
				(DEN_FAR_PLASMA * CHARGE_ELC * CHARGE_ELC));
			INV_DEBYE = 1.0 / DEBYE;
			SUPER_ION_MULT = SIM_VOLUME * evolni[plasma_counter] / NUM_ION;
			CHARGE_ION = CHARGE_SINGLE_ION * SUPER_ION_MULT;
			MASS_ION = MASS_SINGLE_ION * SUPER_ION_MULT;
			SOUND_SPEED = sqrt(BOLTZMANN * TEMP_ELC / MASS_SINGLE_ION);
			vs_sq = 8 * BOLTZMANN * TEMP_ION / PI / MASS_SINGLE_ION 
				+ DRIFT_VEL_ION * DRIFT_VEL_ION;
			RAD_COLL_MULT = 
				2 * Q_DIV_M * COULOMB_CONST / RAD_DUST / vs_sq;
			ELC_CURRENT_0 = 4.0 * PI * RAD_DUST_SQRD * DEN_FAR_PLASMA *
				CHARGE_ELC * sqrt((BOLTZMANN * TEMP_ELC)/(2.0 * PI * ELC_MASS));
			EXTERN_ELC_MULT =
          		((RAD_SPH / DEBYE) + 1.0) * exp(-RAD_SPH / DEBYE) *
          		(CHARGE_SINGLE_ION * DEN_FAR_PLASMA * DEBYE) *
          		(Q_DIV_M) / (PERM_FREE_SPACE);

			//debugFile << plasma_counter << ", " << CHARGE_ION << std::endl;

			// copy updated variables to the device
			d_INV_DEBYE.hostToDev();
			d_E_FIELD.hostToDev();
			d_E_FIELDR.hostToDev();
			d_TEMP_ION.hostToDev();
			d_CHARGE_ION.hostToDev();
			d_DRIFT_VEL_ION.hostToDev();
			d_SOUND_SPEED.hostToDev();
			d_DEN_FAR_PLASMA.hostToDev();
			d_RAD_COLL_MULT.hostToDev();
			d_EXTERN_ELC_MULT.hostToDev();
			d_TEMP_ELC.hostToDev();
			d_MACH.hostToDev();

			}
		} //*** end if TIME_EVOL ***//

	} // ***** end of ion loop *****// 

					
	sumDustIonAcc_103<<<NUM_DUST, DIM_BLOCK, sizeof(float4)*DIM_BLOCK>>> (
		d_accDustIon.getDevPtr(),
		d_NUM_DUST.getDevPtr(),
		d_NUM_ION.getDevPtr()); 
					
	roadBlock_104(statusFile, __LINE__, __FILE__, "sumDustIonAcc_103", print);

	d_accDustIon.devToHost();
					    
		// ***** begin dust updates *****//
		// If dust particles have static positions 
		if(MOVE_DUST ==0) dust_time = ionTime;

		for (int c = 0; c < numCommands; c++){

			// Dust Charging
			if (commands[c] == 4) {

				// print all the dust charges to the trace file

				for (int k = 0; k < NUM_DUST; k++){
					//smooth the simulated dust charge over past timesteps 
					//simCharge[k] = tempCharge[k]/N_IONDT_PER_DUSTDT; 
					simCharge[k] = 0.95 * simCharge[k] 
						+ 0.05*tempCharge[k]/N_IONDT_PER_DUSTDT; 

					// print all the dust charges to the trace file
					dustChargeFile << tempCharge[k] << ", " << simCharge[k] << ", ";

					//reset the tempCharge to zero
					tempCharge[k] = 0;
				}

			dustChargeFile << std::endl;

			// Update Dust Position 
			} else if (commands[c] == 5) {
 
				// Print the command number to the status file 
				statusFile << "5 ";
					
				// copy the dust positions to the host
				d_posDust.devToHost();

				dust_time += dust_dt;
				dustTraceFile << dust_time << std::endl;

				// loop over dust particles 
				for (int j = 0; j < NUM_DUST; j++) {

					//kick half a  time step
					velDust[j].x += accDust[j].x * half_dust_dt;
					velDust[j].y += accDust[j].y * half_dust_dt;
					velDust[j].z += accDust[j].z * half_dust_dt;
				
					//Half of ion-dust momentum transfer (coll. ion drag)
					deltavee.x = momIonDust[j].x*mom_const;
					deltavee.y = momIonDust[j].y*mom_const;
					deltavee.z = momIonDust[j].z*mom_const;

					// Add deltavee due to ion collection drag force
					velDust[j].x += deltavee.x;
					velDust[j].y += deltavee.y;
					velDust[j].z += deltavee.z;

					// drift a whole step
					posDust[j].x += velDust[j].x * dust_dt;
					posDust[j].y += velDust[j].y * dust_dt;
					posDust[j].z += velDust[j].z * dust_dt;

					// periodic BC in z-dir for dust
					if(posDust[j].z > HT_CYL) {
						posDust[j].z -= 2.0*HT_CYL;
					} 
					if(posDust[j].z < -HT_CYL) {
						posDust[j].z += 2.0*HT_CYL;
					}

					//dustTraceFile << "j " << j << "\n";

					// zero the acceleration
					accDust[j].x = 0;
					accDust[j].y = 0;
					accDust[j].z = 0;

					// force of ions on dust
					accDust[j].x = accDustIon[j*NUM_ION].x/N_IONDT_PER_DUSTDT;
					accDust[j].y = accDustIon[j*NUM_ION].y/N_IONDT_PER_DUSTDT;
					accDust[j].z = accDustIon[j*NUM_ION].z/N_IONDT_PER_DUSTDT;

					//print this acceleration to the trace file
					//debugFile << "ion acceleration  ";
					//debugFile << accDust[j].x;
					//debugFile << ", " << accDust[j].y;
					//debugFile << ", " << accDust[j].z;
					//debugFile << ", " << deltavee.x;
					//debugFile << ", " << deltavee.y;
					//debugFile << ", " << deltavee.z << "\n";


					// Calculate dust-dust acceleration 
					if(j == 0) {
						for (int g = 0;  g < NUM_DUST; g++) {
							accDust2[g].x = 0;
							accDust2[g].y = 0;
							accDust2[g].z = 0;
						}
					}

					// forces between the dust grains
					for(int g = j+1; g < NUM_DUST; g++) {
					// calculate the distance between dust grain j
						//and all other grains
						distdd.x = posDust[j].x - posDust[g].x;
						distdd.y = posDust[j].y - posDust[g].y;
						distdd.z = posDust[j].z - posDust[g].z;
			
						distSquared = distdd.x*distdd.x+distdd.y*distdd.y 
							+ distdd.z*distdd.z;
			
						// calculate the hard distance
						dist = sqrt(distSquared);
			
						//calculate a scalar intermediate
						//linForce=DUST_DUST_ACC_MULT*(simCharge[j]) 
						//	* (simCharge[g]) / (dist*dist*dist)
						//	* (1.0+dist/DEBYE) * exp(-dist/DEBYE);
						linForce=DUST_DUST_ACC_MULT*(simCharge[j]) 
							* (simCharge[g]) / (dist*dist*dist);
			
						// add the acceleration to the current dust grain
						accDust[j].x += linForce * distdd.x;
						accDust[j].y += linForce * distdd.y;
						accDust[j].z += linForce * distdd.z;

						// add -acceleration to other dust grain
						accDust2[g].x -= linForce * distdd.x;
						accDust2[g].y -= linForce * distdd.y;
						accDust2[g].z -= linForce * distdd.z;     
					}

					accDust[j].x +=  accDust2[j].x;
					accDust[j].y +=  accDust2[j].y;
					accDust[j].z +=  accDust2[j].z;
						
					// calculate acceleration of the dust
					//Radial position of dust
					rhoDustsq = posDust[j].x * posDust[j].x +
								posDust[j].y * posDust[j].y;

					// radial acceleration from confinement
					///*** linear in r ***///
					acc = OMEGA_DIV_M * simCharge[j];
					///*** cubic in r ***///
					//acc = (OMEGA_DIV_M + OMEGA2_DIV_M * rhoDustsq) * simCharge[j];

					accDust[j].x += acc * posDust[j].x;
					accDust[j].y += acc * posDust[j].y;

				//debugFile << "radial acceleration ";
				//debugFile << acc*posDust[j].x << ", ";
				//debugFile << acc*posDust[j].y << ", " << '\n';

					// Big accel to keep dust from leaving sides of cylinder
					rhoDust = sqrt(rhoDustsq);
					if(rhoDust > radialConfine) {
					acc = OMEGA2_DIV_M * simCharge[j]
							*(rhoDust-radialConfine) / rhoDust;
					accDust[j].x += acc * posDust[j].x;
					accDust[j].y += acc * posDust[j].y;
					}
				
				//debugFile << "confinement acceleration ";
				//debugFile << acc*posDust[j].x << ", ";
				//debugFile << acc*posDust[j].y << ", " << '\n';

					//axial confinement in z for dust near ends of cylinder	
					if(abs(posDust[j].z) > axialConfine) {
						if(posDust[j].z > 0) {
							adj_z = posDust[j].z - axialConfine;
						} else {
							adj_z = posDust[j].z + axialConfine;
						}	
						//accDust[j].z += OMEGA_DIV_M* simCharge[j] * adj_z;
						accDust[j].z += OMEGA2_DIV_M* simCharge[j] * adj_z;
					}
					
					//polarity switching
					q_div_m = (simCharge[j]) / MASS_DUST;
					accDust[j].z -= q_div_m * E_FIELD 
						*(4.0*floor(FREQ*dust_time)-2.0*floor(2.0*FREQ*dust_time)+1.);
			
					if(USE_GRAV == 1){
						// force due to gravity
						accDust[j].z -= 9.81;
		
						// forces for sheath above lower electrode
						// -- adjust pos.z for ht above lower electr.
						ht = posDust[j].z + BOX_CENTER;
						ht2 = ht*ht;
						acc = -8083 + 553373*ht + 2.0e8*ht2 -
							3.017e10*ht*ht2 + 1.471e12*ht2*ht2 - 2.306e13*ht*ht2*ht2;
						// Multiple by E_MULT to change steepness as power changes
						accDust[j].z += q_div_m * acc * E_MULT;
						//accDust[j].z += q_div_m * acc;

						// laser push on lower particle
						if(LASER_PUSH == 1) {
							if(dust_time > LASER_ON &&dust_time < LASER_OFF && j==1) {
								accDust[j].x -= 0.5;
							}
						}

						//dustTraceFile << "sheath E acceleration  ";
						//dustTraceFile << q_div_m <<", "<< ht << ", " << acc << ", ";
						//debugSpecificFile << q_div_m * acc << std::endl;
					}

					// forces from ions outside simulation region
					//rad = sqrt(posDust[j].x * posDust[j].x +
					//		posDust[j].y * posDust[j].y);
					//zsq = posDust[j].z * posDust[j].z;
					//radAcc = P10X + P12X * zsq + P14X * zsq * zsq;
					//vertAcc = P01Z * posDust[j].z +
				//			P21Z * rad * rad * posDust[j].z +
				//			P03Z * posDust[j].z * zsq +
				//			P23Z * rad * rad * posDust[j].z * zsq +
				//			P05Z * posDust[j].z * zsq * zsq;
				//	accDust[j].x += posDust[j].x * radAcc * q_div_m;
				//	accDust[j].y += posDust[j].y * radAcc * q_div_m;
				//	accDust[j].z += vertAcc * q_div_m;
		
					//debugFile << "outside ion acceleration  ";
					//debugFile << posDust[j].x*radAcc*q_div_m;
					//debugFile << ", " << posDust[j].y*radAcc*q_div_m;
					//debugFile << ", " << vertAcc*q_div_m << "\n";

					// drag force
					accDust[j].x -= BETA*velDust[j].x;
					accDust[j].y -= BETA*velDust[j].y;
					accDust[j].z -= BETA*velDust[j].z;

					// Add Brownian motion
					//randNum = (((rand() % (num*2)) - num) / (float)num);
					randNum = normRand_106();
					accDust[j].x += randNum * SIGMA;
					//randNum = (((rand() % (num*2)) - num) / (float)num);
					randNum = normRand_106();
					accDust[j].y += randNum * SIGMA;
					//randNum = (((rand() % (num*2)) - num) / (float)num);
					randNum = normRand_106();
					accDust[j].z += randNum * SIGMA;
							
					//kick half a  time step
					velDust[j].x += accDust[j].x * half_dust_dt;
					velDust[j].y += accDust[j].y * half_dust_dt;
					velDust[j].z += accDust[j].z * half_dust_dt;

					// Add deltavee due to ion collection drag force
					velDust[j].x += deltavee.x;
					velDust[j].y += deltavee.y;
					velDust[j].z += deltavee.z;

					// print the dust position to the dustPosTrace file
					//dustTraceFile << "After the dust timestep" << std::endl;
					dustTraceFile << posDust[j].x;
					dustTraceFile << ", " << posDust[j].y;
					dustTraceFile << ", " << posDust[j].z;
					dustTraceFile << ", " << velDust[j].x;
					dustTraceFile << ", " << velDust[j].y;
					dustTraceFile << ", " << velDust[j].z;
					dustTraceFile << ", " << accDust[j].x;
					dustTraceFile << ", " << accDust[j].y;
					dustTraceFile << ", " << accDust[j].z << std::endl;

				} // End of dust timestep
				
				//debugFile << std::endl;

				// copy the dust position to the GPU
				d_posDust.hostToDev();
			}
		} //end of loop through commands

		// Print and Zero ionDensity
		if (i % N  == 0) { //N will need to be related to frequency
			// copy ion density and potential to host
			d_ionDensity.devToHost();
			d_ionPotential.devToHost();

			roadBlock_104(  statusFile, __LINE__, __FILE__, 
				"Copy d_ionDensity and d_ionPotential to Host", print);
			// print the data to the ionDensOutFile
			//Average the accumulated values over previous time steps by dividing
			// by N and N_IONDT_PER_DUSTDT.  Remember that density was calculated
			// by summing the charge on superion, so must divide by
			// the charge on a single ion to get total number of ions.
			for(int j = 0; j < NUM_GRID_PTS; j++){
				ionDensOutFile << ionDensity[j]/N/N_IONDT_PER_DUSTDT/CHARGE_SINGLE_ION;
				ionDensOutFile << ", " << ionPotential[j]/N/N_IONDT_PER_DUSTDT;
				ionDensOutFile << std::endl;
			}
			ionDensOutFile << std::endl;

			//reset the potential and density to zero
			zeroIonDensityPotential_102 <<<blocksPerGridGrid, DIM_BLOCK >>> (
				d_ionPotential.getDevPtr(),
       	        d_ionDensity.getDevPtr()
			);

			roadBlock_104(  statusFile, __LINE__, __FILE__, "zeroIonDensityPotential_102", print);

		}

		// ****** Print the Ion Forces on the Dust ****** //
		// These are forces in units of Newtons
		// reminder: mom_const = m_i/m_d*dust_dt/(2*N_ION_DT_PER_DUST_DT*ION_TIME_STEP)
		
		for( int j=0 ; j<NUM_DUST ; j++ ) {		
			ionOnDustAccFile << 
					   accDustIon[j*NUM_ION].x/N_IONDT_PER_DUSTDT*MASS_DUST << ", "
					<< accDustIon[j*NUM_ION].y/N_IONDT_PER_DUSTDT*MASS_DUST << ", "
					<< accDustIon[j*NUM_ION].z/N_IONDT_PER_DUSTDT*MASS_DUST << ", "
					<< momIonDust[j].x*mom_const*MASS_DUST/dust_dt*2 << ", "
					<< momIonDust[j].y*mom_const*MASS_DUST/dust_dt*2 << ", "
					<< momIonDust[j].z*mom_const*MASS_DUST/dust_dt*2 << ", " << std::endl;
			// print just the velocity transferred
					//<< momIonDust[j].x << ", "
					//<< momIonDust[j].y << ", "
					//<< momIonDust[j].z << ", " << std::endl;
			momIonDust[j].x = 0;
			momIonDust[j].y = 0;
			momIonDust[j].z = 0;
		}

		// copy back to the device
		d_accIonDust.hostToDev();
		roadBlock_104(statusFile, __LINE__, __FILE__,
			"Copy d_accDustIon to device ", print);
		// zero the ionDustAcc
		zeroDustIonAcc_103<<<blocksPerGridIon, DIM_BLOCK >>> (
			d_accDustIon.getDevPtr(),
			d_NUM_DUST.getDevPtr(),
			d_NUM_ION.getDevPtr());

		roadBlock_104(  statusFile, __LINE__, __FILE__, "zero DustIonAcc", print);		


		statusFile << "|" << std::endl;
		
		if (i % 10  == 0) { //print every 10 dust time steps
			// ****** Update Continue Files ****** //
			// Print ion and dust data needed for using the continue 
			// option. The files are overwritten each time step.
	
			// open an output file for saving final dust data  
			fileName = dataDirName + runName + "_dust-final.txt";
			std::ofstream dustFinalFile(fileName.c_str());
	
			// open an output file for final ion data
			fileName = dataDirName + runName + "_ion-final.txt";
			std::ofstream ionFinalFile(fileName.c_str());
	
			// print out final dust data
			dustFinalFile<<"    rX      rY      rZ      vX      vY      vZ      Q\n";
	
			for( int j=0 ; j<NUM_DUST ; j++ ){
				dustFinalFile << "[" << j << "]   ";
				dustFinalFile << posDust[j].x << " ";
				dustFinalFile << posDust[j].y << " ";
				dustFinalFile << posDust[j].z << " ";
				dustFinalFile << velDust[j].x << " ";
				dustFinalFile << velDust[j].y << " ";
				dustFinalFile << velDust[j].z << " ";
				dustFinalFile << simCharge[j] << std::endl;
			}
	
			// print out final ion data
			for( int j=0 ; j<NUM_ION ; j++ ) {
				ionFinalFile << posIon[j].x << " ";
				ionFinalFile << posIon[j].y << " ";
				ionFinalFile << posIon[j].z << " ";
				ionFinalFile << posIon[j].w << " ";
				ionFinalFile << velIon[j].x << " ";
				ionFinalFile << velIon[j].y << " ";
				ionFinalFile << velIon[j].z << std::endl;
			}
	
			dustFinalFile.close();
			ionFinalFile.close();
		} // if for Update Continue Files
		
	} // ***** end time step loop **** //

				
	/************************
		Post-Processes
	***********************/
	
	/****** Save Data ******/

	// copy ion positions to host
	d_posIon.devToHost();

	// copy ion velocities to host
	d_velIon.devToHost();

	// synchronize threads and check for errors
	roadBlock_104( statusFile, __LINE__, __FILE__, "devToHost", print);
	if (debugMode) {
		// print the index of the traced ion to the debugging file
		debugFile << "Single ion trace index: " << ionTraceIndex << "\n\n";
	}				 
	//Checking Dust charge
	debugFile << "**********DUST CHARGE**********" << std::endl;
	for (int g = 0; g < NUM_DUST ; g++){
		debugFile << "DUST CHARGE: " << g << ": " << simCharge[g] << std::endl;	
	}

	// print final ion positions to the ionPosFile
	// loop over all of the positions
	for (int i = 0; i < NUM_ION; i++) {
		// print the ion position
		ionPosFile << posIon[i].x;
		ionPosFile << ", " << posIon[i].y;
		ionPosFile << ", " << posIon[i].z;
		ionPosFile << ", " << posIon[i].w << std::endl;
	}

	// print final ion velocities to the ionVelFile
	// loop over all of the positions
	//for (int i = 0; i < NUM_ION; i++) {
		// print the ion position
	//	ionVelFile << velIon[i].x;
	//	ionVelFile << ", " << velIon[i].y;
	//	ionVelFile << ", " << velIon[i].z << std::endl;
	//}

	// print the final dust charges to the dustChargeFile
	// loop over all of the dust particles
	for (int i = 0; i < NUM_DUST; i++) {
		// print the dust charge
		dustChargeFile << simCharge[i] << ", ";
	}
	dustChargeFile << std::endl;

	// print the final dust positions to the dustPosFile
	// loop over all of the dust particles
	for (int i = 0; i < NUM_DUST; i++) {
		// print the dust positions
		dustPosFile << posDust[i].x << ", ";
		dustPosFile << posDust[i].y << ", ";
		dustPosFile << posDust[i].z << std::endl;
	}

	if (debugMode) {

		debugFile << "-- Final Ion Positions: First 20 Ions --" << std::endl;
		for (int i = 0; i < 20; i++) {
			debugFile << "#: "  << i
			<< " X: " << posIon[i].x
			<< " Y: " << posIon[i].y
			<< " Z: " << posIon[i].z
			<< std::endl;
		}

		debugFile << '\n' << "-- Final Ion Positions: Last 20 Ions --" << '\n';
		for (int i = 1; i <= 20; i++) {
			int ID = NUM_ION - i;
			debugFile << "#: "  << ID
			<< " X: " << posIon[ID].x
			<< " Y: " << posIon[ID].y
			<< " Z: " << posIon[ID].z
			<< std::endl;
		}

		debugFile << '\n'

		<< "-- Final Ion Velocities: First 20 Ions --" << '\n';
		for (int i = 0; i < 20; i++) {
			debugFile << "#: " << i
			<< " X: " << velIon[i].x
			<< " Y: " << velIon[i].y
			<< " Z: " << velIon[i].z
			<< std::endl;
		}

		debugFile << '\n'

		<< "-- Final Ion Velocities: Last 20 Ions --" << '\n';
		for (int i = 1; i <= 20; i++) {
			int ID = NUM_ION - i;
			debugFile << "#: " << ID
			<< " X: " << velIon[ID].x
			<< " Y: " << velIon[ID].y
			<< " Z: " << velIon[ID].z
			<< std::endl;
		}

		debugFile << '\n'

		<< "-- Final Ion Accelerations: First 20 Ions --" << '\n';
		for (int i = 0; i < 20; i++) {
			debugFile << "#: " << i
			<< " X: " << accIon[i].x
			<< " Y: " << accIon[i].y
			<< " Z: " << accIon[i].z
			<< std::endl;
		}

		debugFile << '\n'

		<< "-- Final Ion Accelerations: Last 20 Ions --" << '\n';
		for (int i = 1; i <= 20; i++) {
			int ID = NUM_ION - i;
			debugFile << "#: " << ID
			<< " X: " << accIon[ID].x
			<< " Y: " << accIon[ID].y
			<< " Z: " << accIon[ID].z
			<< std::endl;
		}

		debugFile << std::endl;
	}

	/****** Check Device "Constants" ******/
	// {{{
	
	d_COULOMB_CONST.compare();
	d_ION_ION_ACC_MULT.compare();
	d_NUM_DIV_QTH.compare();
	d_NUM_DIV_VEL.compare();
	d_NUM_ION.compare();
	d_NUM_DUST.compare();
	d_RAD_DUST.compare();
	d_RAD_DUST_SQRD.compare();
	d_SOFT_RAD_SQRD.compare();
	d_RAD_SPH.compare();
	d_M_FACTOR.compare();
	d_RAD_SPH_SQRD.compare();
	d_RAD_CYL.compare();
	d_RAD_CYL_SQRD.compare();
	d_HT_CYL.compare();
	d_HALF_TIME_STEP.compare();
	d_RAD_COLL_MULT.compare();
	d_PI.compare();
	d_MASS_SINGLE_ION.compare();
	d_BOLTZMANN.compare();
	d_Q_DIV_M.compare();
	d_MAX_DEPTH.compare();
	d_I_CS_RANGES.compare();
	d_TOT_ION_COLL_FREQ.compare();
	d_ION_DUST_ACC_MULT.compare();
	d_DUST_ION_ACC_MULT.compare();

	/****** Free Memory ******/

	free(posDust);
	free(velDust);
	free(accDust2);
	free(tempCharge);
	free(simCharge);
	free(commands);
	free(posIon);
	free(velIon);
	free(accIon);
	free(boundsIon);
	free(m);
	free(timeStepFactor);
	free(minDistDust);
	free(gridPos);
	free(ionPotential);
	free(ionDensity);
	delete[] ionCurrent;

	/****** Close Files ******/

	// close all opened files
	paramFile.close();
	dustParamFile.close();
	timestepFile.close();
	debugFile.close();
	debugSpecificFile.close();
	traceFile.close();
	statusFile.close();
	ionPosFile.close();
	//ionVelFile.close();
	dustPosFile.close();
	dustTraceFile.close();
	dustChargeFile.close();
	paramOutFile.close();
	ionDensOutFile.close();

	return 0;

}
