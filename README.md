# IonWake
Scope:

Running:
  To create an executable for KODIAK, the source files must be compiles in Linux.  To compile the source files on KODIAK load 
  the cuda80/toolkit/8.0.44 module.  Place all the source code files and the make file into the same directory and use the 
  “make” command. This will generate an executable with the name IonWake_000.

  To run the executable on KODIAK, a job needs to submitted to a GPU node by using -q gpu when creating a job.  Place the input 
  files in the same directory as the executable.  

  The program output is determined entirely by the output options.

Options:
  The program output is currently placed in a directory that is hard coded into the program. A future improvement will allow for 
  the use to specify where program output appears.

- Status:
	  Currently cannot be turned off.
	  Prints to “statusFile.txt”.
	  Outputs the current location of the program within the time step.
	  Each time step is given its own line.
	  “|” markes the beginning and end of a time-step.
	  After each process within each time step a alphabetically increasing letter is printed.

Each of the following output types can be toggled on and off within the specified file.

- DebugMode:
	  On/Off switch is in “paramListDebug.txt”.
	  When off all debugging output is stopped.
	  When on, only debugging output which is on is displayed. 

- showParameters  
	  On/Off switch is in “paramListDebug.txt”.  	
	  Is a part of debugging output.
    Saves all the input parameters
	  Prints to “debugFile.txt”
 
 - showConstants        
    On/Off switch is in “paramListDebug.txt”.
    Is a part of debugging output.
    Saves all the program constants
    Prints to “debugFile.txt”


  - showOutputParameters 
    On/Off switch is in “paramListDebug.txt”.
    Is a part of debugging output.
    Saves all the output parameters
    Prints to “debugFile.txt”

  - showInitHostVars     
    On/Off switch is in “paramListDebug.txt”.
    Is a part of debugging output.
    Saves the initial values of the initial values for
	  Host memory sizes
	  First and last 20 ion positions
    Prints to “debugFile.txt”

  - showFinalHostVars    
    On/Off switch is in “paramListDebug.txt”.
    Is a part of debugging output.
    Saves the values of variables after the time step
	  First and last 20 ion positions
	  First and last 20 ion velocities 
	  First and last 20 ion accelerations 
    Prints to “debugFile.txt”

  - singleIonTraceMode   
    On/Off switch is in “paramListDebug.txt”.
    Is a part of debugging output.
    Saves all of the positions of a single ion throughout all of the time-steps 
    Prints to “ionPosTrace.txt.txt”

  - IonTraceIndex
    On/Off switch is in “paramListDebug.txt”.
    Is a part of debugging output.
    Controls which ion singleIonTraceMode tracks 
    Indexed at 0 and may not exceed the number of ions

Each of the following program parameter values can be specified in paramList.txt

  - NUM_ION:
    Number of ions in simulation must be a whole number

  - DEN_FAR_PLASMA:   
    Unperturbed plasma number density far from the dust particle (m^-3)
  
  - TEMP_ELC:
    Electron temperature (K)

  - TEMP_ION:
    Ion temperature (K) 

  - DEN_DUST:         
    Dust particle density (Kg/m^3)

  - MASS_ION:            
	  Ion mass (Kg)

  - MACH:         
    Ion flow speed in the sheath is a multiple of the ion sound speed

  - SOFT_RAD:              
    Softening radius (m).  Used to soften interparticle forces at short distances

  - RAD_DUST:             
	  Dust particle radius (m)

  - CHARGE_DUST:        
  	Dust particle charge (e). particle charge in Q = chargeElc * e

  - CHARGE_ION:     
	  Ion charge (e). Ion charge in Q = chargeIon * e

  - TIME_STEP: 
    Time step length (s)

  - NUM_TIME_STEP:
    Number of time steps. total simulation time = numTimeStep * timeStep

  - RAD_SIM_DEBYE:
	  Radius of the spherical simulation volume as a multiple of the debye length. 
    radius in m = radSimdebye * debye length
