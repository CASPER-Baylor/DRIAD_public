# DRIAD_run_scripts
Scripts for running DRIAD on kodiak at Baylor using the RAPID framework
. A quick how to for the scripts: 

SETUP
•	Clone the repository (into your kodiak data directory)
•	In the root DRIAD_run_scripts directory (data/matthewsl/.../DRIAD_run_scripts) create two files run_dir.txt and source_dir.txt.

USING:

To compile the version of IonWake (DRIAD) located at the specified source directory and copy the executable to the run directory: 
         $ compile.sh

The rest of the scripts take a run name as an argument. Example: 
          $ ./run my_run_name

setup_run.sh: creates appropriately named input and output directories (run_name_input and run_name_output) then copies the input files in the base_input directory into the run's input directory. This only needs to be run once per run name.

run.sh: submits a job to the queue using the executable generated with compile.sh and the appropriate input/output directories for the specified run name. All output, both IonWake generated and queue generated (i.e run_name.o###### and run_name.e#####) files, are placed in the run_name_output directory.  You won’t need this – as you are not submitting jobs to the queue on Kodiak.

local_run.sh: same as run.sh, except that IonWake is run on the current machine (say login000 or my laptop) instead of being submitted to the queue.

check.sh: will display the end of the status file of the specified run name.

disp.sh: will display the run_name.e###### and run_name.o###### files for the specified run name.

follow.sh: will display and follow (update every 1/4 second) the status file of the specified run name.

INTERNALS
qsub_submit_script.sh and qsub_run_script.sh are used by other scripts.

ANALYSIS 
The analysis directory contains the incomplete matlab scripts I mentioned earlier.

EXAMPLE
# Create a run called low_wake_perp_2
./setup_run.sh low_wake_perp_2

# edit input files
vim ./low_wake_perp_2_input/params.txt
vim ./low_wake_perp_2_input/dust-params.txt
vim ./low_wake_perp_2_input/timestep.txt

# compile latest IonWake version
./compile.sh

# submit it to the queue
./run.sh low_wake_perp_2

# Assuming the job is now running, check for any initial errors 
./disp.sh low_wake_perp_2

# Follow the status file until completion
./follow.sh low_wake_perp_2

# Check for any final errors
./disp.sh low_wake_perp_2

All IonWake output files would then be located in low_wake_perp_2_output/
