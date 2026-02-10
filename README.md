# DRIAD_public

Public mirror repository for DRIAD, a GPU-accelerated dusty plasma simulation
code. This repository contains the simulation source code together with scripts
and input templates required to compile and run simulations on local machines
and HPC systems.

The intended workflow is to compile and run the code exclusively through the
Makefile.

DRIAD is a molecular dynamics simulation designed to self-consistently resolve
the coupled dynamics of streaming ions and charged dust grains in dusty plasmas.
The model simultaneously advances ion and dust motion on their respective time
scales while allowing the dust charge to evolve dynamically in response to the
local ion environment.

Ions are treated using a super-ion molecular dynamics approach, where ion–ion
interactions are described by a Yukawa potential to account for electron
shielding, while ion–dust interactions are treated using the Coulomb potential
of the dust grains. Electrons are not modeled explicitly, instead, they are
treated as a Boltzmann fluid that provides background shielding and contributes
to dust charging through an orbital-motion-limited electron current.

The simulation employs an asymmetric time-stepping scheme in which ions are
advanced over many small ion time steps, after which dust positions and charges
are updated using averaged ion properties. This approach allows the formation
and evolution of ion wakefields to be resolved self-consistently while
maintaining computational efficiency.

Dust grains are free to move under the influence of interparticle forces, ion
drag, external electric fields, gravity, and neutral gas damping. Dust charging
is calculated from the balance of collected ion and electron currents, allowing
the grain charge to vary as the particle moves through regions of enhanced ion
density within the wakefield.

# Repository structure

IonWake/
- Core DRIAD simulation code and CMakeLists.txt
- Makefile (all build/run commands are issued from here)

DRIAD_run_scripts/
- Helper scripts invoked by the Makefile
- base_input/ templates used to generate per-run inputs

analysis/
- Incomplete MATLAB analysis scripts (work in progress)

# Execution model

Simulations are executed through the Makefile located in the IonWake directory.
The Makefile internally calls scripts located in DRIAD_run_scripts.

Two execution modes are supported:
- Local execution on a standalone machine
- Batch execution on HPC systems using qsub-based schedulers (PBS/Torque-like)

All commands must be run from a terminal opened in the directory that contains
the Makefile.

# Input files (base_input)

Base input templates are located in: DRIAD_run_scripts/base_input/

For each run, these files are copied into run-specific input directories.

_dust_params.txt
- Initial dust conditions: positions, velocities, and charges (all initial values).

_params.txt
- General simulation and plasma parameters. This includes
  (non-exhaustive examples): pressure, temperature, density, sheath
  electric-field, magnetic field, optional gravity, simulation control
  parameters, and dust properties such as radius
  and material-related parameters.

_plasma_params.txt (optional)
- Parameters for time-evolving plasma conditions.

_timestep.txt
- Switches for enabling/disabling timestep features and physics
  options.

# Timestep options

The following options can be enabled in _timestep.txt:

TR-ion-pos 
- Prints a trace of an ion position to the trace output file.

TR-ion-vel
- Prints a trace of an ion velocity to the trace output file.

TR-ion-acc
- Prints a trace of an ion acceleration to the trace output file.

CH-charge-dust
- Enables dynamic calculation of dust charge.

CH-move-dust
- Enables dynamic calculation of dust position, velocity, and acceleration.

# Compilation (CMake)

Compilation is handled through CMake and invoked from the Makefile.

CUDA compiler path in CMakeLists.txt:
- The CMakeLists.txt may contain an explicit CUDA compiler path
  (CMAKE_CUDA_COMPILER).
- That path is system-specific (it can point to a particular CUDA installation
  on a given laptop/workstation or on an HPC system).
- If nvcc is already available in the PATH, or if a different CUDA version is
  desired, edit that path accordingly or comment it out and let CMake resolve
  nvcc from the PATH.

Adding files:
- To add new source files (.cu or .cpp), list them in the appropriate sections
  of CMakeLists.txt (CUDA_FILES or CPP_FILES).

Template-related note (important):
- If templates are used in a file, do not list that file in CMakeLists.txt;
  otherwise duplicate definition errors can occur.
- If you split declarations (.h/.hpp) from definitions (.cu/.cpp), include the
  .cu/.cpp directly in the main translation unit instead of only including the
  header.

# Local GPU queue system (task-spooler)

Local queued execution uses task-spooler:
https://github.com/justanhduc/task-spooler#manual

This workflow assumes:
- One independent task-spooler queue per GPU.
- These GPU-specific queues are initialized when the system boots (startup
  configuration).
- Shell aliases or functions are defined globally so short commands such as
  tsgpu0 and tsgpu1 expand to longer task-spooler invocations bound to a
  specific GPU queue.

Queue commands used on local PCs:

tsgpu0 or tsgpu1 
- Returns the queue manager status for GPU 0 or GPU 1.

tsgpu0 -K or tsgpu1 -K
- Deletes pending code executions in the corresponding GPU queue.

tsgpu0 -k PID or tsgpu1 -k PID
- Deletes a specific job in the corresponding GPU queue (PID is the job/process id).

tsgpu0 -C or tsgpu1 -C
- Cleans finished code executions from the corresponding GPU queue.

# Makefile usage

All build, run, and job-management operations are performed through the
Makefile. Open a terminal in IonWake/ (the directory that contains the Makefile) and 
write make with one of the next targets (e.g. make build_release).

Targets:

build_debug
- Builds the code in debug mode (slower, more diagnostics).

build_release
- Builds the code in release mode (optimized).

clean
- Removes the build directory and build artifacts.

run_job
- Prompts for a run name, prepares run directories from base_input, and submits
  the job to an HPC scheduler using qsub-based submission.

run_local
- Prompts for a run name, prepares run directories from base_input, and runs the
  code immediately on the local machine.

run_local_gpu0
- Prompts for a run name, prepares run directories from base_input, and submits
  the local run to the task-spooler queue associated with GPU 0 (via tsgpu0).

run_local_gpu1
- Prompts for a run name, prepares run directories from base_input, and submits
  the local run to the task-spooler queue associated with GPU 1 (via tsgpu1).

status
- Prompts for a run name and prints status information for that run.

info
- Prompts for a run name and prints output/error information for that run.

kill_job
- Prompts for a run name and cancels the corresponding HPC batch job.

kill_local
- Prompts for a run name and terminates the corresponding local run.

# Script description

The following scripts are located in DRIAD_run_scripts and are executed
internally by the Makefile. Users normally do not need to run these scripts
directly.

compile.sh
- Compiles the DRIAD executable and prepares it for execution.

getPaths.sh
- Resolves and exports paths required by the runtime environment.

setup_run.sh
- Creates run-specific input and output directories and copies base input
  templates.

runJob.sh
- Submits a simulation job to an HPC batch scheduler (qsub-based).

runLocal.sh
- Executes a simulation locally.

check.sh
- Displays the most recent status information for a run.

disp.sh
- Displays output and error files for a run.

Internal helpers (used by other scripts): 
qsub_submit_script.sh
qsub_run_script.sh

Job deletion helpers: 
deleteJob.sh 
deleteLocal.sh
