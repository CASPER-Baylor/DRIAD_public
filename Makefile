# variable to store the build directory
buildDirectory:=build
# variable to store the source directory
sourceDirectory:=source

# function to build the code in debug mode
build_debug:
    # create the build directory if it doesn't exist
	@mkdir -p ${buildDirectory}

	# Execute the CMake
	@cmake -B ${buildDirectory} -DCMAKE_BUILD_TYPE=Debug ./${sourceDirectory}

    # compile the code
	@make -C ${buildDirectory} && echo "------Compilation done------"

	# setup to run the code
	@cd ../DRIAD_run_scripts && \
	./getPaths.sh && \
	./compile.sh && \
	echo "------Compilation done------"

# function to build the code in release mode
build_release:
    # create the build directory if it doesn't exist
	@mkdir -p ${buildDirectory}

	# Execute the CMake
	@cmake -B ${buildDirectory} -DCMAKE_CSS_FLAGS="-O3" -DCMAKE_BUILD_TYPE=Release ./${sourceDirectory}

    # compile the code
	@make -C ${buildDirectory}

	# setup to run the code
	@cd ../DRIAD_run_scripts && \
	./getPaths.sh && \
	./compile.sh && \
	echo "------Compilation done------"


# function to remove the build directory
clean:
    # remove the build directory if it exists
	@rm -rf ${buildDirectory} && echo "------Build directory removed------"

# function to run the code into a job
run_job:
	# set job name, copy binary file and run the binary file
	@read -p "Name of Job: " name &&  \
	cd ../DRIAD_run_scripts && \
	./setup_run.sh $$name && \
	./runJob.sh $$name &&\
	echo "------Code running------"

# function to run the code into a job
run_local:
	# set job name, copy binary file and run the binary file
	@read -p "Name of Job: " name &&  \
	cd ../DRIAD_run_scripts && \
	./setup_run.sh $$name && \
	./runLocal.sh $$name &&\
	echo "------Code running------"
	
# check the status of a job
status:
	@read -p "Name of Job: " name &&  \
	cd ../DRIAD_run_scripts && ./check.sh $$name
	
# get errors and extra informations of the job
info:
	@read -p "Name of Job: " name &&  \
	cd ../DRIAD_run_scripts && ./disp.sh $$name

kill_job:
	@read -p "Name of Job: " name && \
	cd ../DRIAD_run_scripts && ./deleteJob.sh $$name		

kill_local:
	@read -p "Name of Job: " name && \
	cd ../DRIAD_run_scripts && ./deleteLocal.sh $$name	