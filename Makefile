# variable to store the build directory
buildDirectory:=build
# variable to store the source directory
sourceDirectory:=source

# function to build the code in debug mode
build_debug:
	# create the source and run txt files to run the code
	./getPaths.sh
    # create the build directory if it doesn't exist
	@mkdir -p ${buildDirectory}

	# Execute the CMake
	@cmake -B ${buildDirectory} -DCMAKE_BUILD_TYPE=Debug ./${sourceDirectory}

    # compile the code
	@make -C ${buildDirectory}

# function to build the code in release mode
build_release:
	# create the source and run txt files to run the code
	./getPaths.sh

    # create the build directory if it doesn't exist
	@mkdir -p ${buildDirectory}

	# Execute the CMake
	@cmake -B ${buildDirectory} -DCMAKE_CSS_FLAGS="-O3" -DCMAKE_BUILD_TYPE=Release ./${sourceDirectory}

    # compile the code
	@make -C ${buildDirectory}

# function to remove the build directory
clean:
    # remove the build directory if it exists
	@rm -rf ${buildDirectory}

# function to run the code
run:
	# set job name, copy binary file and run the binary file
	@read -p "Name of Job: " name &&  \
	cd ../DRIAD_run_scripts && \
	./setup_run.sh $$name && \
	./compile.sh && ./run.sh $$name
	
# check the status of a job
status:

	@read -p "Name of Job: " name &&  \
	cd ../DRIAD_run_scripts && ./check.sh $$name
	
# get errors and extra informations of the job
info:

	@read -p "Name of Job: " name &&  \
	cd ../DRIAD_run_scripts && ./disp.sh $$name