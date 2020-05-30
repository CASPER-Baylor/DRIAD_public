# This is the primary CMakeLists.txt file for the IonWake project

################################################
##    GLOBAL SETTINGS
################################################

# A CMake version 3.10 or higher is required 
cmake_minimum_required( VERSION 3.10 FATAL_ERROR )

# set the project language to C++
project( IonWake LANGUAGES CXX CUDA )

# set all targets to use C++14
set( CMAKE_CXX_STANDARD 14 )
# it is an error if C++14 is not available 
set( CMAKE_CXX_STANDARD_REQUIRED ON )

# set the output directory for executable files
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/../bin" )

# increase the output level of the created makefile 
set(CMAKE_VERBOSE_MAKEFILE ON)

add_executable( IonWake.exe
	IonWake_000.cu
	IonWake_100_integrate.cu
	IonWake_101_bounds.cu
	IonWake_102_ionAcc.cu
	IonWake_103_dustAcc.cu
	IonWake_105_ionColl.cu
	IonWake_106_Utilities.cpp
 )

target_compile_features( IonWake.exe PUBLIC cxx_std_14 )
set_target_properties( IonWake.exe PROPERTIES CUDA_SEPARABLE_COMPILATION ON )

