# This is the primary CMakeLists.txt file for the IonWake project

################################################
##    GLOBAL SETTINGS
################################################

# A CMake version 3.10 or higher is required 
cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

# set the CUDA architectures for KODIAK
set(CMAKE_CUDA_ARCHITECTURES 60 70 75)

# set the CUDA architectures for Ubuntu 20.04 in the local machine
#set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80)

# path to the compiler in KODIAK
set(CMAKE_CUDA_COMPILER "/cm/shared/apps/cuda12.3/toolkit/12.3.2/bin/nvcc")

# set the project language to C++
project(IonWake LANGUAGES CUDA CXX)

# set all targets to use C++14
set(CMAKE_CXX_STANDARD 14)

# it is an error if C++14 is not available 
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set the output directory for executable files
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/../bin")

# increase the output level of the created makefile 
set(CMAKE_VERBOSE_MAKEFILE OFF)

#set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS} --generate-line-info")
set(CMAKE_CXX_FLAGS_DEBUG "-g")

# find GSL (GNU Scientific Library)
find_package(GSL REQUIRED)
link_libraries(GSL::gsl GSL::gslcblas)

# Add CUDA files to the project
set(CUDA_FILES
    IonWake_000.cu
    IonWake_100_integrate.cu
    IonWake_101_bounds.cu
    IonWake_102_ionAcc.cu
    IonWake_103_dustAcc.cu
    IonWake_104_roadBlock.cu
    IonWake_105_ionColl.cu
    IonWake_106_Utilities.cu
)

#create the executable
add_executable(IonWake.exe ${CUDA_FILES})

target_compile_options(IonWake.exe PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       -O3
                       --relocatable-device-code=true
                       -Xptxas --fmad=true 
                       --use_fast_math # apply fast math optimizations  
                       #--maxrregcount=40
                       #--generate-line-info # for debugging with nsight compute
                       >)

# allow separate compilation. it is neccessary for multiple .cu files
set_target_properties(IonWake.exe PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link GSL to the executable
#target_link_libraries(IonWake.exe PUBLIC GSL::gsl GSL::gslcblas)

# Include GSL headers
#target_include_directories(IonWake.exe PUBLIC ${GSL_INCLUDE_DIRS})