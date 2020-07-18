#include "IonWake_104_roadBlock.h"

/*
* Name: roadBlock_104
* Description:
*	Prints to the status file if print is on. Then checks for a CUDA error
* 	after the kernel launch. Then it synchronizes the threads and checks
*	for a CUDA error again. If a CUDA error found at any point it displays
*	an erro message and kills the program.
*
* Input:
* 	line: the line number where the function is called from
*	file: the file where the function is called from
*	name: a name that will be associated with the error
*	print: a boolian  value that determins if a message is
*		is printed to hte status file.
*
* Output <void>:
*	If a CUDA error is found the program will be terminated
*	and an error mesage will be printed to stderr
*
* Assumptions:
* 	status holds a variable of type cudaError_t
*
* Includes
* 	cuda_runtime.h
*	device_launch_parameters.h
*
*/

void roadBlock_104(ofstream& statusFile, int line, string file, string name, bool print) {
        cudaError_t cudaStatus;

        if (print) {
                // print the name to the status file
                statusFile << name << std::endl;
        }

        // check if there is a CUDA error after the kernel launch
        cudaStatus = cudaGetLastError();

        if (cudaStatus != cudaSuccess) {
                // print an error
                fprintf(stderr, "ERROR on line number %d in file %s\n", line, file.c_str());
                fprintf(stderr, "Kernel launch failed: %s\n", name.c_str());
                fprintf(stderr, "Error code : %s\n\n", cudaGetErrorString(cudaStatus));

                // terminate the program
                exit(-1);
        }

        // synchronize threads and check for a CUDA error
        cudaStatus = cudaDeviceSynchronize();

        if (cudaStatus != cudaSuccess) {
                // print an error
                fprintf(stderr, "ERROR on line number %d in file %s\n", line, file.c_str());
                fprintf(stderr, "Synchronize threads failed: %s\n", name.c_str());
                fprintf(stderr, "Error code : %s\n\n",cudaGetErrorString(cudaStatus));

                // terminate the program
    		exit(-1);   
	 }
}
