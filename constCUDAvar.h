/*
* Project: CUDAvar
* File Type: class header
* File Name: constCUDAvar.h
*
* Created: 11/12/2017
* Last Modified: 11/12/2017
*
* Description:
*	Includes a class for abstracting constant CUDA variables 
*
* Includes:
*	CUDAvar
*		cuda_runtime.h
*		device_launch_parameters.h
*
*/

#ifndef CONST_CUDA_VAR
#define CONST_CUDA_VAR

	/*
	* Required By:
	*	constCUDAvar
	* For:
	*	CUDA
	*/
	#include "cuda_runtime.h"

	/*
	* Required By:
	*	constCUDAvar
	* For:
	*	CUDA
	*/
	#include "device_launch_parameters.h"

    /*
    * Required By:
    *   constCUDAvar
    * For:
    *   handeling CUDA errors
    */
    #include "CUDAerr.h"
 
/*
 *  An exception type for when the value of a constCUDAvar is edited
 */
class ConstCUDAvarChanged 
{
};
   
	/*
	* Name: constCUDAvar
	*
	* Editors
	*	Dustin Sanford
	*
	* Description:
	*	Provides abstractions for constant CUDA variables.
	*
	* Variables:
	*	hostPtr: pointer to the host side version of the CUDA variable
    *   devPtr: pointer to the device side CUDA variable
    *   memSize: the memory size of the variable 
	*
	* Methods:
	*	constCUDAvar( const Type*, int )
    *   ~constCUDAvar()
    *   int getMemSize()
    *   Type* getDevPtr(){
    *   const Type* getHostPtr()
    *   bool checkHostPtr( const Type* )
    *   bool compare()
    *       
	*
	* Includes:
	*	cuda_runtime.h
	*	device_launch_parameters.h
	*/
    template <class Type>
    class constCUDAvar{
    
    private:
    
        // pointer to the paired host variable
        const Type* hostPtr;
        // pointer to the device pointer
        Type* devPtr;
        // size of memory allocated on the device
        int memSize;
        // holds the CUDA status
        cudaError_t cudaStatus;
    
    public:
        
        /*
        * constructor 
        *
        * Editors
        *	Dustin Sanford
        *
        * Description:
        *	Constructs the object from a host variable and memomry size
        *
        * Input:
        *	newHostPtr: pointer to the host vaiable to pair with
        *   newMemSize: the size of array to allocate memory for
        *
        * Output (void):
        *	memSize: size of the allocated memory
        *   hostPtr: points to the paired host var
        *   devPtr: points to the device variable 
        *
        * Assumptions:
        *
        * Includes:
        *	cuda_runtime.h
        *	device_launch_parameters.h
        *
        */
        constCUDAvar( const Type*, int );
        
        /*
        * defaul destructor
        *
        * Editors
        *	Dustin Sanford
        *
        * Description:
        *	destructs the constCUDAvar and frees the CUDA memory
        *
        * Input:
        *
        * Output (void):
        *	the object is destrcted 
        *
        * Assumptions:
        *
        * Includes:
        *	cuda_runtime.h
        *	device_launch_parameters.h
        *
        */
        ~constCUDAvar();
        
        // getters
        int getMemSize(){ return memSize; };
        Type* getDevPtr(){ return devPtr; };
        const Type* getHostPtr(){ return hostPtr; };
        
        /*
        * checkHostPtr
        *
        * Editors
        *	Dustin Sanford
        *
        * Description:
        *	Checks if a pointer is the host pointer 
        *
        * Input:
        *   ptr: the pointer to check agains the CUDAvar host pointer
        *
        * Output (void):
        *	return: true if the pointer is the host pointer
        *           flase if the pointer is not the host pointer 
        *
        * Assumptions:
        *   the input pointer is of the same type as the CUDAvar
        *
        * Includes:
        *	cuda_runtime.h
        *	device_launch_parameters.h
        *
        */
        bool checkHostPtr( const Type* );
        
        /*
        * compare
        *
        * Editors
        *	Dustin Sanford
        *
        * Description:
        *	compares the host and device variables.
        *
        * Input:
        *
        * Output (bool):
        *	return: true if the values are equal
        *           false if the values are not equal 
        *
        * Assumptions:
        *
        * Includes:
        *	cuda_runtime.h
        *	device_launch_parameters.h
        *
        */
        bool compare();
      
    };
        
    /*
    * Name: constructor 
    * Created: 11/12/2017
    * last edit: 11/12/2017
    *
    * Editors
    *	Name: Dustin Sanford
    *	Contact: Dustin_Sanford@baylor.edu
    *	last edit: 11/12/2017
    *
    * Description:
    *	Constructs the object from a host variable and memomry size
    *
    * Input:
    *	newHostPtr: pointer to the host vaiable to pair with
    *   newMemSize: the size of array to allocate memory for
    *
    * Output (void):
    *	memSize: size of the allocated memory
    *   hostPtr: points to the paired host var
    *   devPtr: points to the device variable 
    *
    * Assumptions:
    *
    * Includes:
    *	cuda_runtime.h
    *	device_launch_parameters.h
    *
    */
    template <class Type>
    constCUDAvar<Type>::constCUDAvar( const Type* newHostPtr, int newMemSize ) {
        
        memSize = newMemSize * sizeof(Type);
        hostPtr = newHostPtr;
        
        // allocate memory on the GPU
        cudaStatus = cudaMalloc(&devPtr, memSize);
        // check if the allocation was successful
        if (cudaStatus != cudaSuccess) {
        }
        
        // copy the host value to the gpu
        cudaStatus = cudaMemcpy(devPtr, hostPtr, 
            memSize, cudaMemcpyHostToDevice);
        // check if the memory copy was successful
        if (cudaStatus != cudaSuccess) {
        }

    }
    
    /*
    * Name: defaul destructor
    * Created: 11/12/2017
    * last edit: 11/12/2017
    *
    * Editors
    *	Name: Dustin Sanford
    *	Contact: Dustin_Sanford@baylor.edu
    *	last edit: 11/12/2017
    *
    * Description:
    *	destructs the CUDAvar and frees the CUDA memory
    *
    * Input:
    *
    * Output (void):
    *	the object is destrcted 
    *
    * Assumptions:
    *
    * Includes:
    *	cuda_runtime.h
    *	device_launch_parameters.h
    *
    */
    template <class Type>
    constCUDAvar<Type>::~constCUDAvar(){
		// check that the value of the constCUDAvar was not changed
		if( !(this->compare()) ) throw( ConstCUDAvarChanged() );

        cudaFree(devPtr);
    }
    
    /*
    * Name: checkHostPtr
    * Created: 11/12/2017
    * last edit: 11/12/2017
    *
    * Editors
    *	Name: Dustin Sanford
    *	Contact: Dustin_Sanford@baylor.edu
    *	last edit: 11/12/2017
    *
    * Description:
    *	Checks if a pointer is the host pointer 
    *
    * Input:
    *   ptr: the pointer to check agains the CUDAvar host pointer
    *
    * Output (void):
    *	return: true if the pointer is the host pointer
    *           flase if the pointer is not the host pointer 
    *
    * Assumptions:
    *   the input pointer is of the same type as the CUDAvar
    *
    * Includes:
    *	cuda_runtime.h
    *	device_launch_parameters.h
    *
    */
    template <class Type>
    bool constCUDAvar<Type>::checkHostPtr( const Type* ptr) {
        if (ptr == hostPtr){
            return true;
        } else {
            return false;
        }
    }
    
    /*
    * Name: compare
    * Created: 11/12/2017
    * last edit: 11/12/2017
    *
    * Editors
    *	Name: Dustin Sanford
    *	Contact: Dustin_Sanford@baylor.edu
    *	last edit: 11/12/2017
    *
    * Description:
    *	compares the host and device variables.
    *
    * Input:
    *
    * Output (bool):
    *	return: true if the values are equal
    *           false if the values are not equal 
    *
    * Assumptions:
    *
    * Includes:
    *	cuda_runtime.h
    *	device_launch_parameters.h
    *
    */
    template <class Type>
    bool constCUDAvar<Type>::compare(){
     
        Type* testVal = new Type[memSize/sizeof(Type)];
        
        // copy the GPU value to the host
        cudaStatus = cudaMemcpy(testVal, devPtr, 
            memSize, cudaMemcpyDeviceToHost);
        // check if the memory copy was successful
        if (cudaStatus != cudaSuccess) {
        } 
        
        if (*testVal == *hostPtr) {
            return true;
        } else {
            return false;
        }
    }
    
#endif // CUDA_VAR
