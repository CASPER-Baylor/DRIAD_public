/*
* Project: CUDAvar
* File Type: class header
* File Name: CUDAvar.h
*
* Created: 10/21/2017
* Last Modified: 11/12/2017
*
* Description:
*	Includes a class for abstracting CUDA variables 
*
* Includes:
*	CUDAvar
*		cuda_runtime.h
*		device_launch_parameters.h
*       CUDAerr.h
*
*/

#ifndef CUDA_VAR
#define CUDA_VAR

	/*
	* Required By:
	*	CUDAvar
	* For:
	*	CUDA
	*/
	#include "cuda_runtime.h"

	/*
	* Required By:
	*	CUDAvar
	* For:
	*	CUDA
	*/
	#include "device_launch_parameters.h"

    /*
	* Required By:
	*	CUDAvar
	* For:
	*	handeling CUDA errors
	*/
	#include "CUDAerr.h"
   
	//! A container for CUDA variables. 
	/*
	* Name: CUDAvar
	*
	* Editors
	*	Dustin Sanford
	*
	* Description:
	*	Provides abstractions for CUDA variables.
	*
	* Variables:
	*	hostPtr: pointer to the host side version of the CUDA variable
    *   devPtr: pointer to the device side CUDA variable
    *   memSize: the memory size of the variable 
	*
	* Methods:
    *   CUDAvar()
    *   CUDAvar( int )
	*	CUDAvar( Type*, int )
    *   ~CUDAvar()
    *   int getMemSize()
    *   Type* getDevPtr(){
    *   const Type* getHostPtr()
    *   void setHostPtr( Type* )
    *   void allocMem( int )
    *   bool checkHostPtr( Type* )
    *   void hostToDev()
    *   void devToHost()
    *   bool compare()    
	*
	* Includes:
	*	cuda_runtime.h
	*	device_launch_parameters.h
	*/
    
    template <class Type>
    class CUDAvar{
    
    private:
    
        // pointer to the paired host variable
        Type* hostPtr;
        // pointer to the device pointer
        Type* devPtr;
        // size of memory allocated on the device
        int memSize;
        // holds the CUDA status
        cudaError_t cudaStatus;
    
    public:
    
        /*
        * default constructor 
        *
        * Editors
        *	Dustin Sanford
        *
        * Description:
        *	constructs a blank CUDAvar
        *
        * Input:
        *
        * Output (void):
        *	memSize: 
        *
        * Assumptions:
        *	All inputs are real values
        *
        * Includes:
        *	cuda_runtime.h
        *	device_launch_parameters.h
        *
        */
        CUDAvar();
        
        /*
        * constructor
        *
        * Editors
        *	Dustin Sanford
        *
        * Description:
        *	constructs a CUDAvar from an array size
        *
        * Input:
        *	newMemSize: size of array to allocate memory for
        *
        * Output (void):
        *	hostPtr: NULL
        *   memSize: size of memory allocated on the device 
        *   devPtr: points to the device variable
        *
        * Assumptions:
        *
        * Includes:
        *	cuda_runtime.h
        *	device_launch_parameters.h
        *
        */
        CUDAvar( int );
        
        /*
        * constructor 
        *
        * Editors
        *   Dustin Sanford
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
        CUDAvar( Type*, int );
        
        /*
        * defaul destructor
        *
        * Editors
        *	Dustin Sanford
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
        ~CUDAvar();
        
        // getters
        int getMemSize(){ return memSize; };
        Type* getDevPtr(){ return devPtr; };
        Type* getHostPtr(){ return hostPtr; };

        /*
        * setHostPtr
        *
        * Editors
        *	Name: Dustin Sanford
        *
        * Description:
        *	Pairs the CUDAvar with a host variable 
        *
        * Input:
        *	newPtr: the pointer to the host variable to pair with
        *
        * Output (void):
        *	hostPtr points to the host pointer 
        *
        * Assumptions:
        *	The CUDAvar is not already paired with another variable 
        *
        * Includes:
        *	cuda_runtime.h
        *	device_launch_parameters.h
        *
        */
        void setHostPtr( Type* );
        
        /*
        * allocMem
        *
        * Editors
        *	Dustin Sanford
        *
        * Description:
        *	allocates memory on the GPU 
        *
        * Input:
        *	newMemSize: the size of array to allocate memory for
        *
        * Output (void):
        *	Memory is allocated on the GPU
        *
        * Assumptions:
        *	No memory has already been alocated 
        *
        * Includes:
        *	cuda_runtime.h
        *	device_launch_parameters.h
        *
        */
        void allocMem( int );
        
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
        bool checkHostPtr( Type* );
        
        /*
        * hostToDev
        *
        * Editors
        *	Dustin Sanford
        *
        * Description:
        *	copies device memory to the host
        *
        * Input:
        *
        * Output (void):
        *	The device memory is copied to the device pointer
        *
        * Assumptions:
        *
        * Includes:
        *	cuda_runtime.h
        *	device_launch_parameters.h
        *
        */
        void hostToDev();
        
        /*
        * devToHost
        *
        * Editors
        *	Dustin Sanford
        *
        * Description:
        *	copies device memory to the host
        *
        * Input:
        *
        * Output (void):
        *	The device memory is copied to the device pointer
        *
        * Assumptions:
        *
        * Includes:
        *	cuda_runtime.h
        *	device_launch_parameters.h
        *
        */
        void devToHost();
        
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
    * Name: default constructor 
    * Created: 10/21/2017
    * last edit: 10/26/2017
    *
    * Editors
    *	Name: Dustin Sanford
    *	Contact: Dustin_Sanford@baylor.edu
    *	last edit: 10/26/2017
    *
    * Description:
    *	constructs a blank CUDAvar
    *
    * Input:
    *
    * Output (void):
    *	memSize: 
    *
    * Assumptions:
    *	All inputs are real values
    *
    * Includes:
    *	cuda_runtime.h
    *	device_launch_parameters.h
    *
    */
    template <class Type>
    CUDAvar<Type>::CUDAvar() {
        hostPtr = NULL;
        devPtr = NULL;
        memSize = 0;
    }
    
    /*
    * Name: constructor
    * Created: 10/21/2017
    * last edit: 10/26/2017
    *
    * Editors
    *	Name: Dustin Sanford
    *	Contact: Dustin_Sanford@baylor.edu
    *	last edit: 10/26/2017
    *
    * Description:
    *	constructs a CUDAvar from an array size
    *
    * Input:
    *	newMemSize: size of array to allocate memory for
    *
    * Output (void):
    *	hostPtr: NULL
    *   memSize: size of memory allocated on the device 
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
    CUDAvar<Type>::CUDAvar( int newMemSize ) {
        
        memSize = newMemSize * sizeof(Type);
        hostPtr = NULL;
        
        // allocate memory on the GPU
        cudaStatus = cudaMalloc(&devPtr, memSize);
        // check if the allocation was successful
        if (cudaStatus != cudaSuccess) {
            
            // CUDAerr error;
            // error.setType("cudaMalloc");
            // error.makeFatal();
            // throw(error);
            
        }

    }

    /*
    * Name: constructor 
    * Created: 10/2/2017
    * last edit: 10/26/2017
    *
    * Editors
    *	Name: Dustin Sanford
    *	Contact: Dustin_Sanford@baylor.edu
    *	last edit: 10/26/2017
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
    CUDAvar<Type>::CUDAvar( Type* newHostPtr, int newMemSize ) {
        
        memSize = newMemSize * sizeof(Type);
        hostPtr = newHostPtr;
        
        // allocate memory on the GPU
        cudaStatus = cudaMalloc(&devPtr, memSize);
        // check if the allocation was successful
        if (cudaStatus != cudaSuccess) {
            // CUDAerr error;
            // error.setType("cudaMalloc");
            // error.makeFatal();
            // throw(error);
        }
       

    }
    
    /*
    * Name: defaul destructor
    * Created: 10/21/2017
    * last edit: 10/26/2017
    *
    * Editors
    *	Name: Dustin Sanford
    *	Contact: Dustin_Sanford@baylor.edu
    *	last edit: 10/26/2017
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
    CUDAvar<Type>::~CUDAvar(){
        cudaFree(devPtr);
    }
    

    /*
    * Name: setHostPtr
    * Created: 10/25/2017
    * last edit: 10/25/2017
    *
    * Editors
    *	Name: Dustin Sanford
    *	Contact: Dustin_Sanford@baylor.edu
    *	last edit: 10/25/2017
    *
    * Description:
    *	Pairs the CUDAvar with a host variable 
    *
    * Input:
    *	newPtr: the pointer to the host variable to pair with
    *
    * Output (void):
    *	hostPtr points to the host pointer 
    *
    * Assumptions:
    *	The CUDAvar is not already paired with another variable 
    *
    * Includes:
    *	cuda_runtime.h
    *	device_launch_parameters.h
    *
    */
    template <class Type>
    void CUDAvar<Type>::setHostPtr( Type* newPtr) {
        
        if (hostPtr == NULL) {
            
            hostPtr = newPtr;
            
        }
        
    }
    
    /*
    * Name: allocMem
    * Created: 10/25/2017
    * last edit: 10/25/2017
    *
    * Editors
    *	Name: Dustin Sanford
    *	Contact: Dustin_Sanford@baylor.edu
    *	last edit: 10/25/2017
    *
    * Description:
    *	allocates memory on the GPU 
    *
    * Input:
    *	newMemSize: the size of array to allocate memory for
    *
    * Output (void):
    *	Memory is allocated on the GPU
    *
    * Assumptions:
    *	No memory has already been alocated 
    *
    * Includes:
    *	cuda_runtime.h
    *	device_launch_parameters.h
    *
    */
    template <class Type>
    void CUDAvar<Type>::allocMem( int newMemSize) {
        
        if (memSize == 0){
            
            memSize = newMemSize * sizeof(Type);
            
            // allocate memory on the GPU
            cudaStatus = cudaMalloc(&devPtr, memSize);
            // check if the allocation was successful
            if (cudaStatus != cudaSuccess) {
                // CUDAerr error;
                // error.setType("cudaMalloc");
                // error.makeFatal();
                // throw(error);
            }
            
        }
        
    }
    
    /*
    * Name: checkHostPtr
    * Created: 10/125/2017
    * last edit: 10/25/2017
    *
    * Editors
    *	Name: Dustin Sanford
    *	Contact: Dustin_Sanford@baylor.edu
    *	last edit: 10/25/2017
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
    bool CUDAvar<Type>::checkHostPtr( Type* ptr) {
        if (ptr == hostPtr){
            return true;
        } else {
            return false;
        }
    }
    
    /*
    * Name: hostToDev
    * Created: 10/25/2017
    * last edit: 10/25/2017
    *
    * Editors
    *	Name: Dustin Sanford
    *	Contact: Dustin_Sanford@baylor.edu
    *	last edit: 10/25/2017
    *
    * Description:
    *	copies device memory to the host
    *
    * Input:
    *
    * Output (void):
    *	The device memory is copied to the device pointer
    *
    * Assumptions:
    *
    * Includes:
    *	cuda_runtime.h
    *	device_launch_parameters.h
    *
    */
    template <class Type>
    void CUDAvar<Type>::hostToDev(){
        
        if( memSize != 0 & hostPtr != NULL ){
        	// copy the host value to the gpu
            cudaStatus = cudaMemcpy(devPtr, hostPtr, 
                memSize, cudaMemcpyHostToDevice);
            // check if the memory copy was successful
            if (cudaStatus != cudaSuccess) {
                // CUDAerr error;
                // error.setType("cudaMemcpy");
                // error.makeFatal();
                // throw(error);
            }
        }       
    }
    
    /*
    * Name: devToHost
    * Created: 10/25/2017
    * last edit: 10/25/2017
    *
    * Editors
    *	Name: Dustin Sanford
    *	Contact: Dustin_Sanford@baylor.edu
    *	last edit: 10/25/2017
    *
    * Description:
    *	copies device memory to the host
    *
    * Input:
    *
    * Output (void):
    *	The device memory is copied to the device pointer
    *
    * Assumptions:
    *
    * Includes:
    *	cuda_runtime.h
    *	device_launch_parameters.h
    *
    */
    template <class Type>
    void CUDAvar<Type>::devToHost(){
        
        if( memSize != 0 & hostPtr != NULL ){
        	// copy the GPU value to the host
            cudaStatus = cudaMemcpy(hostPtr, devPtr, 
                memSize, cudaMemcpyDeviceToHost);
            // check if the memory copy was successful
            if (cudaStatus != cudaSuccess) {
                // CUDAerr error;
                // error.setType("cudaMemcpy");
                // error.makeFatal();
                // throw(error);
            }        
        }       
    }
    
    /*
    * Name: compare
    * Created: 10/25/2017
    * last edit: 10/25/2017
    *
    * Editors
    *	Name: Dustin Sanford
    *	Contact: Dustin_Sanford@baylor.edu
    *	last edit: 10/25/2017
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
    bool CUDAvar<Type>::compare(){
     
        Type* testVal = new Type[memSize/sizeof(Type)];
        
        // copy the GPU value to the host
        cudaStatus = cudaMemcpy(testVal, devPtr, 
            memSize, cudaMemcpyDeviceToHost);
        // check if the memory copy was successful
        if (cudaStatus != cudaSuccess) {
            // CUDAerr error;
            // error.setType("cudaMemcpy");
            // error.makeFatal();
            // throw(error);
        } 
        
        if (*testVal == *hostPtr) {
            return true;
        } else {
            return false;
        }
    }
    
#endif // CUDA_VAR
