/*
* File Type: class header
* File Name: ErrorBase.h
*
* Created: 11/12/2017
* Last Modified: 11/12/2017
*
* Description:
*	Includes an abstract class for error classes
*
*
*/

using namespace std; 

#ifndef ERROR_BASE
#define ERROR_BASE

    /*
	* Required By:
	*	ErrorBase
	* For:
	*	string
	*/
    #include <string>

    /*
	* Name: ErrorBase
	*
	* Editors
	*	Dustin Sanford
	*
	* Description:
	*	An abstract class for exception classes
	*
	* Variables:
	*	errorType: the type of error
	*
	* Methods:
    *   print()       
	*
	* Includes:
    *   none
    *
	*/
    class ErrorBase{
      
    protected:

        string errorType;
        bool isFatal;

    public:
    
        /*
        * print
        *
        * Editors
        *	Dustin Sanford
        *
        * Description:
        *	prints an error message
        *
        * Input:
        *   none
        *
        * Output (void):
        *	none
        *
        * Assumptions:
        *   none
        *
        * Includes:
        *
        */
        virtual void print() = 0;
        
        /*
        * handle
        *
        * Editors
        *	Dustin Sanford
        *
        * Description:
        *	handles the exception
        *
        * Input:
        *   none
        *
        * Output (void):
        *	none
        *
        * Assumptions:
        *   none
        *
        * Includes:
        *
        */
        virtual void handle() = 0;
      
    };

#endif // ERROR_BASE