/*
 * Project: CUDAvar
 * File Type: class header
 * File Name: CUDAerr.h
 *
 * Created: 10/24/2017
 * Last Modified: 10/24/2017
 *
 * Description:
 *	Includes classes for handling CUDA errors
 *
 * Includes:
 *	BaseError
 *
 */

using namespace std;

#ifndef CUDA_ERR
#define CUDA_ERR

/*
 * Required By:
 *	CUDAerr
 * For:
 *	ErrorBase
 */
#include "ErrorBase.h"

/*
 * Required By:
 *	CUDAerr
 * For:
 *	string
 */
#include <string>

/*
 * Name: CUDAerr
 *
 * Editors
 *	Dustin Sanford
 *
 * Description:
 *	An exception class for CUDA errors
 *
 * Variables:
 *	errorType: the type of error
 *   isFatal: true if the exception requires the program to terminate
 *
 * Methods:
 *   setType( string )
 *   print()
 *
 * Includes:
 *	ErrorBase
 *
 */
class CUDAerr : public ErrorBase
{

private:
public:
    /*
     * default constructor
     *
     * Editors
     *	Dustin Sanford
     *
     * Description:
     *	constructs an object without parameters
     *
     * Input:
     *   none
     *
     * Output (void):
     *	errorType: set to a blank string
     *   isFatal: set to false
     *
     * Assumptions:
     *   none
     *
     * Includes:
     *
     */
    CUDAerr();

    /*
     * print
     *
     * Editors
     *	Dustin Sanford
     *
     * Description:
     *	prints the error type to stderr
     *
     * Input:
     *   none
     *
     * Output (void):
     *	stderr: prints the error typ
     *
     * Assumptions:
     *   the error type is set
     *
     * Includes:
     *
     */
    virtual void print();

    /*
     * setType
     *
     * Editors
     *	Dustin Sanford
     *
     * Description:
     *	sets the type of error
     *
     * Input:
     *   typeName; the type of exception
     *
     * Output (void):
     *	setType: set to the input string
     *
     * Assumptions:
     *   none
     *
     * Includes:
     *
     */
    void setType(string);

    /*
     * makeFatal
     *
     * Editors
     *	Dustin Sanford
     *
     * Description:
     *	makes the exception fatal
     *
     * Input:
     *   none
     *
     * Output (void):
     *	isFatal: set to true
     *
     * Assumptions:
     *   none
     *
     * Includes:
     *
     */
    void makeFatal() { isFatal = true; }

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
    virtual void handle();
};

/*
 * Name: default constructor
 * Created: 11/12/2017
 * last edit: 11/12/2017
 *
 * Editors
 *	Name: Dustin Sanford
 *	Contact: Dustin_Sanford@baylor.edu
 *	last edit: 11/12/2017
 *
 * Description:
 *	constructs an object without parameters
 *
 * Input:
 *   none
 *
 * Output (void):
 *	errorType: set to a blank string
 *   isFatal: set to false
 *
 * Assumptions:
 *   none
 *
 * Includes:
 *
 */
CUDAerr::CUDAerr()
{

    errorType = "";
    isFatal = false;
}

/*
 * Name: setType
 * Created: 11/12/2017
 * last edit: 11/12/2017
 *
 * Editors
 *	Name: Dustin Sanford
 *	Contact: Dustin_Sanford@baylor.edu
 *	last edit: 11/12/2017
 *
 * Description:
 *	sets the type of error
 *
 * Input:
 *   typeName; the type of exception
 *
 * Output (void):
 *	setType: set to the input string
 *
 * Assumptions:
 *   none
 *
 * Includes:
 *
 */
void CUDAerr::setType(string typeName) { errorType = typeName; }

/*
 * Name: print
 * Created: 11/12/2017
 * last edit: 11/12/2017
 *
 * Editors
 *	Name: Dustin Sanford
 *	Contact: Dustin_Sanford@baylor.edu
 *	last edit: 11/12/2017
 *
 * Description:
 *	prints the error type to stderr
 *
 * Input:
 *   none
 *
 * Output (void):
 *	stderr: prints the error typ
 *
 * Assumptions:
 *   the error type is set
 *
 * Includes:
 *
 */
void CUDAerr::print()
{

    // print the error type to standard error
    fprintf(stderr, "%s\n", errorType.c_str());
}

/*
 * Name: handle
 * Created: 11/12/2017
 * last edit: 11/12/2017
 *
 * Editors
 *	Name: Dustin Sanford
 *	Contact: Dustin_Sanford@baylor.edu
 *	last edit: 11/12/2017
 *
 * Description:
 *	There are
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
void CUDAerr::handle() {}

#endif // CUDA_ERR