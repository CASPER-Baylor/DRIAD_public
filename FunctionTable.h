#ifndef FUNCTION_TABLE_INCLUDED
#define FUNCTION_TABLE_INCLUDED

/*!
 *  \file FunctionTable.h
 *
 *  \author Dustin Sanford
 *	\date October 12, 2018
 *
 *	Contains the FunctionTable class and free functions 
 */

// Used by FunctionTable
#include <vector>

// tablulate() uses std::swap()
#include <utility>

#include <iostream>

//!	Container for tables of function data
/*!
 *	Contains a list of i data points. Each data point consists of an 
 *	independent and dependent variable. Both the independent and dependent 
 *	variables are single-precision floating-point.   
 *
 *	Use push() to add data points to the FunctionTable. The data members are 
 *	accessed with getIndependent() and getDependent(). 
 *	
 *	\warning FunctionTable accessor functions do not perform bounds checking.
 */
class FunctionTable
{
public:

	//! Add a data point to the FunctionTable
	/*!
	 *	The data point is added to the end of the FunctionTable and
	 *	the length is incremented.
	 *
	 *	\param[in] i the independent variable value
	 *	\param[in] d the dependent variable value	
	 */
	void push( float i, float d ) 
	{
		dependent_data.push_back( d );
		independent_data.push_back( i );
	}

	//! Dependent variable accessor
	/*!
	 *	Returns the dependent variable value of the ith data point. No bounds 
	 *	checking is performed.	
	 *
	 *	\param[in] i the index of the data point to access.
	 *	\return the dependent variable value of the ith data point.
	 */ 
	float getDependent( int i ) const { return dependent_data[i]; }

	//! Independent variable accessor
	/*!
	 *	Returns the independent variable value of the ith data point. The data 
	 *	points are zero indexed. No bounds checking is performed.	
	 *
	 *	\param[in] i the index of the data point to access.
	 *	\return the independent variable value of the ith data point.
	 */ 
	float getIndependent( int i ) const { return independent_data[i]; }

	//!	Size accessor
	/*!
	 *	Accesses the number of data points in the FunctionTable.
	 *
	 *	\return the number of data points in the FunctionTable. 
	 */
	int size() const { return dependent_data.size(); }

	//! Cumulative sum of a FunctionTable
	/*!
	 *	Perform an in-place cumulative sum of the dependent variable in the 
	 *	FunctionTable.
	 */
	FunctionTable& cumSum();

	//! Cumulative integration of a FunctionTable
	/*
	 *	Perform an in-place cumulative integration of the dependent variable in
	 *	the FunctionTable.
	 *
	 *	If there are fewer than two entries in the table, nothing is done. 
	 */
	FunctionTable& cumInt();

	//! Integral of the FunctionTable
	/*!
	 *	Take the definite integral over the entire FunctionTable. If the 
	 *	FunctionTable is empty zero is returned.
	 *
	 *	NEED_MATH
	 *
	 *	\return the value of the integral.
	 */ 
	float integrate();

	//! Integral of the FunctionTable
	/*!
	 *	Take the definite integral of the FunctionTable over the specified 
	 *	bounds. The starting point is included and the end point is excluded.
	 *	The starting point is allowed to be greater than the end point. If the 
	 *	bounds are the same, then zero is returned. 
	 *
	 *	NEED_MATH
	 *
	 *	\return the value of the integral. 
	 *
	 *	\warning Bounds checking is not performed for the starting or ending 
	 *	points. 
	 */
	float integrate( int start, int end );

	//! Maximum Value
	/*!
	 *	\return the maximum value in the FunctionTable.	
	 *	\warning it is an error to call maxVal() on an empty FunctionTable.
	 */
	float maxVal();

	//! Normalize using the maximum value
	/*!
	 *	Normalizes all of the dependent variable values in the FunctionTable
	 *	based on the maximum dependent variable value.
	 *
	 *	NEED_MATH
	 *
	 *	\param[in] new_max the new maximum value after normalizing. 
	 *
 	 *	\warning it is an error to normalize a FunctionTable with a maximum 
	 *	value of zero or an empty FunctionTable.
	 */ 
	FunctionTable& normMax( float new_max = 1 );

	//! Normalize the Integral
	/*!
	 *	Normalizes all of the dependent variable values in the FunctionTable
	 *	based on the integral of the function table.
	 *
	 *	NEED_MATH
	 *
	 *	\param[in] new_int the new value of the integral of the FunctionTable
	 *	after normalizing.
	 *
	 *	\warning it is an error to call normInt() on a FunctionTable with
	 *	an integral of zero or an empty FunctionTable. 
	 */
	FunctionTable& normInt( float new_int = 1 );

	//! Stream extraction operator
	/*!
	 *	Prints the FunctionTable as a comma separated list. Each data point is
	 *	on a separate line, with the independent variable first. 
	 *
	 *	\param[in] out the stream to print the FunctionTable to.
	 *	\param[in] table the FunctionTable to print.
	 *
	 *	\return the passed in std::ostream is returned.
	 */
	friend std::ostream& operator<<( std::ostream& out, FunctionTable table );

private:
	// list of all the dependent data points
	std::vector<float> dependent_data;
	// list of all the independent data points
	std::vector<float> independent_data;
};

//! Creates a FunctionTable from a Function Generator
/*!
 *	\related FunctionTable
 *	Constructs a FunctionTable by evaluating a Function Generator at evenly 
 *	spaced intervals.
 *
 *	\param[in] func the Function Generator object used to calculate the 
 *		dependent variable values for the FunctionTable.
 *	\param[in] beginning the beginning value for the independent variable.
 *	\param[in] end the end point value for the independent variable.
 *	\param[in] num_steps the number of independent variable steps to take 
 *		between the start and end points including the start and end points. 
 *
 *	\return A FunctionTable with size num_steps whose dependent variable entries 
 *		are determined by the provided Function Generator. The value of the 
 *		independent variable of the first data point is beginning and the last
 *		is end.
 *
 *	\warning If beginning is less than end, the two values are switched. 
 */
template< typename FunctionGenerator >
FunctionTable tabulate
	( const FunctionGenerator& func, float beginning, float end, int num_steps )
{
	FunctionTable table;

	// ensure that beginning is less than end
	if( end < beginning ) std::swap( beginning, end ); 

	float step_size = (end - beginning) / (num_steps - 1);
	float pos = beginning;

	// populate the FunctionTable
	for( int i = 0; i < num_steps ; i++ )
	{
		table.push( pos,  func(pos) );
		pos += step_size;
	}

	return table;
}

#endif // include guard
