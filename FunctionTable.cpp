#include "FunctionTable.h"

/*******************************************************
 *****                                             *****
 ***     Implementation File For FunctionTable.h     ***
 *****                                             *****
 *******************************************************/
 
// FunctionTable stream insertion operator 
std::ostream& operator<< ( std::ostream& out, FunctionTable table )  
{
	for( int i=0 ; i<table.size() ; i++ )
		out << table.getIndependent(i) << ", " << table.getDependent(i) << '\n';
	return out; 
}

// Cumulative sum of a function table
FunctionTable& FunctionTable::cumSum()
{
	float cum_sum = 0;

	for( int i=0 ; i<dependent_data.size() ; i++ )
	{
		cum_sum += dependent_data[i];
		dependent_data[i] = cum_sum;
	}
	
	return *this;
}

// Cumulative integration of a FunctionTable
FunctionTable& FunctionTable::cumInt()
{
	float cum_int = 0;

	// The integral is zero if there is one or less data points 
	if ( independent_data.size() > 1 )
	{
		float step_size = independent_data[1] - independent_data[0];

		// calculate the integral for the first data point		
		cum_int += (step_size / 2) * dependent_data[0];
		dependent_data[0] = cum_int;

		// calculate the integral for all data points which are not the first 
		// or last data point 
		for( int i=1 ; i<(independent_data.size() - 1) ; i++ )
		{
			cum_int += step_size * dependent_data[i];
			dependent_data[i] = cum_int;
		}

		// calculate the integral for the last data point
		int end = dependent_data.size() - 1;
		cum_int += (step_size / 2) * dependent_data[end];
		dependent_data[end] = cum_int;
	}

	return *this;
}

// Integral of a FunctionTable
float FunctionTable::integrate()
{
	return this->integrate( 0, dependent_data.size() );
}

// Integral of a FunctionTable
float FunctionTable::integrate( int start, int end )
{

	int bounds_mult = 1;
	float result = 0;

	// If the start is greater than the end, swap the bounds and add a negative
	if( start > end ) 
	{
		std::swap( start, end );
		bounds_mult = -1;
	}
		
	// If there is one or less data point in the range, then the integral is 0
	if( start != end )
	{
		float step_size = independent_data[1] - independent_data[0];
		
		// calculate the integral for the first point
		result += (step_size / 2) * dependent_data[ start ];

		// calculate the integral for all the points which are not the first
		// or last
		for( int i=start + 1; i<end - 1 ; i++ )
		{
			result += step_size * dependent_data[i];
		}
		
		// calculate the integral of the last point
		result += (step_size / 2) * dependent_data[ end - 1 ];
	} 

	return result * bounds_mult;

}

// Maximum Value
float FunctionTable::maxVal()
{
	float max = dependent_data[0];
	
	for( int i=1; i<dependent_data.size() ; i++ )
	{
		if( dependent_data[i] > max ) max = dependent_data[i];
	}
	
	return max;
}


// Normalize using the maximum value 
FunctionTable& FunctionTable::normMax( float new_max )
{
	float norm = new_max / this->maxVal();

	for( int i=0; i<dependent_data.size(); i++ )
	{
		dependent_data[i] *= norm;
	}

	return *this;
}

// Normalize using the integral 
FunctionTable& FunctionTable::normInt( float new_int )
{
	float norm = new_int / this->integrate();

	for( int i=0; i<dependent_data.size(); i++ )
	{
		dependent_data[i] *= norm;
	}

	return *this;
}
