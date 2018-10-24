#ifndef FUNCTION_GENERATORS_INCLUDED
#define FUNCTION_GENERATORS_INCLUDED

/*!
 *	\file FunctionGenerators.h
 *	
 *	\author Dustin Sanford
 *	\date October 12, 2018
 *
 *	Contains classes following the Function Generator concept.
 */

// included for sin() and exp()
#include <cmath>

//! Contains classes following the Function Generator concept.
namespace FunctionGenerator
{

//! Function Generator for a Sin function
/*! 
 *	Represents a sin function and follows the Function Generator concept.
 *	Both the frequency and  phase shift can be adjusted. The sin function is
 *	given by
 *
 *		\f[ g(x) = \sin{ \left( x \frac{f}{2 \pi} + \phi \right) } \f]
 *
 *	where \f$ f \f$ is the frequency and \f$ \phi \f$ is the phase shift.
 */
class Sin
{
public:

	//! Default constructor
	/*!
	 *	Constructs a Sin Function Generator object with a frequency of 
	 *  \f$ 2\pi \f$ and phase shift of zero.
	 */
	Sin() : phase{0} { this->setFrequency(6.283185307179586476925286766559); }

	//! Specific constructor
	/*!
	 *	Constructs a Sin Function Generator object with the specified frequency
	 *	and phase shift.
	 *
	 *	\param[in] f the frequency
	 *	\param[in] p the phase shift
	 */
	Sin( float f, float p ) : frequency{f}, phase{p} {}

	//! Function call operator
	/*!
	 *	Evaluates the sin function at x.
	 *
	 *	\param[in] x the location to evaluate the sin function at.
	 *
	 *	\return the sin function evaluated at x. 
	 */
	float operator()( float x )const{ return std::sin( phase + x*frequency ); }

	//! Frequency mutator 
	/*!
	 *	Sets the frequency of the sin function.
	 *	
	 *	\param[in] f the new frequency.
	 */
	void setFrequency( float f )
	{
		// frequency is saved as f / (2 pi) 
		frequency = f / 6.283185307179586476925286766559; 
	}
 
	//! Phase shift mutator
	/*!
	 *	Sets the phase shift of the sin function.
	 *
	 * \param[in] p the new phase shift.
	 */
 	void setPhase( float p ){ phase = p; }

	//! Frequency accessor
	/*!
	 *	Accesses the frequency of the sin function
	 *
	 *	\return the frequency of the sin function.
	 */
	float getFrequency()
	{ 
		// frequency is saved as f / (2 pi)
		return frequency * 6.283185307179586476925286766559; 
	} 

	//! Phase shift accessor
	/*!
	 *	Accesses the phase shift of the sin function.
	 *
	 *	\return the phase shift of the sin function.
	 */
	float getPhase(){ return phase; }

private:
	// the frequency of the sin function
	float frequency;
	// the phase shift of the sine funcution
	float phase;
};



//! Function Generator for a Gaussian function
/*! 
 *	Represents a Gaussian function and follows the Function Generator concept.
 *	The peak location, height and width can all be adjusted. The Gaussian is 
 *	given by
 *
 *	\f[ f(x) = A e^{-\frac{(x-p)^2}{w}} \f]
 *
 *	where \f$A\f$ is the peak height, \f$p\f$ is the peak location, and \f$w\f$ 
 *	is the width of the peak.
 */
class Gaussian
{
public:

	//! Default constructor 
	/*!
	 *	Constructs a Gaussian function peaked at zero with a width and height 
	 *	of one.
	 */	
	Gaussian() : peak{0}, width{1}, height{1} {}

	//! Specific constructor 
	/*! 
	 *	Constructs a Gaussian function with the specified peak location, height
	 *	and width
	 *
	 *	\param[in] p the location of the peak.
	 *	\param[in] w the width of the Gaussian.
	 *	\param[in] h the height of the peak.
	 */
	Gaussian( float p, float w, float h )
		: peak{p}, width{w}, height{h}
	{}

	//! Function call operator
	/*!
	 *	Evaluates the Gaussian function.
	 *
	 *	\param[in] x the location to evaluate the Gaussian at.
	 *
	 *	\return the value of the Gaussian at x.
	 */
	float operator()( float x )const
	{ 
		return height * exp( -(x - peak)*(x - peak)/width ); 
	} 

	//! Set the location of the peak
	void setPeak( float p ){ peak = p; }
	//! Set the width of the Gaussian
	void setWidth( float w ){ width = w ;}
	//! Set the height of the Gaussian
	void setHeight( float h ){ height = h; }

	//! Access the location of the peak
	float getPeak(){ return peak; }
	//! Access the width of the Gaussian
	float getWidth(){ return width; }
	//! Access the height of the peak
	float getHeight(){ return height; }

private:
	// The location of the peak, the width of the peak, and the height of the
	// peak respectively.
	float peak, width, height;
};

}; // namespace FunctionGenerator

#endif // include guard
