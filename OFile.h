#ifndef IONWAKE_OFILE_INCLUDED
#define IONWAKE_OFILE_INCLUDED

#include <ostream>
#include <fstream>
#include <string>

class FileNotOpen{};

class OFile
{
public:

	// constructor 
	explicit OFile(std::string fileName)
	{
		file.open( fileName.c_str() );
	}

	// destructor 
	~OFile()
	{
		file.close();
	}
	
	// copy policy - not copyable 
	OFile& operator=( const OFile& ) = delete;
	OFile( const OFile& ) = delete;

	// move policy - movable 
	OFile& operator=( OFile&& ) = default;
	OFile( OFile&& ) = default;

	template< typename Printable >
	OFile& operator<<( Printable rhs  )
	{
		file << rhs;
		return *this;
	}

	OFile& flush() 
	{
		file.flush();
		return *this;
	}

	void precision( int num )
	{
		file.precision( num );
	}

	OFile& printTitle( std::string title )
	{
    	int titleLength = title.length();    
                                                                                                   
    	int padding = (this->fileWidth - titleLength - 1) / 2;                              
    	if (padding < 0) padding = 0;                 
                                                  
    	int filler = this->fileWidth - (2 + 2*padding + titleLength);                             
                       
    	file << "\n" << std::string(padding, '-')  << " " << title << " " 
			 << std::string(padding + filler, '-') <<  "\n";                             
	
		return *this;
	}

	template< typename Printable >
	OFile& printPair( std::string, Printable );

private:
	std::ofstream file;
	const static int fileWidth = 70;

};

template< typename Printable >
OFile& OFile::printPair( std::string name , Printable value )
{
	file << name << "= " << value << '\n';
	return *this;
}

#endif // include guard
