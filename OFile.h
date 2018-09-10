#ifndef IONWAKE_OFILE_INCLUDED
#define IONWAKE_OFILE_INCLUDED

#include <ostream>
#include <string>

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

	void printTitle( std::string );

	template< typename Printable >
	void printPair( std::string, Printable );

private:
	std::ofstream file;
	const static int fileWidth = 70;

};

void OFile::printTitle( std::string title )
{
    int titleLength = title.length();    
                                                                                                   
    int padding = (this->fileWidth - titleLength - 1) / 2;                              
    if (padding < 0) padding = 0;                 
                                                  
    int filler = this->fileWidth - (2 + 2*padding + titleLength);                             
                       
    file << "\n" << std::string(padding, '-')  << " " << title << " " 
		 << std::string(padding + filler, '-') <<  "\n";                             
}

template< typename Printable >
void OFile::printPair( std::string name , Printable value )
{
	file << '\n' << name << "= " << value;
}

#endif // include guard
