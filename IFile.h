#ifndef IONWAKE_IFILE_INCLUDED
#define IONWAKE_IFILE_INCLUDED

#include <cassert>

class IFile {
public:

    IFile( std::string fileName )
    {    
        std::ifstream file(fileName.c_str());
        assert( file );
    }    

    ~IFile()
    {    
        file.close();
    }    

    template< typename T >  
    IFile& operator>>( T& rhs )
    {    
        file >> rhs; 
        return *this;
    }    

    bool getline( std::string line ) 
    {    
        return  std::getline(file, line);
    }    

    void clear() 
    {    
        file.clear();
    }    

    void reset()
    {    
        file.seekg(0, std::ios::beg);
    }    

private:

    std::ifstream file;

};

#endif // include guard
