#ifndef IONWAKE_OFILE_INCLUDED
#define IONWAKE_OFILE_INCLUDED

#include <ostream>

class OFile
{
public:

	void printTitle();
	void printPair();
	void print();

private:

	std::ofstream file;

};



#endif // include guard
