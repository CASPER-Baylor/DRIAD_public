
#ifndef IONWAKE_103
#define IONWAKE_103

#include <stdio.h>
#include <fstream>
#include <string>
#include <iostream>


void getUsserParams(float params[], int numUsserParams, std::string fileName)
{

	std::string dump;

	std::ifstream paramFile;

	// try to open the parameter file
	paramFile.open(fileName.c_str());

	// if the file did not open display an error
	if (!paramFile)
	{
		fprintf(stderr, "ERROR: file not open (IonWake_103_getUsserParams.h)");
	}

	// skip the first 5 lines of the file
	for (int i = 1; i <= 5; i++)
	{
		std::getline(paramFile, dump);
	}

	// loop over the contents of the file
	for (int i = 0; i < numUsserParams; i++)
	{
		// skip two columns
		paramFile >> dump >> dump;

		// save the parameter
		paramFile >> params[i];
	}

}

void getUsserParams(int params[], int numUsserParams, std::string fileName)
{

	std::string dump;

	std::ifstream paramFile;

	// try to open the parameter file
	paramFile.open(fileName.c_str());

	// if the file did not open display an error
	if (!paramFile)
	{
		fprintf(stderr, "ERROR: file not open (IonWake_103_getUsserParams.h)");
	}

	// skip the first 5 lines of the file
	for (int i = 1; i <= 5; i++)
	{
		std::getline(paramFile, dump);
	}

	// loop over the contents of the file
	for (int i = 0; i < numUsserParams; i++)
	{
		// skip two columns
		paramFile >> dump >> dump;

		// save the parameter
		paramFile >> params[i];
	}

}

#endif // IONWAKE_103