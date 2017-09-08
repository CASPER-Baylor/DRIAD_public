#ifndef IONWAKE_107_PLOTBMP
#define IONWAKE_107_PLOTBMP

#include <string>
#include <string>
#include <iostream>
#include "EasyBMP.h"
#include "IonWake_106_Utilities.h"

int getGreen(int Ival, int numColor);

int getRed(int num, int numColor);

int getBlue(int num, int numColor);

void createBmp(int imageHieght, int imageWidth, int* data, std::string imageName);

#endif //IONWAKE_107_PLOTBMP