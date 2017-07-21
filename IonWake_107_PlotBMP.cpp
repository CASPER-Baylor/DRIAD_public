#include "IonWake_107_PlotBMP.h"
#include <string>
#include <iostream>
#include "EasyBMP.h"
#include "IonWake_106_Utilities.h"


int getGreen(int Ival, int numColor)
{
	float numShifted = (static_cast<float>(Ival) * 2);
	if (numShifted > numColor)
	{
		numShifted -= 2 * (numShifted - numColor);
	}
	float val = (255 * numShifted) / (numColor);;
	return static_cast<int>(val);
}

int getRed(int num, int numColor)
{
	float fNum = static_cast<float>(num);
	float val = (255 * ((fNum * 2) - numColor)) / (numColor);
	if (val < 0)
	{
		val = 0;
	}
	return static_cast<int>(val);
}

int getBlue(int num, int numColor)
{
	float fNum = static_cast<float>(num);
	float val = (255 * (numColor - (fNum * 2))) / (numColor);
	if (val < 0)
	{
		val = 0;
	}

	return static_cast<int>(val);
}

void createBmp(int imageHieght, int imageWidth, int* data, std::string imageName)
{

	int colorValue = 0;

	BMP imageOut;

	imageOut.SetSize(imageWidth, imageHieght);

	int numColors = imageOut.TellNumberOfColors();

	float maxVal = findMax(data, imageHieght*imageWidth);

	for (int i = 0; i < imageWidth; i++)
	{
		for (int j = 0; j < imageHieght; j++)
		{

			colorValue = *(data + ((j * imageWidth) + i));

			if (colorValue >= 0)
			{
				colorValue = (static_cast<float>(colorValue) / maxVal)*numColors;
				imageOut(i, j)->Red = getRed(colorValue, numColors);
				imageOut(i, j)->Green = getGreen(colorValue, numColors);
				imageOut(i, j)->Blue = getBlue(colorValue, numColors);
			}
			else
			{
				imageOut(i, j)->Red = 0;
				imageOut(i, j)->Green = 0;
				imageOut(i, j)->Blue = 0;
			}
		}
	}

	imageOut.WriteToFile(imageName.c_str());
}