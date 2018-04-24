#ifndef _DATA_STRUCT_H_
#define _DATA_STRUCT_H_

#pragma once
#include <vector> 
#include <iostream>
#include <fstream>
#include "opencv2/nonfree/features2d.hpp" 

using namespace cv;

struct pointInfo          //2 dimension point information
{
	int no;   //object number label
	Point2d imagePoint;       //feature position
};


struct cloudPoint          //3 dimension point information
{
	int no;              //manage to stock 3D point in the position index as it no.
	Point3d objectPoint;
	int rgb[3];
	vector<int> frameIndex;
};

struct frameInfo
{
	int no;
	Mat_<double> ProjectMatrix;
	vector<pointInfo> pointSet;
};

struct globs_
{
	double *rot0params;
	double *intrcalib;
	int nccalib; 
	int ncdist; 
	int cnp, pnp, mnp; 
	double *ptparams;
	double *camparams;
};

#endif // _DATA_STRUCT_H_


