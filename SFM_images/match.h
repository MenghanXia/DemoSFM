#pragma once
#include "dataStruct.h"
#include <string>

using namespace std;
using namespace cv;

class readMatch
{
public:
	readMatch(string foldPath_)
	{
		foldPath = foldPath_;
	};
	bool getMatchResult(int frame1,int frame2,vector<Point2d> &pointSet1,vector<Point2d> &pointSet2,Mat_<int> &RGB);

public:
	string foldPath;

};