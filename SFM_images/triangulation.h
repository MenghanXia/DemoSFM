#pragma once
#include "dataStruct.h"
#include "match.h"
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace std;


class Triangulation
{
public:
	Triangulation(Mat_<double> innierMat,Size imgSize_,vector<frameInfo> *frameList_,vector<cloudPoint> *objectClouds_)
	{
		K = innierMat;
		imgSize = imgSize_;
		sumOutlier = 0;
		frameList = frameList_;
		objectClouds = objectClouds_;
		int k = 0;
		while(k < 1000000) {joinStopTimes.push_back(1);k ++;}
	};
	double reprojectError(Point3d object, Mat_<double> P, Point2d pointSet);
	double meanReprojectError(int start, int end, double gapCoeff);

	void getPointsfromMultiview(int curFrameIndex,bool isFirstPair);
	Point3d triangulatePoint(vector<int> frameRelated,int pointLabel);
	void addNewPoint(vector<int> newPointLabel,vector<Point3d> &newPoints);
	Point3d triangulate3Dpoint(Mat_<double> P1,Mat_<double> P2,Point2d u,Point2d u1);
	bool shallStart(int pointLabel,int curFrameIndex);

	void cloudRadiusFilter(vector<Point3d> inputSet, double radius, int thresholdSum,vector<char> &status);
	bool isIncluded(vector<int> dataSet,int queryIndex);
	bool isIncluded(vector<Point2d> dataSet,Point2d queryPoint);
	void extractObjectPoints(int frameIndex, vector<Point3d> &objectPoints);
	bool isDriftExist(Mat_<double> Transform);
	bool checkRepeatStruct(vector<int> neigborPointLabel,vector<int> &relateFrames_);
	void modifycurSaM(vector<vector<Point2d> > PtList1,vector<vector<Point2d> > PtList2,int frameIndex);
	bool findPerspectiveMat(vector<Point3d> objectPts1,vector<Point3d> objectPts2,Mat_<double> &Hmatrix);
	bool getTranformation1(vector<vector<Point2d> > PtList1,vector<vector<Point2d> > PtList2,int frameIndex,Mat_<double> &Hmatrix);
	bool getTranformation(vector<vector<Point2d> > PtList1,vector<vector<Point2d> > PtList2,int frameIndex,Mat_<double> &Hmatrix);
	void findSamePoints(vector<Point2d> querySet,vector<pointInfo> trainSet,vector<int> &status);

public:
	int sumOutlier;
	Size imgSize;
	vector<frameInfo>* frameList;
	vector<cloudPoint> *objectClouds;
	vector<int> joinStopTimes;
	Mat_<double> K;
	vector<pointInfo> pointSet1;
	vector<pointInfo> pointSet2;

	vector<int> relateFrames;
	vector<int> neigborFrames;

};