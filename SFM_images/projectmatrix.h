#pragma once
#include "dataStruct.h"
#include "match.h"
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/contrib/contrib.hpp>

#define NEW_POINT_NUM 500

using namespace std;
using namespace cv;

class ProjectMat
{
public:
	ProjectMat(Mat_<double> inierMatix,vector<frameInfo> *frameLists_,vector<cloudPoint> *objectClouds_)
	{
		frameLists = frameLists_;
		objectClouds = objectClouds_;
		K = inierMatix;
		maxPointLabel = 0;
		frameNo = 0;
	};
	void printM(Mat_<double> matrice);
	void setPointLabel(vector<Point2d> pointSet1,vector<Point2d> pointSet2);
	void findNewPoints(vector<Point2d> PointSet);

	Point3d triangulate3Dpoint(Mat_<double> P1,Mat_<double> P2,Point2d u,Point2d u1);
	bool isRTcorrect(Mat_<double> R,Mat_<double> T);
	bool DecomposeEtoRandT(Mat E,Mat_<double> &R1,Mat_<double> &R2,Mat_<double> &T1,Mat_<double> &T2);
	void getPointsonPolarline(vector<Point2d> &PointSet1,vector<Point2d> &PointSet2,Mat_<double> F,double T);
	void findRobustFundamentalMat(vector<Point2d> PointSet1,vector<Point2d> PointSet2);
	void FindPoseEstimation(Mat_<double> &R,Mat_<double> &T);

	void FindCameraMatrices(vector<Point2d> pointSet1,vector<Point2d> pointSet2, Mat_<int> rgb, bool isFirstPair);

	void getNewFramePointLabels(vector<int> &oldPointLabels, vector<int> &newPointLabels);


public:
	vector<frameInfo> *frameLists;
	vector<cloudPoint> *objectClouds;
	vector<int> dividePointLabels;
	vector<int> newPointLabel;
	vector<pointInfo> newFrameoldPoints;          //used to estimate the pose of camera

	vector<Point2d> testPoint1,testPoint2;
	Mat_<double> Fmatrice;      //fundamental matrix
	Mat_<double> K;       //camera internal matrix
	Mat_<int> RGB;

	int maxPointLabel;
	int frameNo;      //the index of the frame which will contain all its points after this iteration
};
