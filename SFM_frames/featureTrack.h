#pragma once
#include "dataStruct.h"
#include <opencv2/video/video.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp> 
#include "opencv2/nonfree/nonfree.hpp" 

using namespace std;
using namespace cv;

#define KEY_FRAME_GAP 3
#define POINT_THRESHOLD 50
#define PROTECT_RATIO 0.08    //default : 0.03 for normal video

struct PointInfo
{
	int PtNo;
	Point2d coord;
};

class Track
{
public:
	Track(String videoPath_, double sizeRatio_)  /////construction function 1
	{
		videoPath = videoPath_;
		sizeRatio = sizeRatio_;
		if (sizeRatio == 1.0)    //OpenCV resize function make the frame good to track!
		{
			sizeRatio -= 0.001;
		}
		frameNo = 0;
		intialization();
	};
	Track(vector<String> fileNameList, double sizeRatio_)  /////construction function 2
	{
		String videoPath_ = "data/InputFiles/synth.avi";
		double fps = 10;
		imageSequence2Video(fileNameList, videoPath_, fps);
		videoPath = videoPath_;
		sizeRatio = sizeRatio_;
		if (sizeRatio == 1.0)    //OpenCV resize function make the frame good to track!
		{
			sizeRatio -= 0.001;
		}
		frameNo = 0;
		intialization();
	};
public:
	bool keyFramePairs();
	void intialization();
	bool shouldAddPoints(vector<int> &subAreaIndexList);
	void addNewFeaturePts(vector<int> subAreaIndexList);
	bool isKeyFrame();
	void showFeatures(vector<Point2f> showPts1, vector<Point2f> showPts2);

	void imageSequence2Video(vector<String> fileNameList, String videoSavePath, double fps);
	void outputMatchPair(vector<Point2d> &PtSet1, vector<Point2d> &PtSet2, Mat_<int> &RGB);
	void getFrameSize(int &rows, int &cols);

private:
	int width;
	int height;
	double sizeRatio;
	String videoPath;
	VideoCapture *capture;
	int frameNo;
	Mat preFrame;
	Mat curFrame;
	vector<Point2d> pointSet1;
	vector<Point2d> pointSet2;

	int maxPtNo;
	vector<PointInfo> keyFramePtSet;
	vector<PointInfo> curFramePtSet;
	int maxCount;	
	int newMaxCount;
	int maxDist;
	double qLevel;	
	double minDist;
	vector<uchar> status;
	vector<float> err;
};