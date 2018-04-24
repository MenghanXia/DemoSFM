#include "featureTrack.h"


bool Track::keyFramePairs()
{
	*capture >> curFrame;
	if (curFrame.empty())
	{
		return false;
	}
	Size imgSize(width, height);
	resize(curFrame, curFrame, imgSize);
	Mat gray_cur, gray_pre;
	cvtColor(preFrame, gray_pre, CV_BGR2GRAY);
	cvtColor(curFrame, gray_cur, CV_BGR2GRAY);
	vector<Point2f> prePoints, curPoints;
	for (int i = 0; i < curFramePtSet.size(); i ++)
	{
		Point2f point;
		point.x = curFramePtSet[i].coord.x;
		point.y = curFramePtSet[i].coord.y;
		prePoints.push_back(point);
	}
	calcOpticalFlowPyrLK(gray_pre, gray_cur, prePoints, curPoints, status, err);
	int i, j;
	vector<PointInfo> PointSet;
	vector<Point2f> showPts1, showPts2;
	for (i = 0; i < curFramePtSet.size(); i ++)
	{
		double dist = fabs(prePoints[i].x-curPoints[i].x)+fabs(prePoints[i].y-curPoints[i].y);
		if (status[i] && dist < maxDist)
		{
			PointInfo point;
			point.PtNo = curFramePtSet[i].PtNo;
			point.coord.x = curPoints[i].x;
			point.coord.y = curPoints[i].y;

			PointSet.push_back(point);

			showPts1.push_back(prePoints[i]);
			showPts2.push_back(curPoints[i]);
		}
	}
	if (showPts1.size() == 0)
	{
		cout<<"The tracking point protect distance is not big enough!"<<endl;
		cout<<"You can solve this by setting PROTECT_RATIO as a bigger value!"<<endl;
		exit(1);
	}
	showFeatures(showPts1, showPts2);
	curFramePtSet = PointSet;
	preFrame = curFrame;  //set current frame as previous frame
	frameNo ++;
	return true;
}


bool Track::isKeyFrame()
{
	if (frameNo%KEY_FRAME_GAP == 0)
	{
		pointSet1.clear();
		pointSet2.clear();
		int i, j;
		for (i = 0; i < keyFramePtSet.size(); i ++)
		{
			int PtNo = keyFramePtSet[i].PtNo;
			for (j = 0; j < curFramePtSet.size(); j ++)
			{
				if (PtNo == curFramePtSet[j].PtNo)
				{
					pointSet1.push_back(keyFramePtSet[i].coord);
					pointSet2.push_back(curFramePtSet[j].coord);
					break;
				}
			}
		}
		vector<int> bareSubAreaIndexList;
		if (shouldAddPoints(bareSubAreaIndexList))
		{
			addNewFeaturePts(bareSubAreaIndexList);
		}
		keyFramePtSet = curFramePtSet;

		return true;
	}
	else
	{
		return false;
	}
}


void Track::intialization()
{
	capture = new VideoCapture(videoPath);
	maxCount = 1500;
	newMaxCount = 500;
	qLevel = 0.01;	
	minDist = 5.0;
	maxPtNo = 0;
	if(capture->isOpened())	 //read the file switch
	{
		*capture >> preFrame;
		int Row = preFrame.rows, Col = preFrame.cols;
		height = sizeRatio*Row;
		width = sizeRatio*Col;
		Size imgSize(width, height);
		resize(preFrame, preFrame, imgSize);

		Mat grayImage;
		cvtColor(preFrame, grayImage, CV_BGR2GRAY);
		vector<Point2f> points;
		goodFeaturesToTrack(grayImage, points, maxCount, qLevel, minDist);
		for (int i = 0; i < points.size(); i ++)
		{
			PointInfo temp;
			temp.PtNo = maxPtNo++;
			temp.coord = points[i];
			keyFramePtSet.push_back(temp);
		}
		curFramePtSet = keyFramePtSet;
		maxDist = min(height,width)*PROTECT_RATIO;
		frameNo ++;
	}
}


void Track::addNewFeaturePts(vector<int> subAreaIndexList)
{
	int i, j;
	Mat mask(height, width, CV_8U, Scalar(0));
	Mat newfeature_gray;               //areas where new features should be got
	newfeature_gray.setTo(0);
	Mat gray_cur;
	cvtColor(curFrame, gray_cur, CV_BGR2GRAY);
	for (i = 0; i < subAreaIndexList.size(); i ++)
	{
		int index = subAreaIndexList[i];
		int subWidth = width/3, subHeight = height/3;
		int iniX = index%3, iniY = index/3;
		iniX *= subWidth;
		iniY *= subHeight;
		Mat specified(mask, Rect(iniX, iniY, subWidth, subHeight));
		specified.setTo(1);
		gray_cur.copyTo(newfeature_gray,mask);
	}
	vector<Point2f> initialfeature;
	goodFeaturesToTrack(newfeature_gray, initialfeature, newMaxCount, qLevel, minDist);
	for (i = 0; i < initialfeature.size(); i ++)     //adding new feature
	{
		if ((initialfeature[i].x >= width/3-1 && initialfeature[i].x <= width/3+1)
			|| (initialfeature[i].x >= width*2/3-1 && initialfeature[i].x <= width*2/3+1)
			|| (initialfeature[i].y >= height/3-1 && initialfeature[i].y <= height/3+1)
			|| (initialfeature[i].y >= height*2/3-1 && initialfeature[i].y <= height*2/3+1))
		{
			continue;
		}
		bool isOldPt = false;
		for (j = 0; j < curFramePtSet.size(); j ++)    // avoiding repeated points
		{
			Point2d oldPoint = curFramePtSet[j].coord;
			if (initialfeature[i].x == oldPoint.x && initialfeature[i].y == oldPoint.y)
			{
				isOldPt = true;
				break;
			}
		}
		if (isOldPt == false)
		{
			PointInfo newPoint;
			newPoint.PtNo = maxPtNo++;
			newPoint.coord.x = initialfeature[i].x;
			newPoint.coord.y = initialfeature[i].y;
			curFramePtSet.push_back(newPoint);
		}
	}
}


bool Track::shouldAddPoints(vector<int> &subAreaIndexList)
{
	int pointDistribute[9] = {0};
	int i, j;
	int cols = width, rows = height;
	for (i = 0; i < curFramePtSet.size(); i ++)
	{
		Point2d testPoint = curFramePtSet[i].coord;
		if (testPoint.x < cols/3)
		{
			if (testPoint.y < rows/3)
			{
				pointDistribute[0]++;
			}
			else if (testPoint.y >= rows*2/3)
			{
				pointDistribute[6]++;
			}
			else
			{
				pointDistribute[3]++;
			}
		}
		else if (testPoint.x >= cols*2/3)
		{
			if (testPoint.y < rows/3)
			{
				pointDistribute[2]++;
			}
			else if (testPoint.y >= rows*2/3)
			{
				pointDistribute[8]++;
			}
			else
			{
				pointDistribute[5]++;
			}
		}
		else
		{
			if (testPoint.y < rows/3)
			{
				pointDistribute[1]++;
			}
			else if (testPoint.y >= rows*2/3)
			{
				pointDistribute[7]++;
			}
			else
			{
				pointDistribute[4]++;
			}
		}
	}
	for (i = 0; i < 9; i ++)
	{
		if (pointDistribute[i] > POINT_THRESHOLD)
		{
			continue;
		}
		subAreaIndexList.push_back(i);
	}
	if (subAreaIndexList.size() == 0)
	{
		return false;
	}
	return true;
}


void Track::outputMatchPair(vector<Point2d> &PtSet1, vector<Point2d> &PtSet2, Mat_<int> &RGB)
{
	PtSet1 = pointSet1;
	PtSet2 = pointSet2;
	int PtNum = pointSet1.size();
	RGB = Mat(3, PtNum, CV_16UC1, Scalar(0));
	for (int i = 0; i < PtNum; i ++)
	{
		RGB(0,i) = 255;  //red point
		RGB(1,i) = 255;
		RGB(2,i) = 255;
	}
}


void Track::getFrameSize(int &rows, int &cols)
{
	rows = height;
	cols = width;
}


void Track::showFeatures(vector<Point2f> showPts1, vector<Point2f> showPts2)
{
	Mat image;
	curFrame.copyTo(image);
	for (int i = 0; i < showPts1.size(); i ++)
	{
		circle(image, showPts2[i], 2, Scalar(0,0,255), -1);
		line(image, showPts2[i], showPts1[i], Scalar(0,255,0), 1);
	}
	imshow("trackMap", image);
	if (frameNo%KEY_FRAME_GAP == 0)
	{
		char name[512];
		sprintf(name,"data/keyFrames/keyframe%d.jpg", frameNo);
		String fileName = name;
		imwrite(fileName, curFrame);
	}
}


void Track::imageSequence2Video(vector<String> fileNameList, String videoSavePath, double fps)
{
	int frameCount=0;
	int fourcc = CV_FOURCC('M','J','P','G');
	Mat test = imread(fileNameList[0]);
	Size frameSize(test.cols, test.rows);
	VideoWriter pWriter(videoSavePath, fourcc, fps, frameSize);
	int i, j;
	for (i = 0; i < fileNameList.size(); i ++)
	{
		String fileName = fileNameList[i];
		Mat frame = imread(fileName);
		pWriter.write(frame);
	}
	cout<<"Image Sequence convert to Video done!"<<endl;
}
