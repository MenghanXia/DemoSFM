#pragma once
#include "dataStruct.h"
#include "featureTrack.h"
#include "ProjectMatrix.h"
#include "triangulation.h"
#include <string>
#include <iostream>
#include <fstream>
#include <time.h>
#include <windows.h>

using namespace std;
using namespace cv;

vector<string> get_filelist(string foldname);
void getInlierParameters(String intrinsics, Mat_<double> &cameraMatice,vector<double> &distCoeffs, double resizeRatio);
void cloudRadiusFilter(vector<Point3d> inputSet, double radius, int thresholdSum,vector<uchar> &status);
void saveSFMResults(string savePath,vector<frameInfo> framelists,vector<cloudPoint> Point3dLists,vector<uchar> status);


void main()
{ 
	String folderPath = "data";    //data root directory
	vector<frameInfo> frameLists;
	vector<cloudPoint> objectClouds;  
	Mat_<double> cameraMatrix = Mat(3,3,CV_64FC1,Scalar(0));
	vector<double> distCoeffs;
	String intrinsics = folderPath + "/InputFiles/intrinsic.txt";
	double resizeRatio = 1.0;   //Image Resize Ratio
	getInlierParameters(intrinsics,cameraMatrix,distCoeffs, resizeRatio);     //get the inlier parameters of camera
	cout<<"Camera Intrinsic:"<<cameraMatrix<<endl;
//==========================IMAGE SEQUENCE INPUT===============================//
	//String imgSeqPath = "data/InputFiles/frames";
	//vector<String> imgSeqPathList = get_filelist(imgSeqPath);
	//Track trackObj(imgSeqPathList, resizeRatio);
	//int rows, cols;
	//trackObj.getFrameSize(rows, cols);

//==============================VIDEO INPUT==================================//
	String videoPath = "data/InputFiles/synth.avi"; 
	Track trackObj(videoPath, resizeRatio);
	int rows, cols;
	trackObj.getFrameSize(rows, cols);
	
	Size imageSize = Size(cols,rows);
	ProjectMat obj_ProjectMat(cameraMatrix,&frameLists,&objectClouds);
	Triangulation obj_Triangulation(cameraMatrix,imageSize,&frameLists,&objectClouds);
	/////////////////////////////////processing steps
	frameInfo firstFrame;
	firstFrame.no = 0;
	firstFrame.ProjectMatrix = Mat(Matx34d( 1, 0, 0, 0,  
		                                    0, 1, 0, 0,  
		                                    0, 0, 1, 0));
	frameLists.push_back(firstFrame);
	Mat_<int> RGB;
	bool isFirstPair = true;
	clock_t start_time, end_time;
	start_time = clock();
	int keyFrameNo = 0;
	int cnt = 0;
	cout<<"Video SFM is Starting<"<<rows<<"x"<<cols<<"> ..."<<endl;
	while(1)
	{
		bool notEnd = trackObj.keyFramePairs();
		waitKey(10);
		if (!notEnd)
		{
			cout<<"The End of Video Flow!"<<endl;
			break;
		}
		if (trackObj.isKeyFrame())
		{
			keyFrameNo ++;     // new added key frame No.
			vector<Point2d> PtSet1, PtSet2;
			trackObj.outputMatchPair(PtSet1, PtSet2, RGB);
			cout<<"#1-Tracking Features..."<<endl;
			cout<<"Got"<<PtSet1.size()<<" Points between Key-Frame Pair "<<keyFrameNo<<endl<<endl;
			
			cout<<"#2-Finding project matrix..."<<endl;
			obj_ProjectMat.FindCameraMatrices(PtSet1, PtSet2, RGB, isFirstPair);
			cout<<"done!\n"<<endl;

			cout<<"#3-Projecting 3D points..."<<endl;
			vector<int> newPointLabels, oldPointLabels;
			obj_ProjectMat.getNewFramePointLabels(oldPointLabels, newPointLabels);
			obj_Triangulation.getPointsfromMultiview(keyFrameNo, isFirstPair);
			vector<Point3d> newObjects;
			obj_Triangulation.addNewPoint(newPointLabels, newObjects);
			cout<<"adding "<<newObjects.size()<<" points"<<endl;

			cout<<"Current Global-RMS: "<<obj_Triangulation.meanReprojectError(0,keyFrameNo-1,0.2)<<endl;
			cout<<"done!\n"<<endl;
			isFirstPair = false;
		}

		
	}
	cout<<"reconstructed "<<objectClouds.size()<<" points included "<<obj_Triangulation.sumOutlier<<" outliers "<<endl;

	vector<Point3d> origin_Points;
	vector<Point3d> filter_Points;
	vector<uchar> status;
	for (int i = 0; i < objectClouds.size(); i ++)
	{
		Point3d tempPoint = objectClouds[i].objectPoint;
		origin_Points.push_back(tempPoint);
		status.push_back(1);    //initialize status value as good point
	}
//	cloudRadiusFilter(origin_Points, 2.0, 50, status);

	String savePath = folderPath;
	saveSFMResults(savePath,frameLists,objectClouds,status);

	end_time=clock();
	cout<<"The total process has consumed "<<(end_time-start_time)/CLOCKS_PER_SEC<<" seconds! "<<endl;

}


void getInlierParameters(String intrinsics, Mat_<double> &cameraMatrice, vector<double> &distCoeffs, double resizeRatio)
{
	FILE *fp = fopen(intrinsics.c_str(), "r");
	if (fp == nullptr)
	{
		cout<<"Invalid Intrinsic File!"<<endl;
		return;
	}
	int i, j;
	for (i = 0; i < 3; i ++)    //read camera intrinsic matrix
	{
		double a, b, c;
		fscanf(fp, "%lf%lf%lf", &a,&b,&c);
		cameraMatrice(i,0) = a;
		cameraMatrice(i,1) = b;
		cameraMatrice(i,2) = c;
	}
	cameraMatrice(0,0) *= resizeRatio;
	cameraMatrice(1,1) *= resizeRatio;
	cameraMatrice(0,2) *= resizeRatio;
	cameraMatrice(1,2) *= resizeRatio;
	int coeffNum = 0;
	fscanf(fp, "%d", &coeffNum);
	for (i = 0; i < coeffNum; i ++)   //read distort coefficients
	{
		double coeff = 0.0;
		fscanf(fp, "%lf", &coeff);
		distCoeffs.push_back(coeff);
	}
	fclose(fp);
	cout<<"Intrinsic Parameters Read Over!"<<endl;
}


void saveSFMResults(string savePath, vector<frameInfo> framelists, vector<cloudPoint> Point3dLists, vector<uchar> status)
{
	string PtSetPath, CamListPath;
	PtSetPath = savePath + "/CloudPoints.txt";
	CamListPath = savePath + "/CamPoses.txt";
	FILE* fp = fopen(PtSetPath.c_str(), "w");
	vector<int> frameSet;
	int i, j, k;
	int filterNum = 0;
	for (i = 0; i < Point3dLists.size(); i ++)
	{
		if (status[i] == 0)
		{
			filterNum ++;
			continue;
		}
		fprintf(fp,"%lf %lf %lf ",Point3dLists[i].objectPoint.x,Point3dLists[i].objectPoint.y,Point3dLists[i].objectPoint.z);
		int *colorHeader = Point3dLists[i].rgb;
		fprintf(fp,"%d %d %d\n", colorHeader[0],colorHeader[1],colorHeader[2]);
	}
	fclose(fp);

	fp = fopen(CamListPath.c_str(), "w");
	Mat_<double> R, T;    
	vector<Point3d> positions;
	for (i = 0; i < framelists.size(); i ++)
	{   
		R = Mat(framelists[i].ProjectMatrix,Rect(0,0,3,3));
		T = Mat(framelists[i].ProjectMatrix,Rect(3,0,1,3));
		Mat_<double> Treal = -R.inv() * T;
		fprintf(fp,"%lf %lf %lf\n%lf %lf %lf\n%lf %lf %lf\n\n",R(0,0),R(0,1),R(0,2),R(1,0),R(1,1),R(1,2),R(2,0),R(2,1),R(2,2));
		fprintf(fp,"%lf %lf %lf\n\n",T(0),T(1),T(2));
		positions.push_back(Point3d(Treal(0),Treal(1),Treal(2)));
	}
	fclose(fp);
	string camPositions = savePath + "/Cam-positions.txt";
	FILE *fp1 = fopen(camPositions.c_str(), "w");
	for (i = 0; i < positions.size(); i ++)
	{
		fprintf(fp1, "%lf %lf %lf\n", positions[i].x, positions[i].y, positions[i].z);
	}
	fclose(fp1);
	cout<<"Result are saved!<filterd "<<filterNum<<" Points>"<<endl;
}


void cloudRadiusFilter(vector<Point3d> inputSet, double radius, int thresholdSum,vector<uchar> &status)
{
	clock_t start_time, end_time;
	start_time = clock();
	int i, j;
	vector<Point3d> realInput;
	for (i = 0; i < inputSet.size(); i ++)
	{
		if (inputSet[i] != Point3d(0,0,0))
		{
			realInput.push_back(inputSet[i]);
		}
	}
	int num = realInput.size();
	Mat queryPoints(num, 3, CV_32FC1);
	for (i = 0; i < num; i ++)
	{
		queryPoints.at<float>(i, 0) = realInput[i].x;
		queryPoints.at<float>(i, 1) = realInput[i].y;
		queryPoints.at<float>(i, 2) = realInput[i].z;
	}
	flann::Index flannIndex(queryPoints, cv::flann::KDTreeIndexParams(4));
	bool isGood;
	status.clear();
	int no = 0;
	for (i = 0; i < num; i ++)
	{
		isGood = true;
		status.push_back(0);
		Mat indices(1, thresholdSum, CV_32SC1, Scalar::all(-1) );
		Mat dists(1, thresholdSum, CV_32FC1, Scalar::all(-1) );
		flannIndex.radiusSearch( queryPoints.row(i), indices, dists, radius, thresholdSum, cv::flann::SearchParams(64));
		for (j = 0; j < thresholdSum; j ++)
		{
			if (dists.at<float>(0,j) < 0)
			{
				isGood = false;
				break;
			}
		}
		if (isGood)
		{
			Point3d goodPoint = inputSet[i];
			status[i] = 1;
			no ++;
		}
	}
	end_time=clock();
	cout<<"Filtered out "<<(realInput.size()-no)<<" points(used "<<(end_time-start_time)/CLOCKS_PER_SEC<<" seconds)"<<endl;
}


vector<string> get_filelist(string foldname)
{
	String foldname1 = foldname;
	foldname += "/*.*";
	const char * mystr=foldname.c_str();
	vector<string> flist;
	string lineStr;
	vector<string> extendName;
	extendName.push_back("jpg");
	extendName.push_back("JPG");
	extendName.push_back("png");
	extendName.push_back("bmp");
	extendName.push_back("tif");

	HANDLE file;
	WIN32_FIND_DATA fileData;
	char line[1024];
	wchar_t fn[1000];
	mbstowcs(fn,mystr,999);
	file = FindFirstFile(fn, &fileData);
	FindNextFile(file, &fileData);
	while(FindNextFile(file, &fileData))
	{
		wcstombs(line,(const wchar_t*)fileData.cFileName,259);
		lineStr = line;
		for (int i = 0; i < 5; i ++)       //ÅÅ³ý·ÇÍ¼ÏñÎÄ¼þ
		{
			if (lineStr.find(extendName[i]) < 999)
			{
				lineStr = foldname1 + "/" + lineStr;
				flist.push_back(lineStr);
				break;
			}
		}	
	}
	return flist;
}




