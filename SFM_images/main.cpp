#pragma once
#include "dataStruct.h"
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
void getInlierParameters(Mat_<double> &cameraMatice,vector<double> &distCoeffs);
void cloudRadiusFilter(vector<Point3d> inputSet, double radius, int thresholdSum,vector<uchar> &status);
void findSamePoints(vector<Point2d> basePoints,vector<Point2d> queryPoints);
void save3dPoint(string path_,vector<cloudPoint> Point3dLists,vector<uchar> status);
void rotMatrixToQuternion(Mat_<double> rotMatrix,vector<double> &quaternion);
void savetoSBA(string path_,vector<frameInfo> framelists,vector<cloudPoint> Point3dLists,vector<uchar> status);
void drawPoseTrack(vector<Point2d> positions);

void main()
{ 
	char* folderPath = "data";
	vector<string> framePaths;
	string path_match = folderPath;
	path_match += "/Match_Result";
	framePaths = get_filelist(path_match);
	
	///////////////////////////////////global data variable
	vector<frameInfo> frameLists;          //total frame data  
	vector<cloudPoint> objectClouds;             //object 3D points
	Mat_<double> cameraMatrix;
	vector<double> distCoeffs;
	getInlierParameters(cameraMatrix,distCoeffs);     //get the inlier parameters of camera

	int rows = 846;
	int cols = 1506;
	Size imageSize = Size(cols,rows);
	readMatch obj_Matcher(path_match);
	ProjectMat obj_ProjectMat(cameraMatrix,&frameLists,&objectClouds);
	Triangulation obj_Triangulation(cameraMatrix,imageSize,&frameLists,&objectClouds);
	/////////////////////////////////processing steps
	int imageNum = framePaths.size()+1;
	vector<Point2d> pointSet1, pointSet2;
	vector<int> newPointLabels,oldPointLabels;
	frameInfo firstFrame;
	firstFrame.no = 0;
	firstFrame.ProjectMatrix = Mat(Matx34d( 1, 0, 0, 0,  
		                                    0, 1, 0, 0,  
		                                    0, 0, 1, 0));
	frameLists.push_back(firstFrame);
	unsigned i,j;
	Mat_<int> RGB;
	bool isFirstPair = true;
	clock_t start_time, end_time;
	start_time = clock();
	int imgNum = 3;
	for (i = 0; i < imgNum-1; i ++)
	{	
		cout<<"matching image"<<i<<" & image"<<i+1<<endl;
		bool isvalid = obj_Matcher.getMatchResult(i, i+1, pointSet1, pointSet2, RGB);
		if (!isvalid)
		{
			return;
		}
		cout<<"matched "<<pointSet1.size()<<" pair of points\n"<<endl;

		cout<<"finding project matrix..."<<endl;
		obj_ProjectMat.FindCameraMatrices(pointSet1,pointSet2,RGB,isFirstPair);	
		cout<<"done!\n"<<endl;
	
		cout<<"projecting 3D points..."<<endl;
		oldPointLabels.clear();
		newPointLabels.clear();
		obj_ProjectMat.getNewFramePointLabels(oldPointLabels,newPointLabels);

		obj_Triangulation.getPointsfromMultiview(i+1,isFirstPair);
		vector<Point3d> newObjects;
		obj_Triangulation.addNewPoint(newPointLabels,newObjects);
		cout<<"adding "<<newObjects.size()<<" points"<<endl;

		vector<int> relateFrames;
		vector<vector<Point2d> > imgPointList1,imgPointList2;

		if (i == 118 && obj_Triangulation.checkRepeatStruct(oldPointLabels,relateFrames))
		{
			Mat_<int> tempRGB;
			for (j = 0; j < relateFrames.size(); j ++)                
			{
				vector<Point2d> tempImgPt1,tempImgPt2;
				obj_Matcher.getMatchResult(i+1,relateFrames[j],tempImgPt1,tempImgPt2,tempRGB);
				imgPointList1.push_back(tempImgPt1);
				imgPointList2.push_back(tempImgPt2);
			}
			obj_Triangulation.modifycurSaM(imgPointList1,imgPointList2,i+1);            //Warning: here supposing relateFrames has only one element
		}

		cout<<"AVE error: "<<obj_Triangulation.meanReprojectError(0,i,0.2)<<endl;
		cout<<"done!\n"<<endl;
		isFirstPair = false;
	}

	cout<<"reconstructed "<<objectClouds.size()<<" points included "<<obj_Triangulation.sumOutlier<<" outliers "<<endl;

	vector<Point3d> origin_Points;
	vector<Point3d> filter_Points;
	for (i = 0; i < objectClouds.size(); i ++)
	{
		Point3d tempPoint = objectClouds[i].objectPoint;
		origin_Points.push_back(tempPoint);
	}
	string path = folderPath;
	vector<uchar> status;
	cloudRadiusFilter(origin_Points, 2.0, 50, status);
	for (i = 0; i < status.size(); i ++)
	{
		status[i] = 1;
	}

	savetoSBA(path,frameLists,objectClouds,status);
	path += "/cloudPoints.txt";
	save3dPoint(path,objectClouds,status);
	
	end_time=clock();
	cout<<"The total process has consumed "<<(end_time-start_time)/CLOCKS_PER_SEC<<" seconds! "<<endl;
}


void drawPoseTrack(vector<Point2d> positions)
{
	int i;
	double max_X = 0,max_Y = 0;
	double min_X = 999,min_Y = 999;
	Point2d centre = Point2d(0,0);
	for (i = 0; i < positions.size(); i ++)
	{
		Point2d tempPoint = positions[i];
		if (max_X < tempPoint.x)
		{
			max_X = tempPoint.x;
		}
		if (min_X > tempPoint.x)
		{
			min_X = tempPoint.x;
		}
		if (max_Y < tempPoint.y)
		{
			max_Y = tempPoint.y;
		}
		if (min_Y > tempPoint.y)
		{
			min_Y = tempPoint.y;
		}
		centre += tempPoint;
	}
	centre.x /= positions.size();
	centre.y /= positions.size();
	int width = int(max_X-min_X+0.5);
	int height = int(max_Y-min_Y+0.5);

	double scale = 500/(max(width,height));

	vector<Point2d> newPositions;
	for (i = 0; i < positions.size(); i ++)
	{
		Point2d temp = (positions[i]-centre) * scale;
		newPositions.push_back(temp);
	}
	int rows = 600, cols = 600;
	Point2d offset = Point2d(300,300);
	Mat image(rows,cols,CV_8UC3,Scalar(255,255,255));
//	circle(image,newPositions[0]+offset,4,Scalar(0,255,0),-1);
	circle(image,newPositions[0]+offset,6,Scalar(255,0,0),-1);
	for (i =  1; i < positions.size()-1; i ++)
	{
		line(image,newPositions[i-1]+offset,newPositions[i]+offset,Scalar(0,0,255),2);
		circle(image,newPositions[i]+offset,4,Scalar(255,0,0),-1);
		circle(image,newPositions[i]+offset,6,Scalar(0,255,255),2);
	}
	line(image,newPositions[positions.size()-2]+offset,newPositions[positions.size()-1]+offset,Scalar(0,0,255),2);
//	circle(image,newPositions[positions.size()-1]+offset,4,Scalar(0,255,0),-1);
	circle(image,newPositions[positions.size()-1]+offset,6,Scalar(0,255,255),-1);
	imwrite("C:/Users/daddy_000/Desktop/Video2SaM/data/CloudPoints/cameraCircle.jpg",image);
}


void getInlierParameters(Mat_<double> &cameraMatrice,vector<double> &distCoeffs)
{
	//cameraMatrice = Mat(Matx33d(1103.792, 0.000000, 620.124,                 //Castal_s        
	//	                        0.000000, 1105.664, 416.376,
	//	                        0.000000, 0.000000, 1.000000));   

	cameraMatrice = Mat(Matx33d(919.826, 0, 506.897,
		0, 921.387, 335.603,
		0, 0, 1));

	distCoeffs.push_back(0.0);              //indoor coeffs
	distCoeffs.push_back(0.0);
	distCoeffs.push_back(0.0);
	distCoeffs.push_back(0.0);
	distCoeffs.push_back(0.0);
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


void save3dPoint(string path_,vector<cloudPoint> Point3dLists,vector<uchar> status)
{
	const char * path = path_.c_str();
	FILE* fp = fopen(path,"w");
//	fprintf(fp,"%d\n",Point3dLists.size());
	int no = 0;
	for (unsigned int i = 0; i < Point3dLists.size(); i ++)
	{
		if (Point3dLists[i].objectPoint != Point3d(0,0,0))
		{
			
			if (status[no] == 0)
			{
				no ++;
				continue;
			}
			fprintf(fp,"%lf %lf %lf %d %d %d\n",Point3dLists[i].objectPoint.x,Point3dLists[i].objectPoint.y,Point3dLists[i].objectPoint.z,Point3dLists[i].rgb[0],Point3dLists[i].rgb[1],Point3dLists[i].rgb[2]);
//			fprintf(fp,"%lf %lf %lf\n",Point3dLists[i].objectPoint.x,Point3dLists[i].objectPoint.y,Point3dLists[i].objectPoint.z);
			no ++;
		}
	}
	fclose(fp);
	cout<<"clouds of points are saved!"<<endl;
}


vector<string> get_filelist(string foldname)
{
	foldname += "/*.*";
	const char * mystr=foldname.c_str();
	vector<string> flist;
	string lineStr;
	vector<string> extendName;
	extendName.push_back("txt");
	extendName.push_back("TXT");

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
		for (int i = 0; i < 2; i ++)       //ÅÅ³ý·ÇÍ¼ÏñÎÄ¼þ
		{
			if (lineStr.find(extendName[i]) < 999)
			{
				flist.push_back(lineStr);
				break;
			}
		}	
	}
	return flist;
}


void savetoSBA(string path_,vector<frameInfo> framelists,vector<cloudPoint> Point3dLists,vector<uchar> status)
{
	unsigned int i,j,k;
	string path1,path2;
	path1 = path2 = path_;
	path1 += "/pts.txt";
	const char * path;
	path = path1.c_str();
	FILE* fp = fopen(path,"w");
	vector<int> frameSet;
/*	for (i = 0; i < Point3dLists.size(); i ++)
	{
		if (status[i] == 0)
		{
			continue;
		}
		fprintf(fp,"%lf %lf %lf  ",Point3dLists[i].objectPoint.x,Point3dLists[i].objectPoint.y,Point3dLists[i].objectPoint.z);
		fprintf(fp,"%d ",Point3dLists[i].frameIndex.size());
		frameSet = Point3dLists[i].frameIndex;
		for (j = 0; j < frameSet.size(); j ++)
		{
			for (int j1 = j+1; j1 < frameSet.size()-1; j1 ++)
			{
				if (frameSet[j] == frameSet[j1])
				{
					cout<<"repetitive frameIndex:"<<frameSet[j]<<endl;;
				}
			}
		}
		for (j = 0; j < frameSet.size(); j ++)
		{
			fprintf(fp,"%d ",frameSet[j]);
			int no = frameSet[j];
			vector<pointInfo> pointSet = framelists[no].pointSet;
			for (k = 0; k < pointSet.size(); k ++)
			{
				if (pointSet[k].no == i)
				{
					fprintf(fp,"%lf %lf ",pointSet[k].imagePoint.x,pointSet[k].imagePoint.y);
					break;
				}
			}
		}
		fprintf(fp,"\n");
	}
	fclose(fp);*/

	path2 += "/cams.txt";
	path = path2.c_str();
	fp = fopen(path,"w");
	Mat_<double> R,T;
	vector<double> quternion;      
	vector<Point2d> positions;
	for (i = 0; i < framelists.size(); i ++)
	{   
		R = Mat(framelists[i].ProjectMatrix,Rect(0,0,3,3));
		T = Mat(framelists[i].ProjectMatrix,Rect(3,0,1,3));
		Mat_<double> Treal = -R.inv() * T;
		fprintf(fp,"%lf %lf %lf\n%lf %lf %lf\n%lf %lf %lf\n\n",R(0,0),R(0,1),R(0,2),R(1,0),R(1,1),R(1,2),R(2,0),R(2,1),R(2,2));
		fprintf(fp,"%lf %lf %lf\n\n",T(0),T(1),T(2));
		positions.push_back(Point2d(Treal(0),Treal(2)));
		//rotMatrixToQuternion(R,quternion);
		//fprintf(fp,"%lf %lf %lf %lf %lf %lf %lf\n",quternion[0],quternion[1],quternion[2],quternion[3],T(0),T(1),T(2));
	}
	fclose(fp);
	drawPoseTrack(positions);
	cout<<"result are saved!"<<endl;
}


void rotMatrixToQuternion(Mat_<double> rotMatrix,vector<double> &quaternion)
{
	quaternion.clear();
	float tr = rotMatrix(0,0) + rotMatrix(1,1) + rotMatrix(2,2);
	float temp = 0.0;
	double x,y,z,w;
	if(tr > 0.0)
	{
		temp = 0.5f / sqrtf(tr+1);
		w = 0.25f / temp; 
		x = (rotMatrix(1,2) - rotMatrix(2,1)) * temp;
		y = (rotMatrix(2,0) - rotMatrix(0,2)) * temp;
		z = (rotMatrix(0,1) - rotMatrix(1,0)) * temp;
	}
	else
	{
		if(rotMatrix(0,0) > rotMatrix(1,1) && rotMatrix(0,0) > rotMatrix(2,2))
		{
			temp = 2.0f * sqrtf(1.0f + rotMatrix(0,0) - rotMatrix(1,1) - rotMatrix(2,2));
			w = (rotMatrix(2,1) - rotMatrix(1,2)) / temp;
			x = 0.25f * temp;
			y = (rotMatrix(0,1) + rotMatrix(1,0)) / temp;
			z = (rotMatrix(0,2) + rotMatrix(2,0)) / temp;
		}
		else if( rotMatrix(1,1) > rotMatrix(2,2))
		{
			temp = 2.0f * sqrtf(1.0f + rotMatrix(1,1) - rotMatrix(0,0) - rotMatrix(2,0));
			w = (rotMatrix(0,2) - rotMatrix(2,0)) / temp;
			x = (rotMatrix(0,1) + rotMatrix(1,0)) / temp;
			y =  0.25f * temp;
			z = (rotMatrix(1,2) + rotMatrix(2,1)) / temp;
		}
		else
		{
			temp = 2.0f * sqrtf(1.0f + rotMatrix(2,2) - rotMatrix(0,0) - rotMatrix(1,1));
			w = (rotMatrix(1,0) - rotMatrix(0,1)) / temp;
			x = (rotMatrix(0,2) + rotMatrix(2,0)) / temp;
			y = (rotMatrix(1,2) + rotMatrix(2,1)) / temp;
			z = 0.25f * temp;
		}
	}
	quaternion.push_back(w);
	quaternion.push_back(-x);
	quaternion.push_back(-y);
	quaternion.push_back(-z);
}
