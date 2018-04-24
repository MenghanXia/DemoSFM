#include "match.h"


bool readMatch::getMatchResult(int frame1,int frame2,vector<Point2d> &pointSet1,vector<Point2d> &pointSet2,Mat_<int> &RGB)
{
	pointSet1.clear();
	pointSet2.clear();
	int i,j;
	int pointNum,temp;
	Point2d temp1,temp2;
	FILE* fp;
	char fileName[1024];
	const char * path = foldPath.c_str();
	sprintf_s(fileName,"%s/result%d&%d.txt",path,frame1,frame2);
	fp = fopen(fileName,"r");
	if (fp == NULL)
	{
		cout<<"invalid match file!"<<endl;
		return false;
	}
	fscanf(fp,"%d",&pointNum);
	vector<Point2d> pointSet_1,pointSet_2;
	Mat_<int> RGB_Pre;
	RGB_Pre = Mat(3,pointNum,CV_8UC1,Scalar(0));
	int r=0,g=0,b=0;
	for (i = 0; i < pointNum; i ++)
	{
//		fscanf(fp,"%lf %lf %lf %lf\n",&temp1.x,&temp1.y,&temp2.x,&temp2.y);
		fscanf(fp,"%d %lf %lf %lf %lf %d %d %d\n",&temp,&temp1.x,&temp1.y,&temp2.x,&temp2.y,&r,&g,&b);
		pointSet_1.push_back(temp1);
		pointSet_2.push_back(temp2);
		RGB_Pre.at<int>(0,i) = r;
		RGB_Pre.at<int>(1,i) = g;
		RGB_Pre.at<int>(2,i) = b;
	}
	fclose(fp);
	bool isexisting = false;
	vector<uchar> status;
	for (i = 0; i < pointNum; i ++)
	{
		isexisting = false;
		status.push_back(0);
		for (j = 0; j < pointSet1.size(); j ++)
		{
			if (pointSet_1[i] == pointSet1[j] || pointSet_2[i] == pointSet2[j])
			{
				isexisting = true;
				break;
			}
		}
		if (isexisting)
		{
			status[i] = 1;
			continue;
		}
		pointSet1.push_back(pointSet_1[i]);
		pointSet2.push_back(pointSet_2[i]);
	}
	int realNum = pointSet1.size();
	int n = 0;
	RGB = Mat(3,realNum,CV_8UC1,Scalar(0));
	for (i = 0; i < pointNum; i ++)
	{
		if (status[i] == 1)
		{
			continue;
		}
		RGB.at<int>(0,n) = RGB_Pre.at<int>(0,i);
		RGB.at<int>(1,n) = RGB_Pre.at<int>(1,i);
		RGB.at<int>(2,n) = RGB_Pre.at<int>(2,i);
		n ++;
	}
	return true;
}