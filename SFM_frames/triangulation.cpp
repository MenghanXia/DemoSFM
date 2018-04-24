#pragma once
#include "triangulation.h"


Point3d Triangulation::triangulatePoint(vector<int> frameRelated,int pointLabel)
{
	unsigned i,j;
	vector<Point2d> pointSet;
	for (i = 0; i < frameRelated.size(); i ++)
	{
		for (j = 0; j < (*frameList)[frameRelated[i]].pointSet.size(); j ++)
		{
			if (pointLabel == (*frameList)[frameRelated[i]].pointSet[j].no)
			{
				pointSet.push_back((*frameList)[frameRelated[i]].pointSet[j].imagePoint);
				break;
			}
		}
	}
	int frameSum = frameRelated.size();
	Mat A(2*frameSum,3,CV_32FC1);
	Mat B(2*frameSum,1,CV_32FC1);
	Point2d u,u1;
	Mat_<double> P;
	Mat_<double> rowA1,rowA2,rowB1,rowB2;
	int k = 0;
	for (i = 0; i < frameSum; i ++)     //get the coefficient matrix A and B
	{
		u = pointSet[i];
		P = K * (*frameList)[frameRelated[i]].ProjectMatrix;
		Mat(Matx13d(u.x*P(2,0)-P(0,0),u.x*P(2,1)-P(0,1),u.x*P(2,2)-P(0,2))).copyTo(A.row(k));
		Mat(Matx13d(u.y*P(2,0)-P(1,0),u.y*P(2,1)-P(1,1),u.y*P(2,2)-P(1,2))).copyTo(A.row(k+1));
		Mat rowB1(1,1,CV_32FC1,Scalar(-(u.x*P(2,3)-P(0,3))));
		Mat rowB2(1,1,CV_32FC1,Scalar(-(u.y*P(2,3)-P(1,3))));
		rowB1.copyTo(B.row(k));
		rowB2.copyTo(B.row(k+1));
		k += 2;
	}
	Mat X;  
	solve(A,B,X,DECOMP_SVD);  
	return Point3d(X); 
}


void Triangulation::getPointsfromMultiview(int curFrameIndex,bool isFirstPair)
{
	if (isFirstPair)
	{
		return;
	}
	unsigned i;
	int no,minPointLabel= 99999,maxPointLabel = 0;
	for (i = 0; i < (*frameList)[curFrameIndex-2].pointSet.size(); i ++)    //find the maximum point label of all the triangulated 3D points in all frames (those overlapped at least 3 frames points)
	{
		no = (*frameList)[curFrameIndex-2].pointSet[i].no;
		if (no > maxPointLabel)
		{
			maxPointLabel = no;
		}
	}
	for (i = 0; i < (*frameList)[curFrameIndex].pointSet.size(); i ++)    //find the minimum point label in the current frame
	{
		no = (*frameList)[curFrameIndex].pointSet[i].no;
		if (no < minPointLabel)
		{
			minPointLabel = no;
		}
	}

	vector<int> frameRelated;
	Point3d objectPoint;
	int k = 0;
	for (i = minPointLabel; i <= maxPointLabel; i ++)
	{
		if (shallStart(i,curFrameIndex))
		{
			frameRelated = (*objectClouds)[i].frameIndex;
			objectPoint = triangulatePoint(frameRelated,i);
			(*objectClouds)[i].objectPoint = objectPoint;  
		}
	}
}


bool Triangulation::shallStart(int pointLabel,int curFrameIndex)
{
	vector<int> frameIndex = (*objectClouds)[pointLabel].frameIndex;
	bool isContianed = false;
	for (unsigned i = 0; i < frameIndex.size(); i ++)
	{
		if (frameIndex[i] == curFrameIndex) 
		{
			isContianed = true;
			break;
		}
	}
	if (!isContianed)
	{
		return false;
	}
	//else
	//{
	//	if (joinStopTimes[pointLabel]*1.1 >= 0.1 * frameIndex.size())
	//	{
	//		joinStopTimes[pointLabel] = 1;
	//		return true;
	//	}
	//	else
	//	{
	//		joinStopTimes[pointLabel] ++;
	//		return false;
	//	}
	//}
	return true;
}


void Triangulation::cloudRadiusFilter(vector<Point3d> inputSet, double radius, int thresholdSum,vector<char> &status)
{
	int i, j;
	int num = inputSet.size();
	Mat queryPoints(num, 3, CV_32FC1);

	for (i = 0; i < num; i ++)
	{
		queryPoints.at<float>(i, 0) = inputSet[i].x;
		queryPoints.at<float>(i, 1) = inputSet[i].y;
		queryPoints.at<float>(i, 2) = inputSet[i].z;
	}

	flann::Index flannIndex(queryPoints, cv::flann::KDTreeIndexParams(4));
	bool isGood;
	for (i = 0; i < num; i ++)
	{
		isGood = true;
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
		if (isGood && inputSet[i] != Point3d(0,0,0))
		{
			status.push_back(1);
		}
		else
		{
			status.push_back(0);
		}
	}
}


void Triangulation::addNewPoint(vector<int> newPointLabel,vector<Point3d> &newPoints)
{
	unsigned i,j;
	int frameNum = (*frameList).size();
	Mat_<double> P1 = K * (*frameList)[frameNum-2].ProjectMatrix;
	Mat_<double> P2 = K * (*frameList)[frameNum-1].ProjectMatrix;

	Point2d tempPoint1,tempPoint2;
	Point3d temp3DPoint;
	int num1 = (*frameList)[frameNum-2].pointSet.size();
	int num2 = (*frameList)[frameNum-1].pointSet.size();
	int num = newPointLabel.size();

	vector<Point3d> objectPoints;
	vector<pointInfo> imagePoints1 = (*frameList)[frameNum-2].pointSet;
	vector<pointInfo> imagePoints2 = (*frameList)[frameNum-1].pointSet;
	for (i = 0; i < num; i ++)
	{
		tempPoint1 = imagePoints1[num1-num+i].imagePoint;	
		tempPoint2 = imagePoints2[num2-num+i].imagePoint;
		temp3DPoint = triangulate3Dpoint(P1,P2,tempPoint1,tempPoint2);

		objectPoints.push_back(temp3DPoint);
	}
	int Sum = 0;
	double rms1,rms2;
	double meanError1=0, meanError2=0;	
	for (i = 0; i < num; i ++)
	{
		int no = newPointLabel[i];
		temp3DPoint = objectPoints[i];
		tempPoint1 = imagePoints1[num1-num+i].imagePoint;	
		tempPoint2 = imagePoints2[num2-num+i].imagePoint;
		rms1 = reprojectError(temp3DPoint,P1,tempPoint1);
		rms2 = reprojectError(temp3DPoint,P2,tempPoint2);

		if (rms1 <= 3.0 && rms2 <= 3.0)           //count those data satisfying the normal condition
		{
			newPoints.push_back(temp3DPoint);       //stock these new-added points for detecting whether there being repeated structures on current image
			meanError1 += rms1;
			meanError2 += rms2;
			Sum ++;
			(*objectClouds)[no].objectPoint = temp3DPoint;
		}
		else                                                 //set outliers as a special position to avoid being used by follow-up frames
		{
			(*objectClouds)[no].objectPoint = Point3d(0,0,0);
			(*frameList)[frameNum-2].pointSet[num1-num+i].imagePoint = Point2d(-1,-1);
			(*frameList)[frameNum-1].pointSet[num2-num+i].imagePoint = Point2d(-1,-1);
			sumOutlier ++;
		}	
	}
	if (newPointLabel.size() != 0)
	{
		cout<<"the new-added points re-project errors is:"<<meanError1/Sum<<" & "<<meanError2/Sum<<endl;
	}
	else
	{
		cout<<"no new points in this image pair!"<<endl;
	}
}


bool Triangulation::isIncluded(vector<int> dataSet,int queryIndex)
{
	for (unsigned int i = 0; i < dataSet.size(); i ++)
	{
		if (queryIndex == dataSet[i])
		{
			return true;
		}
	}
	return false;
}


bool Triangulation::isIncluded(vector<Point2d> dataSet,Point2d queryPoint)
{
	for (unsigned int i = 0; i < dataSet.size(); i ++)
	{
		if (queryPoint == dataSet[i])
		{
			return true;
		}
	}
	return false;
}


void Triangulation::extractObjectPoints(int frameIndex, vector<Point3d> &objectPoints)
{
	objectPoints.clear();
	unsigned int i,j;
	vector<pointInfo> PointSet = (*frameList)[frameIndex].pointSet;
	for (i = 0; i < PointSet.size(); i ++)
	{
		int no = PointSet[i].no;
		Point3d temp = (*objectClouds)[no].objectPoint;
		objectPoints.push_back(temp);
	}
}


bool Triangulation::checkRepeatStruct(vector<int> neigborPointLabel,vector<int> &relateFrames_)
{
	int i,j,k;
	int frameNum = frameList->size();
	if (frameNum <= 2)
	{
		return false;
	}
	relateFrames.clear();
	neigborFrames.clear();
	for (i = 0; i < neigborPointLabel.size(); i ++)                  //get cur=-frames' neighbor frames
	{
		int pointIndex = neigborPointLabel[i];
		vector<int> frameIndexs = (*objectClouds)[pointIndex].frameIndex;
		for (j = 0; j < frameIndexs.size(); j ++)
		{
			if (isIncluded(neigborFrames,frameIndexs[j]))
			{
				continue;
			}
			neigborFrames.push_back(frameIndexs[j]);
		}
	}
	int curframeIndex = frameNum-1;
	vector<Point3d> objectPoints;
	extractObjectPoints(curframeIndex,objectPoints);            //get cur=-frames' 3D points

	Mat image(imgSize.height,imgSize.width,CV_8UC3,Scalar(0,0,0));
	vector<Point2d> inliers;
	for (i = 0; i < frameNum; i ++)
	{
		inliers.clear();
		if (isIncluded(neigborFrames,i))
		{
			continue;
		}
		Mat_<double> P = K * (*frameList)[i].ProjectMatrix;
		Mat Transformation = (*frameList)[i].ProjectMatrix;
		int sum = 0;
		for (j = 0; j < objectPoints.size(); j ++)
		{
			Point3d tempPt = objectPoints[j];
			Mat_<double> Z = (Mat_<double>(4,1) << tempPt.x, tempPt.y, tempPt.z, 1);
			Mat_<double> temp = P * Z;

			Mat_<double> Z_local = Transformation * Z;
			if (Z_local(2) >= 0 && temp(0)/temp(2) >= 0 && temp(0)/temp(2) < imgSize.width &&
				temp(1)/temp(2) >= 0 && temp(1)/temp(2) < imgSize.height)
			{
				Point2d tempFeature = Point2d(temp(0)/temp(2),temp(1)/temp(2));
				inliers.push_back(tempFeature);
				sum ++;
			}
		}
		double spreadRatio = sum*1.0 / objectPoints.size();
		if ((spreadRatio >= 0.5 && sum >= 200))// || (spreadRatio >= 0.1 && sum >= 300))
		{
			for (int i1 = 0; i1 < inliers.size(); i1 ++)
			{
				Point2d aa = inliers[i1];
				circle(image,aa,1,Scalar(0,255,0),-1);
			}
			char fileName[256];
			sprintf(fileName,"frame%dfindings.jpg",i);
			imshow(fileName,image);
			waitKey(0);
			relateFrames.push_back(i);
			cout<<"related-frame:"<<i<<" containing current frame "<<sum<<"/"<<objectPoints.size()<<endl;
		}
	}
	relateFrames_ = relateFrames;
	if (relateFrames_.size() > 0)
	{
		return true;
	}
	else
	{
		return false;
	}
}


void Triangulation::modifycurSaM(vector<vector<Point2d> > PtList1,vector<vector<Point2d> > PtList2,int frameIndex)
{
	int i, j, k;
	Mat_<double> Hmatrix;
	bool isright = getTranformation1(PtList1, PtList2, frameIndex, Hmatrix);
	
	if (isright)
	{
		bool shallModify = isDriftExist(Hmatrix);
		//if (!shallModify)
		//{
		//	cout<<"the detected drift is too slight to handle!"<<endl;
		//	return;
		//}
		vector<int> pointLabels;
		int minIndex = 999,maxIndex = 0;
		int maxLabel = 0,minLabel = 999999;
		for (i = 0; i < neigborFrames.size(); i ++)
		{
			if (maxIndex < neigborFrames[i])
			{
				maxIndex = neigborFrames[i];
			}
			if (minIndex > neigborFrames[i])
			{
				minIndex = neigborFrames[i];
			}
		}

		double lamada = 0;             //linear modify co-effects
		Mat_<double> I = Mat::eye(4, 4, CV_64FC1);

		double step = 1.0/(maxIndex-minIndex);
		vector<double> coeffs;
		for (i = minIndex; i < maxIndex+1; i ++)
		{
			lamada = step*(i-minIndex);
			coeffs.push_back(lamada);
			Mat_<double> transform = Hmatrix*lamada + I*(1-lamada);
			(*frameList)[i].ProjectMatrix = (*frameList)[i].ProjectMatrix * Hmatrix;
		}

		for (i = 0; i < (*frameList)[minIndex].pointSet.size(); i ++)
		{
			if (minLabel > (*frameList)[minIndex].pointSet[i].no)
			{
				minLabel = (*frameList)[minIndex].pointSet[i].no;
			}
		}
		for (i = 0; i < (*frameList)[maxIndex].pointSet.size(); i ++)
		{
			if (maxLabel < (*frameList)[maxIndex].pointSet[i].no)
			{
				maxLabel = (*frameList)[maxIndex].pointSet[i].no;
			}
		}

		double lamada1 = 0.0;
		int num = 0;
		vector<int> frames;
		for (i = minLabel; i < maxLabel+1; i ++)
		{
			Point3d tempObjPt = (*objectClouds)[i].objectPoint;
			if (tempObjPt == Point3d(0,0,0))
			{
				continue;
			}
			Mat_<double> Z = (Mat_<double>(4,1) << tempObjPt.x, tempObjPt.y, tempObjPt.z, 1);

			lamada1 = 0.0;
			num = 0;
			frames = (*objectClouds)[i].frameIndex;
			for (j = 0; j < frames.size(); j ++)
			{
				for (k = 0; k < coeffs.size(); k ++)
				{
					if (frames[j] == k+minIndex)
					{
						lamada1 += coeffs[k];
						num ++;
						break;
					}
				}
			}
			if (num == 0)
			{
				continue;
			}
			lamada1 /= num;
			Mat_<double> transform = Hmatrix*lamada1 + I*(1-lamada1);
//			Mat_<double> Z_new = transform * Z;
			Mat_<double> Z_new = Hmatrix * Z;
			Point3d modifiedPt = Point3d(Z_new(0)/Z_new(3), Z_new(1)/Z_new(3), Z_new(2)/Z_new(3));

			(*objectClouds)[i].objectPoint = modifiedPt;
		}
	}
}


bool Triangulation::isDriftExist(Mat_<double> Transform)
{
//	cout<<Transform<<endl;
	Mat_<double> R = Mat(Matx33d(Transform(0,0),Transform(0,1),Transform(0,2),  
		                         Transform(1,0),Transform(1,1),Transform(1,2),  
		                         Transform(2,0),Transform(2,1),Transform(2,2)));
	Mat_<double> T = Mat(Matx31d(Transform(0,3),Transform(1,3),Transform(2,3)));
	Mat_<double> Rvec;
	Rodrigues(R, Rvec);
	double similarity_R = Rvec(0)*Rvec(0)+Rvec(1)*Rvec(1)+Rvec(2)*Rvec(2);
	double similarity_T = T(0)*T(0)+T(1)*T(1)+T(2)*T(2);
//	double Similarity = sqrt(similarity_R+similarity_T);
	if (similarity_R > 0.1*3.1415926/180 && similarity_T > 0.05)
	{
		cout<<Rvec<<endl;
		return true;
	}
	return false;
}					 
					 

bool Triangulation::getTranformation1(vector<vector<Point2d> > PtList1,vector<vector<Point2d> > PtList2,int frameIndex,Mat_<double> &Hmatrix)
{
	int i, j;
	vector<pointInfo> imgPointSet = (*frameList)[frameIndex].pointSet;

	vector<Point2d> imagePts;
	vector<Point3d> objectPts;
	for (i = 0; i < relateFrames.size(); i ++)
	{
		vector<Point2d> queryImgPts1 = PtList1[i];
		vector<Point2d> queryImgPts2 = PtList2[i];
		vector<pointInfo> trainImgPts2 = (*frameList)[relateFrames[i]].pointSet;
		vector<int> status;
		findSamePoints(queryImgPts2,trainImgPts2,status);

		for (j = 0; j < status.size(); j ++)
		{
			if (status[j] != -1)
			{
				if (isIncluded(imagePts,queryImgPts1[j]))
				{
					continue;
				}
				imagePts.push_back(queryImgPts1[j]);
				int no = trainImgPts2[status[j]].no;
				objectPts.push_back((*objectClouds)[no].objectPoint);
			}
		}
	}
	if (objectPts.size() < 100)
	{
		cout<<"the number of 2D-3D:"<<objectPts.size()<<" less than: 100"<<endl;
		return false;
	}
	Mat_<double> Rvec,Tvec,R,T;
	vector<double> distEcoffs;
	vector<int> inliers;
	vector<Point2f> imagePts_;
	vector<Point3f> objectPts_;
	for (i = 0; i < imagePts.size(); i ++)    
	{
		imagePts_.push_back(imagePts[i]);
		objectPts_.push_back(objectPts[i]);
	}
	solvePnPRansac(objectPts_,imagePts_,K,distEcoffs,Rvec,Tvec,false, 100, 3.0, 100, inliers, CV_ITERATIVE);
	Rodrigues(Rvec,R);
	T = Tvec;
	Mat_<double> newProjectMatrix = Mat(Matx34d(R(0,0),R(0,1),R(0,2),T(0),  
		                                        R(1,0),R(1,1),R(1,2),T(1),  
		                                        R(2,0),R(2,1),R(2,2),T(2)));
	Mat_<double> oldProjectMatrix = (*frameList)[frameIndex].ProjectMatrix;
	Mat_<double> preTransform = Mat(Matx44d(oldProjectMatrix(0,0),oldProjectMatrix(0,1),oldProjectMatrix(0,2),oldProjectMatrix(0,3),  
		                                    oldProjectMatrix(1,0),oldProjectMatrix(1,1),oldProjectMatrix(1,2),oldProjectMatrix(1,3),  
		                                    oldProjectMatrix(2,0),oldProjectMatrix(2,1),oldProjectMatrix(2,2),oldProjectMatrix(2,3),
											0,   0,    0,    1));
	Mat_<double> curTransform = Mat(Matx44d(newProjectMatrix(0,0),newProjectMatrix(0,1),newProjectMatrix(0,2),newProjectMatrix(0,3),  
		                                    newProjectMatrix(1,0),newProjectMatrix(1,1),newProjectMatrix(1,2),newProjectMatrix(1,3),  
		                                    newProjectMatrix(2,0),newProjectMatrix(2,1),newProjectMatrix(2,2),newProjectMatrix(2,3),
		                                    0,   0,    0,    1));
	Mat_<double> relateTransform = preTransform.inv() * curTransform;
	Hmatrix = relateTransform;

	return true;
}


bool Triangulation::getTranformation(vector<vector<Point2d> > PtList1,vector<vector<Point2d> > PtList2,int frameIndex,Mat_<double> &Hmatrix)
{
	int i, j;
	vector<pointInfo> imgPointSet = (*frameList)[frameIndex].pointSet;

	vector<Point3d> objectPts1,objectPts2;
	vector<Point3d> curObjectPoints;
	extractObjectPoints(frameIndex,curObjectPoints);
	for (i = 0; i < relateFrames.size(); i ++)
	{
		vector<Point2d> queryImgPts1 = PtList1[i];
		vector<int> status1;
		findSamePoints(queryImgPts1,imgPointSet,status1);

		vector<Point2d> queryImgPts2 = PtList2[i];
		vector<pointInfo> trainImgPts2 = (*frameList)[relateFrames[i]].pointSet;
		vector<int> status2;
		findSamePoints(queryImgPts2,trainImgPts2,status2);

		for (j = 0; j < status1.size(); j ++)
		{
			if (status1[j] != -1 && status2[j] != -1)
			{
				int no1 = status1[j];
				int no2 = trainImgPts2[status2[j]].no;
				
				objectPts1.push_back(curObjectPoints[no1]);
				objectPts2.push_back((*objectClouds)[no2].objectPoint);   //warning: when the relateFrame contains more than 1 elements,there will be many repetitive elements in the objectPts1 and objectPt2	
			}
		}
	}
	if (objectPts1.size() < 32)
	{
		cout<<"point num:"<<objectPts1.size()<<" less than: 32"<<endl;
		return false;
	}
	FILE *fp1 = fopen("point1.txt","w");
	for (i = 0; i < objectPts1.size(); i ++)
	{
		fprintf(fp1,"%lf %lf %lf\n",objectPts1[i].x,objectPts1[i].y,objectPts1[i].z);
	}
	fclose(fp1);
	FILE *fp2 = fopen("point2.txt","w");
	for (i = 0; i < objectPts2.size(); i ++)
	{
		fprintf(fp2,"%lf %lf %lf\n",objectPts2[i].x,objectPts2[i].y,objectPts2[i].z);
	}
	fclose(fp2);

	bool isright = true; //findPerspectiveMat(objectPts1, objectPts2, Hmatrix);
	vector<uchar> status;
	Mat_<double> H;
	estimateAffine3D(objectPts1, objectPts2, H, status, 3.0, 0.99);
	Hmatrix = Mat(Matx44d( H(0,0),H(0,1),H(0,2),H(0,3),  
		H(1,0),H(1,1),H(1,2),H(1,3),  
		H(2,0),H(2,1),H(2,2),H(2,3),
		0,   0,   0,   1)); 
	FILE *fp3 = fopen("new.txt","w");
	for (i = 0; i < objectPts1.size(); i ++)
	{
		Mat_<double> Z = (Mat_<double>(4,1) << objectPts1[i].x,objectPts1[i].y,objectPts1[i].z, 1);
		Mat_<double> Z_new = Hmatrix * Z;
		Point3d modifiedPt = Point3d(Z_new(0)/Z_new(3), Z_new(1)/Z_new(3), Z_new(2)/Z_new(3));
		fprintf(fp3,"%lf %lf %lf\n",modifiedPt.x,modifiedPt.y,modifiedPt.z);
	}
	fclose(fp3);
	if (isright)
	{
		return true;
	}
	return false;
	//bool isright = findPerspectiveMat(objectPts1, objectPts2,Hmatrix);
	//if (isright)
	//{
	//	cout<<Hmatrix<<endl;
	//}
}


bool Triangulation::findPerspectiveMat(vector<Point3d> objectPts1,vector<Point3d> objectPts2,Mat_<double> &Hmatrix)
{
	double a[16*16], w[16], v[16*16];  
	CvMat W = cvMat( 1, 16, CV_64F, w );  
	CvMat V = cvMat( 16, 16, CV_64F, v );  
	CvMat A = cvMat( 16, 16, CV_64F, a );  
	CvMat U, H0, TF;  

	Point3d centre1 = Point3d(0,0,0), centre2 = Point3d(0,0,0);
	double t, scale1 = 1.0, scale2 = 1.0;  

	int i, j, k;
	// compute centers and average distances for each of the two point sets  
	int count = objectPts1.size();
	for( i = 0; i < count; i ++ )  
	{  
		double x = objectPts1[i].x, y = objectPts1[i].y, z = objectPts1[i].z;  
		centre1.x += x; centre1.y += y; centre1.z += z; 
		
		x = objectPts2[i].x, y = objectPts2[i].y, z = objectPts2[i].z;  
		centre2.x += x; centre2.y += y; centre2.z += z; 
	}  
	// calculate the normalizing transformations for each of the point sets:  
	// after the transformation each set will have the mass center at the coordinate origin  
	// and the average distance from the origin will be ~sqrt(2).  
	t = 1.0/count;  
	centre1.x *= t; centre1.y *= t; centre1.z *= t; 
	centre2.x *= t; centre2.y *= t; centre2.z *= t;   
	for( i = 0; i < count; i++ )
	{  
		double dx = objectPts1[i].x -centre1.x, dy = objectPts1[i].y - centre1.y, dz = objectPts1[i].z - centre1.z;  
		scale1 += sqrt(dx*dx + dy*dy + dz*dz);  
		dx = objectPts2[i].x -centre2.x, dy = objectPts2[i].y - centre2.y, dz = objectPts2[i].z - centre2.z;  
		scale2 += sqrt(dx*dx + dy*dy + dz*dz);   
	}  
	scale1 *= t;  
	scale2 *= t;  
	if( scale1 < FLT_EPSILON || scale2 < FLT_EPSILON )  
		return false;  
	scale1 = sqrt(3.)/scale1;  
	scale2 = sqrt(3.)/scale2;  

	cvZero( &A );  

	// form a linear system Ax=0: for each selected pair of points m1 & m2,  
	// the row of A(=a) represents the coefficients of equation: (m2, 1)'*F*(m1, 1) = 0  
	// to save computation time, we compute (At*A) instead of A and then solve (At*A)x=0.   
	for( i = 0; i < count; i++ )  
	{  
		double x1 = (objectPts1[i].x - centre1.x)*scale1;  
		double y1 = (objectPts1[i].y - centre1.y)*scale1;  
		double z1 = (objectPts1[i].z - centre1.z)*scale1;

		double x2 = (objectPts2[i].x - centre2.x)*scale2;  
		double y2 = (objectPts2[i].y - centre2.y)*scale2;
		double z2 = (objectPts2[i].z - centre2.z)*scale2;
		double r[16] = { x2*x1, x2*y1, x2*z1, x2, y2*x1, y2*y1, y2*z1, y2, z2*x1, z2*y1, z2*z1, z2, x1, y1, z1, 1}; 
		for( j = 0; j < 16; j++ )  
			for( k = 0; k < 16; k++ )  
				a[j*16+k] += r[j]*r[k];  
	}  
	cvSVD( &A, &W, 0, &V, CV_SVD_MODIFY_A + CV_SVD_V_T );  
	H0 = cvMat( 4, 4, CV_64F, v + 16*15 ); // take the last column of v as a solution of Af = 0  

	// apply the transformation that is inverse  
	// to what we used to normalize the point coordinates  
	{  
		double tt1[] = {scale1, 0, 0, -scale1*centre1.x, 0, scale1, 0, -scale1*centre1.y, 0, 0, scale1, -scale1*centre1.z, 0, 0, 0, 1};
		double tt2[] = {scale2, 0, 0, -scale2*centre2.x, 0, scale2, 0, -scale2*centre2.y, 0, 0, scale2, -scale2*centre2.z, 0, 0, 0, 1};
		CvMat T1, T2;
		TF = T1 = T2 = H0;  
		T1.data.db = tt1;  
		T2.data.db = tt2;  
		// F0 <- T1'*F0*T0  
		cvGEMM( &T2, &H0, 1., 0, 0., &TF, CV_GEMM_A_T );   
		cvGEMM( &TF, &T1, 1., 0, 0., &H0, 0 );   
	} 
	Mat tempH = Mat(4,4,CV_64FC1,Scalar(0));
	Mat_<double> H = &H0;
	for (i = 0; i < 4; i ++)
	{
		for (j = 0; j < 4; j ++)
		{
			tempH.at<double>(i,j) = H(i,j);
		}
	}
	Hmatrix = tempH; 
	return true;
}


void Triangulation::findSamePoints(vector<Point2d> querySet,vector<pointInfo> trainSet,vector<int> &status)
{
	int i, j;
	status.clear();
	for (i = 0; i < querySet.size(); i ++)
	{
		status.push_back(-1);
		for (j = 0; j < trainSet.size(); j ++)
		{
			if (querySet[i] == trainSet[j].imagePoint)  
			{
				status[i] = j;
				break;
			}
		}
	}
	int a = 0;
}


Point3d Triangulation::triangulate3Dpoint(Mat_<double> P1,Mat_<double> P2,Point2d u,Point2d u1)
{
	Matx43d A(u.x*P1(2,0)-P1(0,0),u.x*P1(2,1)-P1(0,1),u.x*P1(2,2)-P1(0,2),  
		u.y*P1(2,0)-P1(1,0),u.y*P1(2,1)-P1(1,1),u.y*P1(2,2)-P1(1,2),  
		u1.x*P2(2,0)-P2(0,0), u1.x*P2(2,1)-P2(0,1),u1.x*P2(2,2)-P2(0,2),  
		u1.y*P2(2,0)-P2(1,0), u1.y*P2(2,1)-P2(1,1),u1.y*P2(2,2)-P2(1,2));  
	//build B vector  
	Matx41d B(-(u.x*P1(2,3)-P1(0,3)),  
		-(u.y*P1(2,3)-P1(1,3)),  
		-(u1.x*P2(2,3)-P2(0,3)),  
		-(u1.y*P2(2,3)-P2(1,3)));

	//solve for X  
	Mat X;  
	solve(A,B,X,DECOMP_SVD);  
	return Point3d(X);  
}


double Triangulation::reprojectError(Point3d object, Mat_<double> P, Point2d pointSet)
{
	double meanError = 0;
	Mat_<double> Z = (Mat_<double>(4,1) << object.x, object.y, object.z, 1);
	Mat_<double> temp1 = P * Z;
	meanError = sqrt((temp1(0)/temp1(2) - pointSet.x)*(temp1(0)/temp1(2) - pointSet.x)+(temp1(1)/temp1(2) - pointSet.y)*(temp1(1)/temp1(2) - pointSet.y));
	return meanError;
}


double Triangulation::meanReprojectError(int start, int end, double sampleRate)
{
	unsigned i,j,k,n;
	vector<Point2d> graphPoints;
	vector<Point3d> objetPoints;
	int step = 1.0/sampleRate;
	double AveError = 0;
	for (i = start; i <= end; i ++)
	{
		vector<pointInfo> PointSet = (*frameList)[i].pointSet;
		graphPoints.clear();
		objetPoints.clear();
		for (j = 0; j < PointSet.size(); j += step)
		{
			if (PointSet[j].imagePoint == Point2d(-1,-1))
			{
				n = j+1;
				while(n < PointSet.size())
				{
					if (PointSet[n].imagePoint != Point2d(-1,-1))
					{
						break;
					}
					n ++;
				}
				if (n < PointSet.size())
				{
					graphPoints.push_back(PointSet[n].imagePoint);
					objetPoints.push_back((*objectClouds)[PointSet[n].no].objectPoint);
				}
			}
			else
			{
				graphPoints.push_back(PointSet[j].imagePoint);
				objetPoints.push_back((*objectClouds)[PointSet[j].no].objectPoint);
			}
		}

		Mat_<double> P = K * (*frameList)[i].ProjectMatrix;
		double meanError = 0;
		for (k = 0; k < graphPoints.size(); k ++)
		{
			meanError += reprojectError(objetPoints[k], P, graphPoints[k]);
		}
		meanError /= graphPoints.size();

		AveError += meanError;
	}

	AveError /= (end-start+1);
	return AveError;
}


