#include "ProjectMatrix.h"
#include <math.h>

void ProjectMat::printM(Mat_<double> matrice)
{
	int row,col;
	Size matSize;
	matSize = matrice.size();
	col = matSize.width;
	row = matSize.height;

	for (int i = 0; i < row; i ++)
	{
		for (int j = 0; j < col; j ++)
		{
			printf("%lf ",matrice(i,j));
		}
		printf("\n");
	}
	printf("\n");
}


Point3d ProjectMat::triangulate3Dpoint(Mat_<double> P1,Mat_<double> P2,Point2d u,Point2d u1)
{
	Matx43d A(u.x*P1(2,0)-P1(0,0),u.x*P1(2,1)-P1(0,1),u.x*P1(2,2)-P1(0,2),  
		u.y*P1(2,0)-P1(1,0),u.y*P1(2,1)-P1(1,1),u.y*P1(2,2)-P1(1,2),  
		u1.x*P2(2,0)-P2(0,0), u1.x*P2(2,1)-P2(0,1),u1.x*P2(2,2)-P2(0,2),  
		u1.y*P2(2,0)-P2(1,0), u1.y*P2(2,1)-P2(1,1),u1.y*P2(2,2)-P2(1,2)                                          
		);  
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


bool ProjectMat::isRTcorrect(Mat_<double> R,Mat_<double> T)
{
	if (fabs(determinant(R))-1.0 > 1e-07 || fabs(determinant(R))-1.0 < -1e-07)
	{
		return false;
	}
	Point3d objectCenter = Point3d(0,0,0);
	Mat_<double> left_P,right_P;
	left_P =  K * Mat(Matx34d( 1, 0, 0, 0,  
		0, 1, 0, 0,  
		0, 0, 1, 0)); 
	right_P =  K * Mat(Matx34d( R(0,0),R(0,1), R(0,2), T(0),  
		R(1,0),R(1,1), R(1,2), T(1),  
		R(2,0),R(2,1), R(2,2), T(2))); 
	int numPoints = testPoint1.size();
	for (int i = 0; i < testPoint1.size(); i ++)
	{
		Point3d temp;
		temp = triangulate3Dpoint(left_P,right_P,testPoint1[i],testPoint2[i]);
		objectCenter += Point3d(temp);
	}
	Mat Transformation = Mat(Matx34d(R(0,0),R(0,1), R(0,2), T(0),  
		R(1,0),R(1,1), R(1,2), T(1),  
		R(2,0),R(2,1), R(2,2), T(2)));
	Mat_<double> X1 = (Mat_<double>(4,1) << objectCenter.x/numPoints, objectCenter.y/numPoints, objectCenter.z/numPoints,1);
	Mat_<double> X2 = Transformation * X1;
	if (X1(2) > 0 && X2(2) > 0)
	{
		return true;
	}
	else
	{
		return false;
	}
}


bool ProjectMat::DecomposeEtoRandT(Mat E,Mat_<double> &R1,Mat_<double> &R2,Mat_<double> &T1,Mat_<double> &T2)
{
	SVD svd(E);  
	Mat_<double> w = svd.w;
	double singular_values_ratio = fabsf(w.at<double>(0) / w.at<double>(1));
	if(singular_values_ratio > 1.0)
		singular_values_ratio = 1.0/singular_values_ratio; // flip ratio to keep it [0,1]
	if (singular_values_ratio < 0.7) {
		printf("singular values are too far apart\n");
		return false;
	}
	double newEigvalue = (w(0)+ w(1))/2;          //force S to be an standard matrix as required
	Mat_<double> S(Matx33d(newEigvalue, 0, 0,
		0, newEigvalue, 0,
		0, 0, 0));
	Mat_<double> newE = svd.u * S * svd.vt;

	SVD svd_test(newE);
	Matx33d W(0,-1,0,    
		1,0,0,  
		0,0,1);  
	R1 = svd_test.u * Mat(W) * svd_test.vt;         
	if(determinant(R1) < 0)
	{
		newE = -newE;
	}
	SVD svd_new(newE);
	R1 = svd_new.u * Mat(W) * svd_new.vt;
	R2 = svd_new.u * Mat(W.t()) * svd_new.vt;
	T1 = svd_new.u.col(2);    
	T2 = -svd_new.u.col(2);

	return true;
}


void ProjectMat::findNewPoints(vector<Point2d> PointSet)
{  
	dividePointLabels.clear();
	unsigned i,j;
	vector<pointInfo> tempSet = (*frameLists)[frameNo].pointSet;
	//vector<Point2f> basePoints;
	//Mat m(tempSet.size(),2,CV_32FC1);
	//Mat m1(PointSet.size(),2,CV_32FC1);
	//if (tempSet.size() > 0)
	//{
	//	for (i = 0; i < tempSet.size(); i ++)
	//	{
	//		basePoints.push_back(tempSet[i].point2D);
	//		Mat(Matx12f(tempSet[i].point2D.x,tempSet[i].point2D.y)).copyTo(m.row(i));
	//	}
	//	for (i = 0; i < PointSet.size(); i ++)
	//	{
	//		Mat(Matx12f(PointSet[i].x,PointSet[i].y)).copyTo(m1.row(i));
	//	}
	//	cv::flann::Index flann_index(m, cv::flann::KDTreeIndexParams(4));  // using 4 randomized kdtrees  
	//	vector<int> index;
	//	vector<float> dist;
	//	//	flann_index.knnSearch(PointSet,index,dist,2);
	//	flann_index.radiusSearch(m1,index,dist,2,1);

	//	//	flann_index.knnSearch(PointSet,index,dist,2);
	//	int  kkk;
	//}
	for (i = 0; i < PointSet.size(); i ++)
	{
		dividePointLabels.push_back(-1);
		for (j = 0; j < tempSet.size(); j ++)
		{
			if (PointSet[i] == tempSet[j].imagePoint)     //already existing points
			{
				dividePointLabels[i] = tempSet[j].no;
				break;
			}
		}
	}
}


void ProjectMat::setPointLabel(vector<Point2d> pointSet1,vector<Point2d> pointSet2)
{
	unsigned i;
	pointInfo tempPoint1,tempPoint2;
	cloudPoint tempPoint3d;
	int framePre,frameCur;
	framePre = frameNo;
	frameCur = frameNo+1;
	frameInfo newFrame;               
	newFrame.no = frameCur;
	frameLists->push_back(newFrame);            //add a new frame into the lists

	findNewPoints(pointSet1);
	newPointLabel.clear();
	newFrameoldPoints.clear();
	for (i = 0; i < dividePointLabels.size(); i ++)
	{    
		if (dividePointLabels[i] != -1)                //already existing points before new frame
		{
			tempPoint1.no = dividePointLabels[i];
			tempPoint1.imagePoint = pointSet2[i];
			(*frameLists)[frameCur].pointSet.push_back(tempPoint1);
			(*objectClouds)[dividePointLabels[i]].frameIndex.push_back(frameCur);     //add the new frame index to corresponding 3D point     

			newFrameoldPoints.push_back(tempPoint1);
		}
	}
	for (i = 0; i < dividePointLabels.size(); i ++)
	{   
		if (dividePointLabels[i] == -1)
		{
			tempPoint1.no = maxPointLabel;
			tempPoint1.imagePoint = pointSet1[i];
			(*frameLists)[framePre].pointSet.push_back(tempPoint1);

			tempPoint2.no = maxPointLabel;
			tempPoint2.imagePoint = pointSet2[i];
			(*frameLists)[frameCur].pointSet.push_back(tempPoint2);         

			tempPoint3d.frameIndex.clear();
			tempPoint3d.no = maxPointLabel;
			tempPoint3d.rgb[0] = RGB(0,i);
			tempPoint3d.rgb[1] = RGB(1,i);
			tempPoint3d.rgb[2] = RGB(2,i);
			tempPoint3d.frameIndex.push_back(framePre);
			tempPoint3d.frameIndex.push_back(frameCur);
			objectClouds->push_back(tempPoint3d);     //add the two frame index to new 3D point

			newPointLabel.push_back(maxPointLabel);      //remember new points' label
			maxPointLabel ++;
		}
	}
}


void ProjectMat::FindPoseEstimation(Mat_<double> &R,Mat_<double> &T)
{
	unsigned int i;
	vector<Point3f> objectPoints;            //opencv's function solvePnP or solvePnPRansac can only handle data of float class
	vector<Point2f> imagePoints;
	for (i = 0; i < newFrameoldPoints.size(); i ++)
	{
		int no = newFrameoldPoints[i].no;
		objectPoints.push_back((*objectClouds)[no].objectPoint);
		imagePoints.push_back(newFrameoldPoints[i].imagePoint);
	}
	Mat_<double> Rvec,Tvec;
	vector<double> distEcoff;
	vector<int> inliers;
	if (imagePoints.size() < 30)
	{
		cout<<imagePoints.size()<<" co-vision points is not enough!"<<endl;
		exit(0);
	}

	solvePnPRansac(objectPoints,imagePoints,K,distEcoff,Rvec,Tvec,false, 100, 3.0, 200, inliers, CV_ITERATIVE);
	vector<Point2f> reprojectPoints;
	projectPoints(objectPoints,Rvec,Tvec,K,distEcoff,reprojectPoints);
	double dist;
	vector<Point3f> goodOPoints;
	vector<Point2f> goodMPoints;
	for (i = 0; i < imagePoints.size(); i ++)                                
	{
		dist = sqrt((imagePoints[i].x - reprojectPoints[i].x)*(imagePoints[i].x - reprojectPoints[i].x)
			+ (imagePoints[i].y - reprojectPoints[i].y)*(imagePoints[i].y - reprojectPoints[i].y));
		if (dist <= 5.0)
		{
			goodMPoints.push_back(imagePoints[i]);
			goodOPoints.push_back(objectPoints[i]);
		}
	}
	solvePnP(goodOPoints,goodMPoints,K,distEcoff,Rvec,Tvec,true, CV_ITERATIVE);
	double ratio = (double)goodMPoints.size()/imagePoints.size();
	if (ratio >= 0.1 && inliers.size() >= 30)
	{
		if (inliers.size() < 100)
		{
			cout<<"the Pose estimation maybe noisy which is calculated by only "<<inliers.size()<<" points"<<endl;
		}
		Rodrigues(Rvec,R);
		T = Tvec;
		vector<Point2f> testPoints;
		projectPoints(goodOPoints,Rvec,Tvec,K,distEcoff,testPoints);

		double meanError = 0;
		for (i = 0; i < goodMPoints.size(); i ++)
		{   
			meanError += sqrt((goodMPoints[i].x - testPoints[i].x)*(goodMPoints[i].x - testPoints[i].x) + (goodMPoints[i].y - testPoints[i].y)*(goodMPoints[i].y - testPoints[i].y));
		}
		meanError /= goodMPoints.size();
		cout<<"RT-re_project-RMS:"<<meanError<<endl;
		if (meanError >= 5.0)
		{
			cout<<"the RMS is too big to continue!"<<endl;
			exit(0);
		}
	}
	else
	{
		cout<<"Pose estimation is failed!("<<"ratio "<<ratio<<" &inliers "<<inliers.size()<<")"<<endl;
		exit(0);
	}
}


void ProjectMat::getPointsonPolarline(vector<Point2d> &PointSet1,vector<Point2d> &PointSet2,Mat_<double> F,double T)
{
	vector<Point2d> tempSet1,tempSet2;
	tempSet1 = PointSet1;
	tempSet2 = PointSet2;
	PointSet1.clear();
	PointSet2.clear();
	double a,b,c,dist;
	Mat_<double> homoPoint,lineCoeffs;
	Point2d refPoint,testPoint;
	for (unsigned i = 0; i < tempSet1.size(); i ++)
	{
		refPoint = tempSet1[i];
		testPoint = tempSet2[i];
		homoPoint = (Mat_<double>(3,1) << refPoint.x, refPoint.y, 1);
		lineCoeffs = F * homoPoint;		
		a = lineCoeffs(0);
		b = lineCoeffs(1);
		c = lineCoeffs(2);

		dist = fabs(a*testPoint.x + b*testPoint.y + c)/sqrt(a*a + b*b);
		if (dist <= T)
		{
			PointSet1.push_back(refPoint);
			PointSet2.push_back(testPoint);
		}
	}
}

void ProjectMat::findRobustFundamentalMat(vector<Point2d> PointSet1,vector<Point2d> PointSet2)
{
	unsigned i;
	vector<uchar> status;
	Mat_<double> F;
	findFundamentalMat(PointSet1,PointSet2,CV_RANSAC, 1.0, 0.99,status);
	int time = 0;
	vector<Point2d> goodPoints1,goodPoints2;
	for (i = 0; i < status.size(); i ++)
	{
		if (status[i] == 1)
		{
			goodPoints1.push_back(PointSet1[i]);
			goodPoints2.push_back(PointSet2[i]);
			if (time % (status.size()/10) == 0)          //get the good corresponding about 10 points
			{
				testPoint1.push_back(PointSet1[i]);
				testPoint2.push_back(PointSet2[i]);
			}
			time ++;
		}
	}
	F = findFundamentalMat(goodPoints1,goodPoints2,CV_LMEDS,1.0,0.99);
	getPointsonPolarline(PointSet1,PointSet2, F, 1.0);
	Fmatrice = findFundamentalMat(PointSet1,PointSet2,CV_LMEDS,1.0,0.99);
}


void ProjectMat::FindCameraMatrices(vector<Point2d> pointSet1,vector<Point2d> pointSet2, Mat_<int> rgb, bool isFirstPair)
{
	RGB = rgb;
	setPointLabel(pointSet1,pointSet2);     //give 2D points the label corresponding to object point
	Mat_<double> R, T;
	if (isFirstPair)
	{
		Mat_<double> ProjectMatrice;
		vector<uchar> status;
		testPoint1.clear();
		testPoint2.clear();
				
		findRobustFundamentalMat(pointSet1, pointSet2);

		//FILE* fp = fopen("likai.txt","w");
		//fprintf(fp,"%lf %lf %lf\n",Fmatrice(0,0),Fmatrice(0,1),Fmatrice(0,2));
  //      fprintf(fp,"%lf %lf %lf\n",Fmatrice(1,0),Fmatrice(1,1),Fmatrice(1,2));
		//fprintf(fp,"%lf %lf %lf\n",Fmatrice(2,0),Fmatrice(2,1),Fmatrice(2,2));
		//fclose(fp);

/*		Fmatrice = findFundamentalMat(pointSet1,pointSet2,CV_LMEDS, 1.0, 0.99,status);
//		Fmatrice = findFundamentalMat(pointSet1,pointSet2,CV_RANSAC, 1.0, 0.99,status);
		vector<Point2f> goodPoints1,goodPoints2;
		int time = 0;
		for (unsigned i = 0; i < status.size(); i ++)
		{
			if (status[i] == 1)
			{
				goodPoints1.push_back(pointSet1[i]);
				goodPoints2.push_back(pointSet2[i]);
				if (time % (status.size()/10) == 0)          //get the good corresponding about 10 points
				{
					testPoint1.push_back(pointSet1[i]);
					testPoint2.push_back(pointSet2[i]);
				}
				time ++;
			}
		}*/

		Mat_<double> E = K.t() * Fmatrice * K;
		Mat_<double> R1, R2, T1, T2;
		if (DecomposeEtoRandT(E, R1, R2, T1, T2))
		{
			if (isRTcorrect(R1,T1))
			{	
				R = R1;
				T = T1;
			}
			else if (isRTcorrect(R1,T2))
			{
				R = R1;
				T = T2;
			}
			else if (isRTcorrect(R2,T2))
			{
				R = R2;
				T = T2;
			}
			else if (isRTcorrect(R2,T1))
			{
				R = R2;
				T = T1;
			}
			else
			{
				printf("there is error in estimating rotate and tranlation!\n");
				exit(1);
			}

		}
		else
		{
			printf("there is error in decomposing Essential matrix!\n");
			exit(1);
		}
	}
	else
	{
		FindPoseEstimation(R,T);
	}
	//cout<<Fmatrice<<endl<<endl;
	//Mat_<double> Fn = buildFundamentalMat(K, R, T);
	//cout<<Fn<<endl<<Fn.t()<<endl;
//	cout<<R<<endl<<T<<endl;
	(*frameLists)[++frameNo].ProjectMatrix = Mat(Matx34d(R(0,0),R(0,1), R(0,2), T(0),  
		                                                 R(1,0),R(1,1), R(1,2), T(1),  
		                                                 R(2,0),R(2,1), R(2,2), T(2)));
}


void ProjectMat::getNewFramePointLabels(vector<int> &oldPointLabels, vector<int> &newPointLabels)
{
	newPointLabels = newPointLabel;
	for (unsigned int i = 0; i < newFrameoldPoints.size(); i ++)
	{
		int label = newFrameoldPoints[i].no;
		oldPointLabels.push_back(label);
	}
}


Mat_<double> ProjectMat::buildFundamentalMat(Mat_<double> K, Mat_<double> R, Mat_<double> T)    //Fmatrix : u2*F*u1 = 0
{
	Mat_<double> A = K*T;
	Mat_<double> Ax = (Mat_<double>(3,3) << 0, -A(2), A(1), A(2), 0, -A(0), -A(1), A(0), 0);
	Mat_<double> B = K*R*K.inv();
	Mat_<double> Fmatrix = Ax * B;
	double* data = (double*)Fmatrix.data;
	for (int i = 0; i < 9; i ++)
	{
		data[i] /= data[8];
	}
	return Fmatrix;
}
