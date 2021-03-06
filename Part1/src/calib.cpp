#include "calib.h"
#include <sstream>

vector<cv::Point2f> FeatureExtraction(cv::Mat & image, cv::Size & board_size, int & image_index)
{
  static ParamReader pr;
  vector<cv::Point2f> corners;
  cv::Mat imageGray;
  cv::cvtColor(image, imageGray , CV_RGB2GRAY);
  bool patternfound = cv::findChessboardCorners(image, board_size, corners,cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE );
  if (!patternfound)   
  {
    cout<<"can not find chessboard corners!\n";  
    exit(1);   
  } 
  else
  {
    cv::cornerSubPix(imageGray, corners, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
    cv::Mat imageTemp = image.clone();
    for (int j = 0; j < corners.size(); j++)
    {
      cv::circle( imageTemp, corners[j], 10, cv::Scalar(0,0,255), 2, 8, 0);
    }
    string image_dir = pr.getData("savedir");
    string imageFileName;
    stringstream StrStm;
    StrStm << image_index + 1;
    StrStm >> imageFileName;
    imageFileName += "_t.jpg";
    imageFileName = image_dir + imageFileName;
    cv::imwrite(imageFileName, imageTemp);
    return corners;
  }   
}

void FisheyeCalib(vector<vector<cv::Point2f> >  corners_Seq, vector<vector<cv::Point3f> > & object_Points, cv::Size & image_size, cv::Matx33d & intrinsic_matrix, cv::Vec4d & distortion_coeffs, 
  vector<cv::Vec3d> & rotation_vectors, vector<cv::Vec3d> & translation_vectors, int & successImageNum, cv::Size & board_size)
{
  static ParamReader pr;
  int bs = atoi(pr.getData("bs").c_str());
  cv::Size square_size = cv::Size(bs, bs);     
  vector<int>  point_counts;                                                         
  /* 初始化定标板上角点的三维坐标 */
  for (int t = 0; t< successImageNum; t++)
  {
    vector<cv::Point3f> tempPointSet;
    for (int i = 0; i< board_size.height; i++)
    {
      for (int j = 0; j< board_size.width; j++)
      {
        /* 假设定标板放在世界坐标系中z=0的平面上 */
        cv::Point3f tempPoint;
        tempPoint.x = i*square_size.width;
        tempPoint.y = j*square_size.height;
        tempPoint.z = 0;
        tempPointSet.push_back(tempPoint);
      }
    }
    object_Points.push_back(tempPointSet);
  }
  for (int i = 0; i< successImageNum; i++)
  {
    point_counts.push_back(board_size.width*board_size.height);
  }
  /* 开始定标 */
  int flags = 0;
  flags |= cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC;
  flags |= cv::fisheye::CALIB_CHECK_COND;
  flags |= cv::fisheye::CALIB_FIX_SKEW;
  cv::fisheye::calibrate(object_Points, corners_Seq, image_size, intrinsic_matrix, distortion_coeffs, rotation_vectors, translation_vectors, flags, cv::TermCriteria(3, 20, 1e-6));
}

void Estimation(vector<vector<cv::Point3f> > & object_Points, cv::Matx33d & intrinsic_matrix, cv::Vec4d & distortion_coeffs, vector<cv::Vec3d> & rotation_vectors, 
		vector<cv::Vec3d> & translation_vectors, int & image_count, cv::Size & board_size, vector<vector<cv::Point2f> > & corners_Seq)
{
  double total_err = 0.0;                   /* 所有图像的平均误差的总和 */   
  double err = 0.0;                        /* 每幅图像的平均误差 */   
  vector<cv::Point2f>  image_points2;             /****   保存重新计算得到的投影点    ****/   
  cout<<"Calibration error for each image: " << endl;   
  for (int i=0;  i< image_count;  i++) 
  {
    vector<cv::Point3f> tempPointSet = object_Points[i];
    cv::fisheye::projectPoints(tempPointSet, image_points2, rotation_vectors[i], translation_vectors[i], intrinsic_matrix, distortion_coeffs);
    vector<cv::Point2f> tempImagePoint = corners_Seq[i];
    cv::Mat tempImagePointMat = cv::Mat(1,tempImagePoint.size(),CV_32FC2);
    cv::Mat image_points2Mat = cv::Mat(1,image_points2.size(), CV_32FC2);
    for (size_t j = 0 ; j != tempImagePoint.size(); j++)
    {
      image_points2Mat.at<cv::Vec2f>(0,j) = cv::Vec2f(image_points2[j].x, image_points2[j].y);
      tempImagePointMat.at<cv::Vec2f>(0,j) = cv::Vec2f(tempImagePoint[j].x, tempImagePoint[j].y);
    }
    err = cv::norm(image_points2Mat, tempImagePointMat, cv::NORM_L2);
    total_err += err/=  (board_size.width*board_size.height);   
    cout<<"The error of Image "<<i+1<<"is: "<<err<<" pixels."<<endl;     
  }   
  cout<<"The whole average error is "<<total_err/image_count<<" pixels."<<endl;    
}


