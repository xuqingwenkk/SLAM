#include "sfm.h"
#include "opencv2/nonfree/nonfree.hpp" //SIFT and SURF

#include "pcl/io/pcd_io.h"
#include "pcl/point_types.h"
#include "pcl/common/transforms.h"
#include "pcl/visualization/cloud_viewer.h"

CAMERA_INTRINSIC_PARAM getDefaultCamera()
{
  static ParamReader pr;
  CAMERA_INTRINSIC_PARAM camera;
  camera.cx = atof(pr.getData("camera.cx").c_str());
  camera.cy = atof(pr.getData("camera.cy").c_str());
  camera.fx = atof(pr.getData("camera.fx").c_str());
  camera.fy = atof(pr.getData("camera.fy").c_str());
  camera.scale = atof(pr.getData("camera.scale").c_str());
  camera.k1 = atof(pr.getData("camera.k1").c_str());
  camera.k2 = atof(pr.getData("camera.k2").c_str());
  camera.k3 = atof(pr.getData("camera.k3").c_str());
  camera.k4 = atof(pr.getData("camera.k4").c_str());
  return camera;
}

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

void computeKeypointAndDesp(FRAME & frame, string detector, string descriptor)
{
  cv::initModule_nonfree();
  cv::Ptr<cv::FeatureDetector> _detector;
  cv::Ptr<cv::DescriptorExtractor> _descriptor;
  _detector = cv::FeatureDetector::create("SIFT");
  _descriptor = cv::DescriptorExtractor::create("SIFT");
  if(!_detector || !_descriptor)
  {
    cout << " Invalid type :  " << detector << ", " << descriptor << endl;
    return;
  }
  _detector->detect(frame.rgb, frame.kp);
  _descriptor->compute(frame.rgb, frame.kp, frame.desp);
}

void getGoodMatches(FRAME & f1, FRAME & f2, vector<cv::DMatch> & good_matches)
{
  static ParamReader pr;
  double good_matches_threshold = atof(pr.getData("good_matches_threshold").c_str());
  double minDis_threshold = atof(pr.getData("minDis_threshold").c_str());
  vector<cv::DMatch> matches;
  cv::FlannBasedMatcher matcher;
  matcher.match(f1.desp, f2.desp, matches);
  cout << "Find " << matches.size() << " matches" << endl;
  // filter good matches
  double minDis = 9999;
  for(size_t i = 0; i < matches.size(); i++)
  {
    if(matches[i].distance < minDis)
      minDis = matches[i].distance;
  }
  
  if(minDis < minDis_threshold)
    minDis = minDis_threshold;
  
  for(size_t i = 0; i < matches.size(); i++)
  {
    if(matches[i].distance < good_matches_threshold*minDis)
      good_matches.push_back(matches[i]);
  }
  cout << "good matches = " << good_matches.size() << endl;
}

void get_matched_points(vector<cv::KeyPoint>& kp1, vector<cv::KeyPoint>& kp2, vector<cv::DMatch> matches, vector<cv::Point2f>& p1, vector<cv::Point2f>& p2)
{  
  p1.clear();
  p2.clear();
  for (int i = 0; i < matches.size(); ++i)
  {
    p1.push_back(kp1[matches[i].queryIdx].pt);
    p2.push_back(kp2[matches[i].trainIdx].pt);
  }
}

bool find_transform(CAMERA_INTRINSIC_PARAM& camera, vector<cv::Point2f>& p1, vector<cv::Point2f>& p2, cv::Mat& R, cv::Mat& T, cv::Mat& mask)
{
  cv::Mat F = cv::findFundamentalMat(p1, p2, CV_FM_RANSAC, 3, 0.99, mask);
  if (F.empty()) return false;
  double feasible_count = countNonZero(mask);
  cout << (int)feasible_count << " -in- " << p1.size() << endl;
  cv::Mat K= cv::Mat::eye(3,3,CV_64FC1);  
  K.at<double>(0,0) = camera.fx;  
  K.at<double>(1,1) = camera.fy;  
  K.at<double>(0,2) = camera.cx;  
  K.at<double>(1,2) = camera.cy;
  
  cv::Mat Kt=K.t();
  cv::Mat E=Kt*F*K; 
  
  cv::SVD svd(E);
  
  cv::Mat W=cv::Mat::eye(3,3,CV_64FC1);  
  W.at<double>(0,1)=-1;  
  W.at<double>(1,0)=1;  
  W.at<double>(2,2)=1;  
  
  R=svd.u*W*svd.vt;  
  T=svd.u.col(2);
  
  return true;
}

void reconstruct(CAMERA_INTRINSIC_PARAM& camera, cv::Mat& R, cv::Mat& T, vector<cv::Point2f>& p1, vector<cv::Point2f>& p2, cv::Mat& structure)
{
  cv::Mat K= cv::Mat::eye(3,3,CV_64FC1);  
  K.at<double>(0,0) = camera.fx;  
  K.at<double>(1,1) = camera.fy;  
  K.at<double>(0,2) = camera.cx;  
  K.at<double>(1,2) = camera.cy;
  //两个相机的投影矩阵[R T]，triangulatePoints只支持float型
  cv::Mat proj1(3, 4, CV_32FC1);
  cv::Mat proj2(3, 4, CV_32FC1);
  proj1(cv::Range(0, 3), cv::Range(0, 3)) = cv::Mat::eye(3, 3, CV_32FC1);
  proj1.col(3) = cv::Mat::zeros(3, 1, CV_32FC1);
  cv::Mat rotation(3, 3, CV_32FC1);
  cv::Mat trans(3, 1, CV_32FC1);
  R.convertTo(rotation, CV_32FC1);
  cv::Mat temp1 = proj2(cv::Range(0, 3), cv::Range(0, 3));
  rotation.copyTo(temp1);
  T.convertTo(trans, CV_32FC1);
  cv::Mat temp2 = proj2.col(3);
  trans.copyTo(temp2);
//R.convertTo(proj2(Range(0, 3), Range(0, 3)), CV_32FC1);
//T.convertTo(proj2.col(3), CV_32FC1);
  cv::Mat fK;
  K.convertTo(fK, CV_32FC1);
  proj1 = fK*proj1;
  proj2 = fK*proj2;
  //三角重建
  cv::triangulatePoints(proj1, proj2, p1, p2, structure);
}


void maskout_points(vector<cv::Point2f>& p1, cv::Mat& mask)
{
  vector<cv::Point2f> p1_copy = p1;
  p1.clear();
  for (int i = 0; i < mask.rows; ++i)
  {
    if (mask.at<uchar>(i) > 0)
      p1.push_back(p1_copy[i]);
  }
}

void show4NPoints(cv::Mat& structure)
{
  static ParamReader pr;
  string sfm_dir = pr.getData("sfm_dir");
  ofstream fout(sfm_dir);
  pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
  for (size_t i = 0; i < structure.cols; ++i)
  {
    cv::Mat_<float> c = structure.col(i);
    c /= c(3);	//齐次坐标，需要除以最后一个元素才是真正的坐标值
    pcl::PointXYZ point;
    point.x = c(0);
    point.y = c(1);
    point.z = c(2);
    fout << "[ " << c(0) << ", " << c(1) << ", " << c(2) << " ]" << endl;
    //cv::Point3f(c(0), c(1), c(2));
    point_cloud_ptr->points.push_back(point);
  }
  fout.close();
  point_cloud_ptr->width = (int)point_cloud_ptr->points.size();
  point_cloud_ptr->height = 1;
  pcl::visualization::CloudViewer viewer("viewer");
  viewer.showCloud(point_cloud_ptr);
  while(!viewer.wasStopped())
  {
  }
}