#include "sfm.h"
#include <sstream>
#include <iostream>
#include "pcl/io/pcd_io.h"
#include "pcl/point_types.h"
#include "pcl/common/transforms.h"
#include "pcl/visualization/cloud_viewer.h"

using namespace std;

cv::Mat undistort(cv::Mat & img, CAMERA_INTRINSIC_PARAM & camera);

FRAME readFrame(int index, ParamReader & pr, CAMERA_INTRINSIC_PARAM & camera);

TRANSFORM estimateMotion(FRAME & f1, FRAME & f2,  CAMERA_INTRINSIC_PARAM & camera);

double normofTransform(cv::Mat rvec, cv::Mat tvec);

pcl::PointCloud<pcl::PointXYZ>::Ptr jointPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr original, TRANSFORM & result, CAMERA_INTRINSIC_PARAM & camera);

int main(int argc, char** argv)
{
  CAMERA_INTRINSIC_PARAM camera = getDefaultCamera();
  ParamReader pr;
  int start_index = atoi(pr.getData("start_index").c_str());
  int end_index = atoi(pr.getData("end_index").c_str());
  
  cout << "Initializing .... " << endl;
  int currIndex = start_index;
  FRAME lastFrame = readFrame(currIndex, pr, camera);
  
  string detector = pr.getData("detector");
  string descriptor = pr.getData("descriptor");
  computeKeypointAndDesp(lastFrame, detector, descriptor);
  
  int min_inliers = atoi(pr.getData("min_inliers").c_str());
  double max_norm = atof(pr.getData("max_norm").c_str());
  bool visualization = pr.getData("visualize_pointcloud")==string("yes");
  pcl::visualization::CloudViewer viewer("viewer");
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  
  for(currIndex = start_index + 1; currIndex != end_index; currIndex++)
  {
    FRAME currFrame = readFrame(currIndex, pr, camera);
    computeKeypointAndDesp(currFrame, detector, descriptor);
    
    TRANSFORM result = estimateMotion(lastFrame, currFrame, camera);
    if( (result.inliers == -1) || (result.inliers < min_inliers) )
      continue;
    double norm = normofTransform(result.rvec, result.tvec);
    if(norm >= max_norm)
      continue;
    
    cloud = jointPointCloud(cloud, result, camera);
    if(visualization == true)
      viewer.showCloud(cloud);
    
    lastFrame = currFrame;
  }
  
  return 0;
}

cv::Mat undistort(cv::Mat & img, CAMERA_INTRINSIC_PARAM & camera)
{
  cv::Matx33d intrinsic_matrix(camera.fx,0,camera.cx,0,camera.fy,camera.cy,0,0,1);
  cv::Vec4d distortion_coeffs(camera.k1,camera.k2,camera.k3,camera.k4);
  cv::Mat R1 = cv::Mat::eye(3,3,CV_32F);
  cv::Size image_size = img.size();
  cv::Mat mapx = cv::Mat(image_size,CV_32FC1);
  cv::Mat mapy = cv::Mat(image_size,CV_32FC1);
  cv::fisheye::initUndistortRectifyMap(intrinsic_matrix,distortion_coeffs,R1,intrinsic_matrix,image_size,CV_32FC1,mapx,mapy);
  cv::Mat t2 = img.clone();
  cv::remap(img,t2,mapx, mapy, cv::INTER_LINEAR);
  return t2;
  
}

FRAME readFrame(int index, ParamReader & pr, CAMERA_INTRINSIC_PARAM & camera)
{
  FRAME f;
  string image_dir = pr.getData("image_dir");
  string image_ext = pr.getData("image_ext");
  
  stringstream ss;
  ss<<image_dir<<index<<image_ext;
  string filename;
  ss>>filename;
  
  cv::Mat image = cv::imread(filename);
  f.rgb = undistort(image, camera);
  f.frameID = index;
  return f;
}

TRANSFORM estimateMotion(FRAME & f1, FRAME & f2, CAMERA_INTRINSIC_PARAM & camera)
{
  static ParamReader pr;
  int min_good_matches = atoi(pr.getData("min_good_matches").c_str());
  TRANSFORM result;
  vector<cv::DMatch> good_matches;
  getGoodMatches(f1, f2, good_matches);
  
  if(good_matches.size() < min_good_matches)
  {
    cout << "Not enough good matches" << endl;
    result.inliers = -1;
    return result;
  }
  
  vector<cv::Point2f> p1;
  vector<cv::Point2f> p2;
  cv::Mat R, T;
  cv::Mat mask;
  get_matched_points(f1.kp, f2.kp, good_matches, p1, p2);
  bool flag = find_transform(camera, p1, p2, R, T, mask);
  if (!flag)
  {
    cout << "find transform failed! " << endl;
    result.inliers = -1;
  }
  result.rvec = R;
  result.tvec = T;
  result.p1 = p1;
  result.p2 = p2;
  result.inliers = cv::countNonZero(mask);
  result.mask = mask;
  return result;
}

double normofTransform(cv::Mat rvec, cv::Mat tvec)
{
  return fabs(min(cv::norm(rvec), 2*M_PI-cv::norm(rvec)))+ fabs(cv::norm(tvec));
}

pcl::PointCloud<pcl::PointXYZ>::Ptr jointPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr original, TRANSFORM & result, CAMERA_INTRINSIC_PARAM & camera)
{
  cv::Mat structure;	//4行N列的矩阵，每一列代表空间中的一个点（齐次坐标）
  maskout_points(result.p1, result.mask);
  maskout_points(result.p2, result.mask);
  reconstruct(camera, result.rvec, result.tvec, result.p1, result.p2, structure);
  pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
  for (size_t i = 0; i < structure.cols; ++i)
  {
    cv::Mat_<float> c = structure.col(i);
    c /= c(3);	//齐次坐标，需要除以最后一个元素才是真正的坐标值
    pcl::PointXYZ point;
    point.x = c(0);
    point.y = c(1);
    point.z = c(2);
    point_cloud_ptr->points.push_back(point);
  }
  *original += *point_cloud_ptr;
  return original;
  
}