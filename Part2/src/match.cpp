#include "sfm.h"

int main(int argc, char** argv)
{
  ParamReader pr;
  string image_dir = pr.getData("image_dir");
  FRAME f1, f2;
  string imageName("test_0.jpg");
  imageName = image_dir + imageName;
  cv::Mat img1 = cv::imread(imageName);
  imageName.clear();
  imageName = image_dir + string("test_1.jpg");
  cv::Mat img2 = cv::imread(imageName);
  
  CAMERA_INTRINSIC_PARAM camera = getDefaultCamera();
  cv::Matx33d intrinsic_matrix(camera.fx,0,camera.cx,0,camera.fy,camera.cy,0,0,1);
  cv::Vec4d distortion_coeffs(camera.k1,camera.k2,camera.k3,camera.k4);
  cv::Mat R = cv::Mat::eye(3,3,CV_32F);
  cv::Size image_size = img1.size();
  cv::Mat mapx = cv::Mat(image_size,CV_32FC1);
  cv::Mat mapy = cv::Mat(image_size,CV_32FC1);
  cv::fisheye::initUndistortRectifyMap(intrinsic_matrix,distortion_coeffs,R,intrinsic_matrix,image_size,CV_32FC1,mapx,mapy);
  cv::Mat t1 = img1.clone();
  cv::remap(img1,t1,mapx, mapy, cv::INTER_LINEAR);
  f1.rgb = t1;
  cv::Mat t2 = img2.clone();
  cv::remap(img2,t2,mapx, mapy, cv::INTER_LINEAR);
  f2.rgb = t2;
  
  string detector = pr.getData("detector");
  string descriptor = pr.getData("descriptor");
  computeKeypointAndDesp(f1, detector, descriptor);
  computeKeypointAndDesp(f2, detector, descriptor);
  
  vector<cv::DMatch> good_matches = getGoodMatches(f1, f2);
  
  return 0;
}