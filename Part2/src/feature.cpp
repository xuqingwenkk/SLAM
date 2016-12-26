#include "sfm.h"
#include <sstream>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/nonfree/nonfree.hpp> //SIFT and SURF

int main(int argc, char** argv)
{
  ParamReader pr;
  CAMERA_INTRINSIC_PARAM camera = getDefaultCamera();
  cv::Matx33d intrinsic_matrix(camera.fx,0,camera.cx,0,camera.fy,camera.cy,0,0,1);
  cv::Vec4d distortion_coeffs(camera.k1,camera.k2,camera.k3,camera.k4);
  cv::Mat R = cv::Mat::eye(3,3,CV_32F);
  
  string image_dir = pr.getData("image_dir");
  int test_image_num = atoi(pr.getData("test_image_num").c_str());
  vector<cv::Mat> imageSet;
  for(int i = 0; i < test_image_num; i++)
  {
    string imageName;
    stringstream ss;
    ss << i;
    ss >> imageName;
    imageName += ".jpg";
    imageName = "test_" + imageName;
    imageName = image_dir + imageName;
    cv::Mat temp = cv::imread(imageName);
    cv::Size image_size = temp.size();
    cv::Mat mapx = cv::Mat(image_size,CV_32FC1);
    cv::Mat mapy = cv::Mat(image_size,CV_32FC1);
    cv::fisheye::initUndistortRectifyMap(intrinsic_matrix,distortion_coeffs,R,intrinsic_matrix,image_size,CV_32FC1,mapx,mapy);
    cv::Mat t = temp.clone();
    cv::remap(temp,t,mapx, mapy, cv::INTER_LINEAR);
    imageSet.push_back(t);
    cv::namedWindow("Display raw image", cv::WINDOW_NORMAL);
    cv::imshow("Display raw image", t);
    cv::waitKey(0);
  }
  
  cv::Ptr<cv::FeatureDetector> _detector;
  cv::Ptr<cv::DescriptorExtractor> _descriptor;
  
  cv::initModule_nonfree();
  _detector = cv::FeatureDetector::create("SIFT");
  _descriptor = cv::DescriptorExtractor::create("SIFT");
  if(!_detector || !_descriptor)
  {
    cout << " Invalid type !  " << endl;
  }
  
  for(int i = 0; i < test_image_num - 1; i = i + 2)
  {
    cv::Mat desp1, desp2;
    vector<cv::KeyPoint> kp1, kp2;
    _detector->detect(imageSet[i], kp1);
    _detector->detect(imageSet[i+1], kp2);
    _descriptor->compute(imageSet[i], kp1, desp1);
    _descriptor->compute(imageSet[i+1], kp2, desp2);
    
    vector<cv::DMatch> matches;
    cv::FlannBasedMatcher matcher;
    matcher.match(desp1, desp2, matches);
    cout << "Find " << matches.size() << " matches" << endl;
    
    cv::Mat imgMatches;
    cv::drawMatches(imageSet[i], kp1, imageSet[i+1], kp2, matches, imgMatches);
    cv::namedWindow("matches", cv::WINDOW_NORMAL);
    cv::imshow("matches", imgMatches);
    cv::waitKey(0);
    
    // filter good matches
    vector<cv::DMatch> good_matches;
    double minDis = 9999;
    for(size_t i = 0; i < matches.size(); i++)
    {
      if(matches[i].distance < minDis)
	minDis = matches[i].distance;
    }
    for(size_t i = 0; i < matches.size(); i++)
    {
      if(matches[i].distance < 4*minDis)
	good_matches.push_back(matches[i]);
    }
    cout << "good matches = " << good_matches.size() << endl;
    cv::drawMatches(imageSet[i], kp1, imageSet[i+1], kp2, good_matches, imgMatches);
    cv::namedWindow("good matches", cv::WINDOW_NORMAL);
    cv::imshow("good matches", imgMatches);
    cv::waitKey(0);
    
  }
  
  return 0;
}