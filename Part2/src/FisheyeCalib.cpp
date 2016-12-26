#include "sfm.h"
#include <sstream>

int main()
{
  ParamReader pr;
  string output_dir = pr.getData("output_dir");
  ofstream fout(output_dir);
  cout << "begin to extract corners...." << endl;
  int row = atoi(pr.getData("bs_row").c_str());
  int col = atoi(pr.getData("bs_col").c_str());
  cv::Size board_size = cv::Size(row, col);
  int image_count = atoi(pr.getData("image_count").c_str());
  vector<vector<cv::Point2f> >  corners_Seq;
  vector<cv::Mat>  image_Seq;
  int successImageNum = 0;       //the number of image which has been successfully extracted
  int count = 0;    
  for(int i = 0; i < image_count; i++)
  {
    cout << "Frame # " << i+1 << "..." << endl;
    string image_dir = pr.getData("image_dir");
    string imageFileName;
    stringstream StrStm;
    StrStm << i+1;
    StrStm >> imageFileName;
    imageFileName += ".jpg";
    imageFileName = image_dir + imageFileName;
    cv::Mat image = cv::imread(imageFileName);
    vector<cv::Point2f> corners = FeatureExtraction(image, board_size, i);
    corners_Seq.push_back(corners);
    count = count + corners.size();
    successImageNum = successImageNum + 1;
    image_Seq.push_back(image);
  }
  cout << "Extraction Done! " << endl << "Begin Calibration! " << endl;
  
  vector<vector<cv::Point3f> >  object_Points;
  cv::Size image_size = image_Seq[0].size();
  cv::Matx33d intrinsic_matrix;    
  cv::Vec4d distortion_coeffs;     //k1,k2,k3,k4
  vector<cv::Vec3d> rotation_vectors;  
  vector<cv::Vec3d> translation_vectors;  
  FisheyeCalib(corners_Seq, object_Points, image_size, intrinsic_matrix, distortion_coeffs, rotation_vectors, translation_vectors, successImageNum, board_size);
  cout << "Calibration Done! " << endl << "Begin Estimation! " << endl;
  
  Estimation(object_Points, intrinsic_matrix, distortion_coeffs, rotation_vectors, translation_vectors, image_count, board_size, corners_Seq);
  cv::Mat rotation_matrix = cv::Mat(3,3,CV_32FC1, cv::Scalar::all(0));
  fout << "intrinsic matrix is " << endl;
  fout << intrinsic_matrix << endl;
  fout << "distortion coeffients are " << endl;
  fout << distortion_coeffs << endl;
  for (int i=0; i< image_count; i++)
  {
    cv::Rodrigues(rotation_vectors[i], rotation_matrix);
    fout << "The rotation matrix of image " << i << "is: " << endl;
    fout << rotation_matrix << endl;
    fout << "The translation vector of image " << i << "is: " << endl;
    fout << translation_vectors[i] << endl;
  }
  cout << "Save Done !" << endl << "Begin Undistortion! " << endl;
  
  cv::Mat mapx = cv::Mat(image_size, CV_32FC1);
  cv::Mat mapy = cv::Mat(image_size, CV_32FC1);
  cv::Mat R = cv::Mat::eye(3,3,CV_32F);
  for(int i = 0; i < image_count; i++)
  {
    cout << "Frame # " << i+1 << "..." << endl;
    cv::fisheye::initUndistortRectifyMap(intrinsic_matrix, distortion_coeffs, R, intrinsic_matrix, image_size, CV_32FC1, mapx, mapy); //In case of a monocular camera, newCameraMatrix is usually equal to cameraMatrix
    cv::Mat t = image_Seq[i].clone();
    cv::remap(image_Seq[i], t, mapx, mapy, cv::INTER_LINEAR);
    string image_dir = pr.getData("savedir");
    string imageFileName;
    stringstream StrStm;
    StrStm << i+1;
    StrStm >> imageFileName;
    imageFileName += "_d.jpg";
    imageFileName = image_dir + imageFileName;
    cv::imwrite(imageFileName, t);
  }
  cout << "Save Done! " << endl;
  
  return 0;
}