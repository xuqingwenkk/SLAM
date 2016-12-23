#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <fstream>
#include <vector>
#include <string>
using namespace std;

class ParamReader
{
public:
  map<string, string> data;
public:
  ParamReader(string filename="../param/parameters.txt")
  {
    ifstream fin(filename.c_str());
    if(!fin)
    {
      cerr << "param file does not exist! " << endl;
      return;
    }
    while(!fin.eof())
    {
      string str;
      getline(fin,str);
      if(str[0]=='#')
	continue;
      int pos=str.find("=");
      if(pos == -1)
	continue;
      string key = str.substr(0, pos);
      string value = str.substr(pos+1, str.length());
      data[key] = value;
      if(!fin.good())
	break;
    }
  }
  string getData(string key)
  {
    map<string, string>::iterator iter = data.find(key);
    if(iter == data.end())
    {
      cerr << key << "is not found !" << endl;
      return string("NOT FOUND! ");
    }
    return iter->second;
  }
};

vector<cv::Point2f> FeatureExtraction(cv::Mat & image, cv::Size & board_size, int & image_index);
void FisheyeCalib(vector<vector<cv::Point2f> >  corners_Seq, vector<vector<cv::Point3f> > & object_Points, cv::Size & image_size, cv::Matx33d & intrinsic_matrix, cv::Vec4d & distortion_coeffs, 
  vector<cv::Vec3d> & rotation_vectors, vector<cv::Vec3d> & translation_vectors, int & successImageNum, cv::Size & board_size);

void Estimation(vector<vector<cv::Point3f> > & object_Points, cv::Matx33d & intrinsic_matrix, cv::Vec4d & distortion_coeffs, vector<cv::Vec3d> & rotation_vectors, 
		vector<cv::Vec3d> & translation_vectors, int & image_count, cv::Size & board_size, vector<vector<cv::Point2f> > & corners_Seq);