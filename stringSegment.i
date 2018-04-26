/* File: example.i */
%module stringSegment

%include <opencv.i>
%cv_instantiate_all_defaults
%include "std_string.i"
%include "std_vector.i"
%include "typemaps.i"
%include "pre_process.h"

%{
#define SWIG_FILE_WITH_INIT
#include "stringSegment.h"
#include "pre_process.h"
%}



//%{
//#include <vector>
//%}

namespace std{
    %template(matVector) vector<cv::Mat>;
//    %template(pointVector) vector<vector<cv::Point>>;
}



//#include "stringSegment.h"
int Mshow(int m);
vector<Mat>  getCharsMain(Mat im);
//extern vector<cv::Mat>  getCharsMain(cv::Mat im);



