#ifndef CHAR_RECOGNISE_HPP
#define CHAR_RECOGNISE_HPP

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include "pre_process.h"

//extern vector<Mat>  getCharsMain(Mat im);

class StringSegment
{
public:
    StringSegment();
    ~StringSegment();
    std::vector<cv::Mat> getChars(cv::Mat im);
    int show(int m);
    static const int CHAR_SIZE = 20;
};

int Mshow(int m);
vector<Mat>  getCharsMain(Mat im);

#endif  // CHAR_RECOGNISE_HPP
