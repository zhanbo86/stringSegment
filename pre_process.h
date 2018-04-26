#ifndef textDetetcor_hpp
#define textDetetcor_hpp

#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
//#include <fstream>
//#include <bitset>
#include <time.h>


using namespace std;
using namespace cv;

#define BIGCHAR 1
#define MEDCHAR 2
#define SMALLCHAR 3

#define PIECEWIDTH 200
#define PIECEHEIGHT 100

#define WHITE 1
#define BLACK 2

#define V_PROJECT 1  //垂直投影（vertical）
#define H_PROJECT 2  //水平投影（horizational）

//#define BIGIMG 1


typedef struct
{
    int begin;
    int end;

}char_range_t;




class TextDetector{
public:
    TextDetector();
////    TextDetector(TextDetecorParams &params, std::string imgDir = "");
//    void segmentSobMor(cv::Mat &spineImage, vector<Mat> &single_char_vec, vector<Rect> &vecRect, int im_num, bool save);
//    void segmentSrcMor(cv::Mat &spineImage, vector<Mat> &single_char_vec, vector<Rect> vecContoRect, int im_num, bool save);
//    void segmentSrcPre(cv::Mat &spineImage);
//    void segmentSrcSlide(cv::Mat &spineImage, vector<Mat> &single_char_vec, int char_width, int char_height, int im_num, bool save, int &char_mat_height, int &char_mat_width);
    void segmentSrcProject_l(cv::Mat &spineGray, vector<Mat> &single_char_vec);

    cv::Mat preProcess(cv::Mat &image);
protected:
    //pre-processing
//    cv::Mat preProcess(cv::Mat &image);

    //necessary
    void adaptiveHistEqual_l(cv::Mat &src,cv::Mat &dst,double clipLimit);
    Mat preprocessChar_l(Mat in);
    Rect rectCenterScale_l(Rect rect, Size size);
    int GetTextProjection_l(Mat &src, vector<int>& pos, int mode);
//    void draw_projection_l(vector<int>& pos, int mode);
    int GetPeekRange_l(vector<int> &vertical_pos, vector<char_range_t> &peek_range, int min_thresh, int min_range);
    void segmentRow_l(cv::Mat &open_src_in, cv::Mat &out, float gap);
    void rotateImg_l(Mat source, Mat& img_rotate,float angle);
    void segmentAnalyse_l(vector<char_range_t> &peek_range_v,vector<char_range_t> &char_rang);
    void sharpenImage_l(const cv::Mat &image, cv::Mat &result);
    void removeIsoContour_l(vector<vector<Point> > &contours, vector<vector<Point> > &contours_remove);



//    void findKEdgeFirst(cv::Mat &data, int edgeValue,int k,vector<int> &rows,vector<int> &cols);
//    void findKEdgeLast(cv::Mat &data, int edgeValue,int k,vector<int> &rows,vector<int> &cols);
//    void imgQuantize(cv::Mat &src, cv::Mat &dst, double level);
//    bool verifyCharSizes(Mat r);
//    int sobelOper(const Mat &in, Mat &out, int blurSize);
//    void setMorParameters(int char_size);
//    void setThreParameters(int char_color);
//    int slidingWnd(Mat& src, vector<Mat>& wnd, Size wndSize, double x_percent, double y_percent, int &char_mat_height, int &char_mat_width);
//    float findShortestDistance(vector<Point> &contoursA_, vector<Point> &contoursB_, Point &p_a, Point &p_b);




//    int draw_main_row(vector<int>& pos,char_range_t &main_peek_rang_v);

//    void DrawBox(CvBox2D box,IplImage* img);

private:
//    string imageDirectory;
//    double NaN = nan("not a number");
    static const int DEFAULT_GAUSSIANBLUR_SIZE = 5;
    static const int SOBEL_SCALE = 1;
    static const int SOBEL_DELTA = 0;
    static const int SOBEL_DDEPTH = CV_16S;
    static const int SOBEL_X_WEIGHT = 1;
    static const int SOBEL_Y_WEIGHT = 1;
    static const int DEFAULT_MORPH_SIZE_WIDTH = 17;  // 17
    static const int DEFAULT_MORPH_SIZE_HEIGHT = 3;  // 3
//    static const int CHAR_SIZE = 20;
    static const int CHAR_SIZE = 28;
    Size src_open_val,src_dilate_val,src_erode_val;
    bool inv_bin;
    float connect_dis;
};  /*  class TextDetector */

#endif /* textDetetcor_hpp */
