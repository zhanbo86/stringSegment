#include "stringSegment.h"

using namespace cv;
using namespace std;


StringSegment::StringSegment()
{
}

StringSegment::~StringSegment() {}


int StringSegment::show(int m)
{
    return m;
}

int Mshow(int m)
{
    StringSegment st;
    return st.show(m);
}

vector<Mat>  StringSegment::getChars(Mat im)
{
    std::cout << "preproceed imgs." << std::endl;
    Size low_res = cv::Size(PIECEWIDTH,PIECEHEIGHT);
    Mat img_100(low_res,im.depth(),1);
    if (im.empty())
    {
        std::cout << "Cannot open source image!" << std::endl;
        exit;
    }
    cv::resize(im,img_100,low_res,0,0,CV_INTER_LINEAR);
    std::cout<<"img width = "<<img_100.size().width<<" , "<<" height = "<<img_100.size().height
            <<" , "<<"depth = "<<img_100.depth()<<" , "<<"channel = "<<img_100.channels()<<std::endl;

    while(1)
    {
        imshow( "img_100", img_100 );
        if(char(cvWaitKey(15))==27)break;
    }
    cvDestroyWindow("img_100");

    ////pre-process image
    clock_t a=clock();
    TextDetector detector;
    vector<Mat> single_char_vec;
    single_char_vec.clear();
    detector.segmentSrcProject_l(img_100, single_char_vec);
    return single_char_vec;
}


vector<Mat>  getCharsMain(Mat im)
{
    StringSegment StringSeg;
    return StringSeg.getChars(im);
}

