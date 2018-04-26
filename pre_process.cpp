#include "pre_process.h"

using namespace std;
using namespace cv;



TextDetector::TextDetector(){    
    inv_bin = false;
    connect_dis = 8;
}


cv::Mat TextDetector::preProcess(cv::Mat &image){
    cv::Mat gray;
    cvtColor(image, gray, CV_BGR2GRAY);
    return gray;
}


void TextDetector::segmentRow_l(cv::Mat &open_src_in,cv::Mat &out,float gap)
{
    vector<int> pos_h;
    pos_h.resize(open_src_in.rows,0);
    vector<char_range_t> peek_range_h;
    char_range_t main_peek_rang_h;
    GetTextProjection_l(open_src_in,pos_h,H_PROJECT);
//    draw_projection_l(pos_h,H_PROJECT);
    GetPeekRange_l(pos_h,peek_range_h,2,5);
    //connect neighgour peek
#ifdef DEBUG
    std::cout<<"peek_range.size = "<<peek_range_h.size()<<std::endl;
#endif
    if(peek_range_h.size()==0)
    {
        while(1)
        {
          if(char(cvWaitKey(15))==27)break;
        }
    }


    //connect two neighbour ranges when their distance is shorter than a threshould
//    for(int i =0;i<peek_range_h.size()-1;i++)
//    {
//#ifdef DEBUG
//        std::cout<<"peek_range = "<< peek_range_h.at(i).begin<<" ~ "<< peek_range_h.at(i).end<<std::endl;
//#endif
//        int peek_rang = peek_range_h.at(i).end - peek_range_h.at(i).begin;
//        int peek_rang_next = peek_range_h.at(i+1).end - peek_range_h.at(i+1).begin;
//        int peek_rang_gap = peek_range_h.at(i+1).begin - peek_range_h.at(i).end;
//        if((peek_rang>0.01*PIECEHEIGHT)&&(peek_rang_next>0.01*PIECEHEIGHT)&&(peek_range_h.at(i+1).end<0.95*PIECEHEIGHT)
//                &&(peek_range_h.at(i).begin>0.05*PIECEHEIGHT))
//        {
//           if(peek_rang_gap <gap*PIECEHEIGHT)//gap distance
//           {
//               peek_range_h.at(i+1).begin = peek_range_h.at(i).begin;
//               peek_range_h.at(i+1).end = peek_range_h.at(i+1).end;
//           }
//        }
//    }


    //find main peek
    main_peek_rang_h.begin = peek_range_h.at(0).begin;
    main_peek_rang_h.end = peek_range_h.at(peek_range_h.size()-1).end;
//    main_peek_rang_h = peek_range_h.at(0);
//    int peek_main_scale = main_peek_rang_h.end - main_peek_rang_h.begin;
#ifdef DEBUG
    std::cout<<"peek_range after connect.size = "<<peek_range_h.size()<<std::endl;
#endif
//    //find the longest peek
//    for(int i =0;i<peek_range_h.size();i++)
//    {
//        #ifdef DEBUG
//        std::cout<<"peek_range after connect = "<< peek_range_h.at(i).begin<<" ~ "<< peek_range_h.at(i).end<<std::endl;
//        #endif
//        int peek_scale = peek_range_h.at(i).end - peek_range_h.at(i).begin;
//        if(peek_scale>peek_main_scale)
//        {
//            main_peek_rang_h = peek_range_h.at(i);
//            peek_main_scale = main_peek_rang_h.end - main_peek_rang_h.begin;
//        }
//    }
#ifdef DEBUG
    std::cout<<"main_peek_rang_h = "<<main_peek_rang_h.begin<<" ~ "<<main_peek_rang_h.end<<std::endl;
#endif
    Rect mainRow(0,main_peek_rang_h.begin,open_src_in.cols,main_peek_rang_h.end-main_peek_rang_h.begin);
    Mat charRow(open_src_in, mainRow);//modify on 0309
    out = charRow;
}








//void TextDetector::findKEdgeFirst(cv::Mat &data, int edgeValue,int k,vector<int> &rows,vector<int> &cols){
//    int count = 0;
//    for (int i = 0; i < data.cols; i ++) {
//        uchar *u = data.ptr<uchar>(i);
//        for (int j = 0; j < data.rows; j ++) {
//            if(edgeValue == (int)u[j]){
//                if(count < k){
//                    count ++;
//                    cols.push_back(i);
//                    rows.push_back(j);
//                }

//            }

//        }
//    }

//}

//void TextDetector::findKEdgeLast(cv::Mat &data, int edgeValue,int k,vector<int> &rows, vector<int> &cols){
//    int count = 0;
//    for (int i = data.cols - 1; i >= 0; i --) {
//        uchar *u = data.ptr<uchar>(i);
//        for (int j = data.rows - 1; j >= 0; j --) {
//            if(edgeValue == (int)u[j]){
//                if(count < k){
//                    count ++;
//                    cols.push_back(i);
//                    rows.push_back(j);
//                }

//            }
//        }

//    }

//}

void TextDetector::adaptiveHistEqual_l(cv::Mat &src,cv::Mat &dst,double clipLimit)
{
    Ptr<cv::CLAHE> clahe = createCLAHE();
    clahe->setClipLimit(clipLimit);
    clahe->apply(src, dst);
}



//获取文本的投影以用于分割字符(垂直，水平),默认图片是黑底白色
int TextDetector::GetTextProjection_l(Mat &src, vector<int>& pos, int mode)
{
    if (mode == H_PROJECT)
    {
        for (int i = 0; i < src.rows; i++)
        {

            for (int j = 0; j < src.cols; j++)
            {
                if (src.at<uchar>(i, j) == 255)
                {
                    pos[i]++;
                }
            }
        }

    }
    else if (mode == V_PROJECT)
    {
        for (int i = 0; i < src.cols; i++)
        {

            for (int j = 0; j < src.rows; j++)
            {
                if (src.at<uchar>(j, i) == 255)
                {
                    pos[i]++;
                }
            }
        }
    }

    return 0;
}


//void TextDetector::draw_projection_l(vector<int>& pos,int mode)
//{
//    vector<int>::iterator max = std::max_element(std::begin(pos), std::end(pos)); //求最大值
//    if (mode == H_PROJECT)
//    {
//        int height = pos.size();
//        int width = *max;
//        Mat project;
//        project = Mat::zeros(height, width, CV_8UC1);
//        for (int i = 0; i < height; i++)
//        {
//            for (int j = 0; j < pos[i]; j++)
//            {
//                project.at<uchar>(i, j) = 255;
//            }
////            std::cout<<"pos"<<i<<" = "<<pos[i]<<std::endl;

//        }
//#ifdef DEBUG
//        while(1)
//        {
//          imshow("horizational projection", project);
//          if(char(cvWaitKey(15))==27)break;
//        }
//        cvDestroyWindow("horizational projection");
//#endif
//    }
//    else if (mode == V_PROJECT)
//    {
//        int height = *max;
//        int width = pos.size();
//        Mat project = Mat::zeros(height, width, CV_8UC1);
//        for (int i = 0; i < project.cols; i++)
//        {
//            for (int j = project.rows - 1; j >= project.rows - pos[i]; j--)
//            {
//                //std::cout << "j:" << j << "i:" << i << std::endl;
//                project.at<uchar>(j, i) = 255;
//            }
//        }
//#ifdef DEBUG
//        while(1)
//        {
//          imshow("verticle projection", project);
//          if(char(cvWaitKey(15))==27)break;
//        }
////        cvDestroyWindow("verticle projection");
//#endif
//    }
//}




int TextDetector::GetPeekRange_l(vector<int> &horizental_pos, vector<char_range_t> &peek_range, int min_thresh = 2, int min_range = 5)
{
    int begin = 0;
    int end = 0;
//    std::cout<<"horizental_pos.size() = "<<horizental_pos.size()<<std::endl;
    for (int i = 0; i < horizental_pos.size(); i++)
    {
        if (horizental_pos[i] > min_thresh && begin == 0)
        {
            begin = i;
        }
        else if (horizental_pos[i] > min_thresh && begin != 0)
        {
            if(i == (horizental_pos.size()-1))
            {
                end = horizental_pos.size()-1;
                char_range_t tmp;
                tmp.begin = begin;
                tmp.end = end;
                peek_range.push_back(tmp);
                begin = 0;
                end = 0;
            }
            else
            {
                continue;
            }
        }
        else if (horizental_pos[i] < min_thresh && begin != 0)
        {
            end = i;
            if (end - begin >= min_range)
            {
                char_range_t tmp;
                tmp.begin = begin;
                tmp.end = end;
                peek_range.push_back(tmp);
                begin = 0;
                end = 0;
            }
            else
            {
                begin = 0;
                end = 0;
            }

        }
        else if (horizental_pos[i] < min_thresh || begin == 0)
        {
            continue;
        }
        else
        {
            //printf("raise error!\n");
        }
    }

    return 0;
}


void TextDetector::sharpenImage_l(const cv::Mat &image, cv::Mat &result)
{
    //创建并初始化滤波模板
    cv::Mat kernel(3,3,CV_32F,cv::Scalar(0));
    kernel.at<float>(1,1) = 5.0;
    kernel.at<float>(0,1) = -1.0;
    kernel.at<float>(1,0) = -1.0;
    kernel.at<float>(1,2) = -1.0;
    kernel.at<float>(2,1) = -1.0;

    result.create(image.size(),image.type());

    //对图像进行滤波
    cv::filter2D(image,result,image.depth(),kernel);
}


//bool TextDetector::verifyCharSizes(Mat r) {
//  // Char sizes 45x90
//  float aspect = 45.0f / 90.0f;
//  float charAspect = (float)r.cols / (float)r.rows;
//  float error = 0.7f;
//  float minHeight = 10.f;
//  float maxHeight = 35.f;
//  // We have a different aspect ratio for number 1, and it can be ~0.2
//  float minAspect = 0.05f;
//  float maxAspect = aspect + aspect * error;
//  // area of pixels
//  int area = cv::countNonZero(r);
//  // bb area
//  int bbArea = r.cols * r.rows;
//  //% of pixel in area
//  int percPixels = area / bbArea;

//  if (percPixels <= 1 && charAspect > minAspect && charAspect < maxAspect &&
//      r.rows >= minHeight && r.rows < maxHeight)
//    return true;
//  else
//    return false;
//}



//int TextDetector::sobelOper(const Mat &in, Mat &out, int blurSize)
//{
//  Mat mat_blur;
//  mat_blur = in.clone();
//  GaussianBlur(in, mat_blur, Size(blurSize, blurSize), 0, 0, BORDER_DEFAULT);

//  Mat mat_gray;
//  if (mat_blur.channels() == 3)
//    cvtColor(mat_blur, mat_gray, CV_RGB2GRAY);
//  else
//    mat_gray = mat_blur;

//  int scale = SOBEL_SCALE;
//  int delta = SOBEL_DELTA;
//  int ddepth = SOBEL_DDEPTH;

//  Mat grad_x, grad_y;
//  Mat abs_grad_x, abs_grad_y;

//  Sobel(mat_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
//  convertScaleAbs(grad_x, abs_grad_x);
//  Sobel(mat_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
//  convertScaleAbs(grad_y, abs_grad_y);
//  Mat grad;
//  addWeighted(abs_grad_x, SOBEL_X_WEIGHT, abs_grad_y, SOBEL_X_WEIGHT, 0, grad);
//  out = grad;
//  return 0;
//}


Mat TextDetector::preprocessChar_l(Mat in) {
  // Remap image
  int h = in.rows;
  int w = in.cols;

  int charSize = CHAR_SIZE;
  Mat transformMat = Mat::eye(2, 3, CV_32F);
  int m = max(w, h);
  transformMat.at<float>(0, 2) = float(m / 2 - w / 2);
  transformMat.at<float>(1, 2) = float(m / 2 - h / 2);
  Mat warpImage(m, m, in.type());
  warpAffine(in, warpImage, transformMat, warpImage.size(), INTER_LINEAR,
             BORDER_CONSTANT, Scalar(0));
  Mat out;
  resize(warpImage, out, Size(charSize, charSize));
  return out;
}

//void TextDetector::setMorParameters(int char_size)
//{
//    switch(char_size)
//    {
//        case BIGCHAR:
//           src_open_val = Size(5, 5);
//           src_dilate_val = Size(10, 20);
//           src_erode_val = Size(5, 5);
//           connect_dis = 8;
//        break;
//        case MEDCHAR:
//            src_open_val = Size(5, 5);
//            src_dilate_val = Size(5, 13);
//            src_erode_val = Size(5, 5);
//            connect_dis = 3;
//        break;
//        case SMALLCHAR:
//            src_open_val = Size(3, 5);
//            src_dilate_val = Size(1, 8);
//            src_erode_val = Size(1, 5);
//            connect_dis = 2;
//        break;
//        default:
//           std::cout<<"char size input is wrong!!! use default big char parameters."<<std::endl;
//           src_open_val = Size(5, 5);
//           src_dilate_val = Size(5, 20);
//           src_erode_val = Size(5, 5);
//           connect_dis = 7;
//        break;
//    }
//}


//int TextDetector::slidingWnd(Mat& src, vector<Mat>& wnd,Size wndSize, double x_percent, double y_percent,
//                             int &char_mat_height,int &char_mat_width)
//{
//    std::cout<<"size = "<<wndSize<<std::endl;
//    int count = 0;  //记录滑动窗口的数目
//    int x_step = cvCeil(x_percent*wndSize.width);
//    int y_step = cvCeil(y_percent*wndSize.height);
////    int x_step = 1;
////    int y_step = 1;
//    int64 count1 = getTickCount();
//    double freq = getTickFrequency();
//    std::cout<<"picece_orc size is "<<src.rows<<" * "<<src.cols<<std::endl;
//    int rows_count=0;
//    int cols_count=0;

//    //利用窗口对图像进行遍历
//    for (int i = 0; i < src.rows- wndSize.height; i+=y_step)
//    {
//        rows_count++;
//        cols_count = 0;
//        for (int j = 0; j < src.cols- wndSize.width; j+=x_step)
//        {
//            Rect roi(Point(j, i), wndSize);
//            Mat ROI = src(roi);
//            wnd.push_back(ROI);
//            count++;
//            cols_count++;

////            cv::Mat idx;
////            findNonZero(ROI, idx);
////            int one_count = (int)idx.total();

////            int zero_count = (int)ROI.total() - one_count;
////            float one_percent = (float)one_count/(float)(one_count+zero_count);
//////            std::cout<<"one_count = "<<one_count<<std::endl;
//////            std::cout<<"zero_count = "<<zero_count<<std::endl;
//////            std::cout<<"one_percent = "<<one_percent<<std::endl;
////            if(one_percent>0.1)
////            {
////                 wnd.push_back(ROI);
////                 count++;
////            }
////            else
////            {
////                single_char_precise.at<uchar>(i,j) = 0;
////            }
//        }
//    }
//    char_mat_height = rows_count;
//    char_mat_width = cols_count;

//    int64 count2 = getTickCount();
//    double time = (count2 - count1) / freq;
//    cout << "slide Time=" << time * 100 << "ms"<<endl;
//    return count;
//}

Rect TextDetector::rectCenterScale_l(Rect rect, Size size)
{
    rect = rect + size;
    Point pt;
    pt.x = cvRound(size.width/2.0);
    pt.y = cvRound(size.height/2.0);
    return (rect-pt);
}


void TextDetector::removeIsoContour_l(vector<vector<Point> > &contours,vector<vector<Point> > &contours_remove)
{
    ////detect contous neiboughbour
    vector<vector<Point> >::iterator itc = contours.begin();
    vector<vector<Point> >::iterator itc2 = contours.begin();
    vector<vector<Point> >::iterator itc_next = contours.begin();
    while (itc != contours.end())
    {
        Rect mr = boundingRect(Mat(*itc));
//        Rect mr_3zoom = rectCenterScale(mr,Size(9*mr.width,2*mr.height));
        Rect mr_3zoom = rectCenterScale_l(mr,Size(10*mr.width,5*mr.height));
        itc_next = contours.begin();
        long int mr_cross_acc_width = 0;
        long int mr_cross_acc_height = 0;
        while(itc_next != contours.end())
        {
            if(itc_next==itc)
            {
                itc_next++;
                continue;
            }
            Rect mr_next = boundingRect(Mat(*itc_next));
            Rect mr_cross = mr_3zoom&mr_next;
            mr_cross_acc_width += mr_cross.width;
            mr_cross_acc_height += mr_cross.height;
            itc_next++;
            if((mr_cross_acc_height!=0)||(mr_cross_acc_width!=0))
            {
                break;
            }
        }
        if((mr_cross_acc_width==0)&&(mr_cross_acc_height==0))
        {
            contours_remove.push_back(*itc);
            itc2 = contours.erase(itc);
            itc = itc2;
#ifdef DEBUG
            std::cout<<"erase this contour for obsolute!!!"<<std::endl;
#endif
        }
        else
        {
            ++itc;
        }
    }
}

//float TextDetector::findShortestDistance(vector<Point> &contoursA_, vector<Point> &contoursB_, Point &p_a, Point &p_b)
//{
//    float distance=0;
//    float min_distance=200;
//    vector<Point> contoursA = contoursA_;
//    vector<Point> contoursB = contoursB_;
//    vector<Point>::iterator itc_a = contoursA.begin();
//    vector<Point>::iterator itc_b = contoursB.begin();
//    while (itc_a != contoursA.end())
//    {
//        itc_b = contoursB.begin();
//        while (itc_b != contoursB.end())
//        {
//            distance = sqrt(pow(((*itc_a).x - (*itc_b).x),2)+pow(((*itc_a).y - (*itc_b).y),2));
//            if(distance < min_distance)
//            {
//                min_distance = distance;
//                p_a = *itc_a;
//                p_b = *itc_b;
//            }
//            itc_b++;
//        }
//        itc_a++;
//    }
//    return min_distance;
//}


////segment the spine text
//void TextDetector::segmentSrcSlide(cv::Mat &spineGray, vector<Mat> &single_char_vec,
//                                   int char_width, int char_height, int im_num, bool save,
//                                   int &char_mat_height,int &char_mat_width)
//{
//    srand((unsigned)time(NULL));
//    ////gauss smoothing
//    int m_GaussianBlurSize = 5;
//    Mat mat_blur;
//    GaussianBlur(spineGray, mat_blur, Size(m_GaussianBlurSize, m_GaussianBlurSize), 0, 0, BORDER_DEFAULT);
//#ifdef DEBUG
//    while(1)
//    {
//      imshow("src_gauss", mat_blur);
//      if(char(cvWaitKey(15))==27)break;
//    }
//#endif

//    ////histequal and sharpen
//    Mat spineGrayTemp = mat_blur - 0.5;
//    cv::Mat spineAhe;
//    adaptiveHistEqual(spineGrayTemp, spineAhe, 0.01);
//    cv::Mat spineShrpen;
//    sharpenImage(spineAhe, spineShrpen);
//#ifdef DEBUG
//    while(1)
//    {
//      imshow("sharpen", spineShrpen);
//      if(char(cvWaitKey(15))==27)break;
//    }
//#endif

//    ////threshold
//    cv::Mat thresh_src;
//    if(inv_bin)
//    {
//        threshold(spineShrpen, thresh_src, 0, 255, THRESH_OTSU+ CV_THRESH_BINARY_INV);
//    }
//    else
//    {
//        threshold(spineShrpen, thresh_src, 0, 255, THRESH_OTSU+ CV_THRESH_BINARY);
//    }
//#ifdef DEBUG
//    while(1)
//    {
//      imshow("thresh_src", thresh_src);
//      if(char(cvWaitKey(15))==27)break;
//    }
//#endif

//    ////slide window in src
//    vector<Mat> charWnd;
//    int count=slidingWnd(thresh_src, charWnd,Size(char_width, char_height),0.1,0.1,char_mat_height,char_mat_width);
//    std::cout<<"slide count is "<<count<<std::endl;


//    ////save single char image after segment
//    for(int char_num=0;char_num<charWnd.size();char_num++)
//    {
//         Mat single_char_=charWnd.at(char_num);
//         Mat single_char;
//         single_char = preprocessChar(single_char_);
//         single_char_vec.push_back(single_char);
//        if(save)
//        {
//            const char* single_char_folder_ = "../../../src/easyocr/char_img";
//            std::stringstream ss(std::stringstream::in | std::stringstream::out);
//            ss << single_char_folder_ << "/" << im_num << "_src" << char_num/*<<"_"<<rand()*/<< ".jpg";
//            imwrite(ss.str(),single_char);
//        }
//#ifdef DEBUG
//        while(1)
//        {
//          imshow( "single_char", single_char_ );
//          if(char(cvWaitKey(15))==27)break;
//        }
//#endif
//    }

//#ifdef DEBUG
//    cvDestroyWindow("sharpen");
//    cvDestroyWindow("src_gauss");
//    cvDestroyWindow("thresh_src");
//    cvDestroyWindow("single_char");
//#endif
//}



////segment the spine text
//void TextDetector::segmentSrcPre(cv::Mat &spineGray)
//{
//    ////gauss smoothing
//    int m_GaussianBlurSize = 5;
//    Mat mat_blur;
//    GaussianBlur(spineGray, mat_blur, Size(m_GaussianBlurSize, m_GaussianBlurSize), 0, 0, BORDER_DEFAULT);

//    ////histequal and sharpen
//    Mat spineGrayTemp = mat_blur - 0.5;
//    cv::Mat spineAhe;
//    adaptiveHistEqual(spineGrayTemp, spineAhe, 0.01);
//    cv::Mat spineShrpen;
//    sharpenImage(spineAhe, spineShrpen);

//    ////threshold
//    cv::Mat thresh_src;
//    if(inv_bin)
//    {
//        threshold(spineShrpen, thresh_src, 0, 255, THRESH_OTSU+ CV_THRESH_BINARY_INV);
//    }
//    else
//    {
//        threshold(spineShrpen, thresh_src, 0, 255, THRESH_OTSU+ CV_THRESH_BINARY);
//    }


//    ////morphological open
//    setMorParameters(1);
//    Mat element_src = getStructuringElement(MORPH_RECT, src_open_val);
//    Mat open_src;
//    morphologyEx(thresh_src,open_src,MORPH_OPEN,element_src);
//    cv::Mat thres_window = thresh_src.clone();



//    ////find contours
//    Mat img_contours;
//    thres_window.copyTo(img_contours);
//    vector<vector<Point> > contours;
//    findContours(img_contours,
//                 contours,               // a vector of contours
//                 CV_RETR_LIST,       // retrieve all contours
//                 CV_CHAIN_APPROX_NONE);  // all pixels of each contours
//    Mat sepertate_im(thres_window.size(),thres_window.depth(),Scalar(255));
//    drawContours(sepertate_im,contours,-1,Scalar(0),2);
//#ifdef DEBUG
//    while(1)
//    {
//      imshow("sepertate_im", sepertate_im);
//      if(char(cvWaitKey(15))==27)break;
//    }
//#endif


//    ////rebuild src image

//    Mat rebuilt_src(spineGray.size(),spineGray.depth(),Scalar(0));
//    for(int i=0;i<rebuilt_src.rows;i++)
//    {
//        for(int j=0;j<rebuilt_src.cols;j++)
//        {
//           Point pt;
//           pt.x = j;
//           pt.y = i;
//           int inner = 0;

//           for(vector<vector<Point> >::iterator itc = contours.begin();itc != contours.end();itc++)
//           {
//               int result = pointPolygonTest(*itc,pt,false);
//               if(result==1)
//               {
//                   inner++;
//               }
//               else if(result==0)
//               {
//                   rebuilt_src.at<uchar>(i,j) = 255;
//               }

//           }
//           if(inner==2)
//               rebuilt_src.at<uchar>(i,j) = 255;
//        }
//    }

//    spineGray = rebuilt_src;

//#ifdef DEBUG
//    while(1)
//    {
//      imshow("rebuilt_src", rebuilt_src);
//      if(char(cvWaitKey(15))==27)break;
//    }
//#endif


//#ifdef DEBUG
//    thres_window.release();
//    cvDestroyWindow("rebuilt_src");
//    cvDestroyWindow("sepertate_im");
//#endif
//}



////segment the spine text
//void TextDetector::segmentSrcMor(cv::Mat &spineGray, vector<Mat> &single_char_vec, vector<Rect> vecContoRect,int im_num, bool save)
//{
//    srand((unsigned)time(NULL));
//    ////set parameters
//    int char_size;
//#ifdef DEBUG
//    printf("please input char size: big is 1, mediate is 2, small is 3\n");
//    scanf("%d",&char_size);
//#endif
//    setMorParameters(char_size);


//    ////gauss smoothing
//    int m_GaussianBlurSize = 5;
//    Mat mat_blur;
//    GaussianBlur(spineGray, mat_blur, Size(m_GaussianBlurSize, m_GaussianBlurSize), 0, 0, BORDER_DEFAULT);

//    ////histequal and sharpen
//    Mat spineGrayTemp = mat_blur - 0.5;
//    cv::Mat spineAhe;
//    adaptiveHistEqual(spineGrayTemp, spineAhe, 0.01);
//    cv::Mat spineShrpen;
//    sharpenImage(spineAhe, spineShrpen);

//    ////threshold
//    cv::Mat thresh_src;
//    threshold(spineShrpen, thresh_src, 0, 255, THRESH_OTSU+ CV_THRESH_BINARY);
//    cv::Mat idx;
//    findNonZero(thresh_src, idx);
//    int one_count = (int)idx.total();
//    float one_percent = (float)one_count/(float)thresh_src.total();
//    std::cout<<"one_percent = "<<one_percent<<std::endl;
//    if(one_percent>0.6)
//    {
//        inv_bin = true;
//    }
//    else
//    {
//        inv_bin = false;
//    }

//    cv::Mat thresh_src_temp;
//    if(inv_bin)
//    {
//        threshold(spineShrpen, thresh_src_temp, 0, 255, THRESH_OTSU+ CV_THRESH_BINARY_INV);
//        thresh_src = thresh_src_temp;
//    }

////    while(1)
////    {
////      imshow("window_thresh", thresh_src);
////      if(char(cvWaitKey(15))==27)break;
////    }


//    ////morphological open
//    Mat element_src = getStructuringElement(MORPH_RECT, src_open_val);
//    Mat open_src;
//    morphologyEx(thresh_src,open_src,MORPH_OPEN,element_src);
//    cv::Mat thres_window = open_src.clone();

//    ////find contours
//    Mat img_contours;
//    thres_window.copyTo(img_contours);
//    vector<vector<Point> > contours;
//    findContours(img_contours,
//                 contours,               // a vector of contours
//                 CV_RETR_EXTERNAL,       // retrieve the external contours
//                 CV_CHAIN_APPROX_NONE);  // all pixels of each contours
//    Mat sepertate_im(thres_window.size(),thres_window.depth(),Scalar(255));
//    drawContours(sepertate_im,contours,-1,Scalar(0),2);
////    while(1)
////    {
////      imshow("sepertate_im",sepertate_im);
////      if(char(cvWaitKey(15))==27)break;
////    }

//    ////remove isolate contours
//    Mat sepertate_im_remove(thres_window.size(),thres_window.depth(),Scalar(255));
//    //移除过长或过短的轮廓
//    int cmin = 1; //最小轮廓长度
//    int cmax = 600;    //最大轮廓
//    vector<vector<Point> >::iterator itc_mm = contours.begin();
//    while (itc_mm!=contours.end())
//    {
//        if (itc_mm->size() < cmin || itc_mm->size() > cmax)
//        {
//           itc_mm = contours.erase(itc_mm);
//        }
//        else
//            ++itc_mm;
//    }
////    removeIsoContour(contours);
//    drawContours(sepertate_im_remove,contours,-1,Scalar(0),2);
////    while(1)
////    {
////      imshow("sepertate_im_remove",sepertate_im_remove);
////      if(char(cvWaitKey(15))==27)break;
////    }

//    ////detect contous neiboughbour
//    vector<vector<Point> >::iterator itc = contours.begin();
//    vector<vector<Point> >::iterator itc_next = contours.begin();
//    int contours_num_a = 0;
//    int contours_num_b = 0;
//    while (itc != contours.end())
//    {
//        contours_num_a++;
//        itc_next = itc;
//        itc_next++;
//        Point p_a;
//        Point p_b;
//        float min_distance = 200;
//        float threshold_distance;
//        while(itc_next != contours.end())
//        {
//            vector<Point> contoursA = *itc;
//            vector<Point> contoursB = *itc_next;
//            min_distance = findShortestDistance(contoursA,contoursB,p_a,p_b);
//            if(min_distance!=0)
//            {
//                threshold_distance = connect_dis*(1+2*pow(abs((float)(p_b.y-p_a.y))/min_distance,2));
//            }
//            if(min_distance < threshold_distance)
//            {
//                line(thres_window, p_a, p_b, Scalar(255, 0, 0), 3);
//            }
//            itc_next++;
//            contours_num_b++;
//        }
//      ++itc;
//    }
//#ifdef DEBUG
//    while(1)
//    {
//      imshow("thres_window", thres_window);
//      if(char(cvWaitKey(15))==27)break;
//    }
//#endif

//    ////find contours again
//    Mat img_contours_again;
//    thres_window.copyTo(img_contours_again);
//    vector<vector<Point> > contours_again;
//    findContours(img_contours_again,
//                 contours_again,               // a vector of contours
//                 CV_RETR_EXTERNAL,       // retrieve the external contours
//                 CV_CHAIN_APPROX_NONE);  // all pixels of each contours
////    removeIsoContour(contours_again);    //remove isolate contours again
//    Mat sepertate_im_again(thres_window.size(),thres_window.depth(),Scalar(255));
//    drawContours(sepertate_im_again,contours_again,-1,Scalar(0),2);
//    while(1)
//    {
//      imshow("sepertate_im_again",sepertate_im_again);
//      if(char(cvWaitKey(15))==27)break;
//    }

//    vector<vector<Point> >::iterator itc_again = contours_again.begin();
//    vector<Rect> vecRect;
//    while (itc_again != contours_again.end())
//    {
//      Rect mr = boundingRect(Mat(*itc_again));
//      Mat auxRoi(thres_window, mr);
//      if (/*verifyCharSizes(auxRoi)*/1) vecRect.push_back(mr);
//      ++itc_again;
//    }


//    ////save single char image after segment
//    for(int char_num=0;char_num<vecRect.size();char_num++)
//    {
//         Mat single_char_=thres_window(vecRect.at(char_num));
//         Mat single_char;
//         single_char = preprocessChar(single_char_);
//         single_char_vec.push_back(single_char);
//        if(save)
//        {
//            const char* single_char_folder_ = "../../../src/easyocr/char_img";
//            std::stringstream ss(std::stringstream::in | std::stringstream::out);
//            ss << single_char_folder_ << "/" << im_num << "_src" << char_num<<"_"<<rand()<< ".jpg";
//            imwrite(ss.str(),single_char);
//        }
//#ifdef DEBUG
//        while(1)
//        {
//          imshow( "single_char", single_char_ );
//          if(char(cvWaitKey(15))==27)break;
//        }
//#endif
//    }

//#ifdef DEBUG
//    thres_window.release();
//    cvDestroyWindow("thres_window");
//    cvDestroyWindow("sepertate_im_again");
//    cvDestroyWindow("single_char");
//#endif
//}

//void TextDetector::segmentSobMor(cv::Mat &spineGray, vector<Mat> &single_char_vec, vector<Rect> &vecRect, int im_num, bool save)
//{
//    int char_size;
//#ifdef DEBUG
//    printf("please input char size: big is 1, mediate is 2, small is 3\n");
//    scanf("%d",&char_size);
//#endif
//    setMorParameters(char_size);

//    ////histequal and shrpen
//    Mat spineGrayTemp = spineGray - 0.5;
//    cv::Mat spineAhe;
//    adaptiveHistEqual(spineGrayTemp, spineAhe, 0.01);
//    cv::Mat spineShrpen;
//    sharpenImage(spineAhe, spineShrpen);
//    while(1)
//    {
//      imshow("sharpen", spineShrpen);
//      if(char(cvWaitKey(15))==27)break;
//    }


//    ////soble
//    cv::Mat src_sobel;
//    int m_GaussianBlurSize = 5;
//    sobelOper(spineShrpen, src_sobel, m_GaussianBlurSize);
//    while(1)
//    {
//      imshow("src_sobel", src_sobel);
//      if(char(cvWaitKey(15))==27)break;
//    }

//    ////threshold
////    double minVal,maxVal;
////    Point minLoc,maxLoc;
////    minMaxLoc(src_sobel,&minVal,&maxVal,&minLoc,&maxLoc);
////    double threshold_value = maxVal*0.7;
//    double threshold_value = 150;
////    std::cout<<" threshold_value "<<maxVal<<std::endl;
//    cv::Mat window_tmp;
//    threshold(src_sobel, window_tmp, 0, 255, THRESH_OTSU+ CV_THRESH_BINARY);
////    threshold(src_sobel, window_tmp, threshold_value, 255, CV_THRESH_BINARY);
////    adaptiveThreshold(spineShrpen, window_tmp,255,CV_ADAPTIVE_THRESH_GAUSSIAN_C,CV_THRESH_BINARY, 5, 0);
////    std::cout<<window_tmp<<std::endl;
//    while(1)
//    {
//      imshow("window_thresh", window_tmp);
//      if(char(cvWaitKey(15))==27)break;
//    }
//    ////进行open操作
//    Mat element_sob = getStructuringElement(MORPH_RECT, Size(2, 2));
//    Mat open_sob;
//    morphologyEx(window_tmp,open_sob,MORPH_OPEN,element_sob);
//    while(1)
//    {
//      imshow("open_sob", open_sob);
//      if(char(cvWaitKey(15))==27)break;
//    }
//    cv::Mat thres_window = open_sob.clone();


//    ////find contours
//    Mat img_contours;
//    thres_window.copyTo(img_contours);
//    vector<vector<Point> > contours;
//    findContours(img_contours,
//                 contours,               // a vector of contours
//                 CV_RETR_EXTERNAL,       // retrieve the external contours
//                 CV_CHAIN_APPROX_NONE);  // all pixels of each contours
//    Mat sepertate_im(thres_window.size(),thres_window.depth(),Scalar(255));
//    drawContours(sepertate_im,contours,-1,Scalar(0),2);
//    while(1)
//    {
//      imshow("sepertate_im",sepertate_im);
//      if(char(cvWaitKey(15))==27)break;
//    }


//    ////remove isolate contours
//    Mat sepertate_im_remove(thres_window.size(),thres_window.depth(),Scalar(255));
//    //移除过长或过短的轮廓
//    int cmin = 1; //最小轮廓长度
//    int cmax = 600;    //最大轮廓
//    vector<vector<Point> >::iterator itc_mm = contours.begin();
//    while (itc_mm!=contours.end())
//    {
//        if (itc_mm->size() < cmin || itc_mm->size() > cmax)
//        {
//           itc_mm = contours.erase(itc_mm);
//        }
//        else
//            ++itc_mm;
//    }
////    removeIsoContour(contours);
//    drawContours(sepertate_im_remove,contours,-1,Scalar(0),2);
//    while(1)
//    {
//      imshow("sepertate_im_remove",sepertate_im_remove);
//      if(char(cvWaitKey(15))==27)break;
//    }

//    ////detect contous neiboughbour
//    vector<vector<Point> >::iterator itc = contours.begin();
//    vector<vector<Point> >::iterator itc_next = contours.begin();
//    int contours_num_a = 0;
//    int contours_num_b = 0;
//    while (itc != contours.end())
//    {
//        contours_num_a++;
//        itc_next = itc;
//        itc_next++;
//        Point p_a;
//        Point p_b;
//        float min_distance = 200;
//        float threshold_distance;
//        while(itc_next != contours.end())
//        {
//            vector<Point> contoursA = *itc;
//            vector<Point> contoursB = *itc_next;
//            min_distance = findShortestDistance(contoursA,contoursB,p_a,p_b);
//            if(min_distance!=0)
//            {
//                threshold_distance = connect_dis*(1+1.2*pow(abs((float)(p_b.y-p_a.y))/min_distance,2));
//            }
//            if(min_distance < threshold_distance)
//            {
//                line(thres_window, p_a, p_b, Scalar(255, 0, 0), 3);
//            }
//            itc_next++;
//            contours_num_b++;
//        }
//      ++itc;
//    }
//#ifdef DEBUG
//    while(1)
//    {
//      imshow("thres_window", thres_window);
//      if(char(cvWaitKey(15))==27)break;
//    }
//#endif

//    ////find contours again
//    Mat img_contours_again;
//    thres_window.copyTo(img_contours_again);
//    vector<vector<Point> > contours_again;
//    findContours(img_contours_again,
//                 contours_again,               // a vector of contours
//                 CV_RETR_EXTERNAL,       // retrieve the external contours
//                 CV_CHAIN_APPROX_NONE);  // all pixels of each contours
////    removeIsoContour(contours_again);    //remove isolate contours again
//    Mat sepertate_im_again(thres_window.size(),thres_window.depth(),Scalar(255));
//    drawContours(sepertate_im_again,contours_again,-1,Scalar(0),2);
//    while(1)
//    {
//      imshow("sepertate_im_again",sepertate_im_again);
//      if(char(cvWaitKey(15))==27)break;
//    }

//    vector<vector<Point> >::iterator itc_again = contours_again.begin();
////    vector<Rect> vecRect;
//    while (itc_again != contours_again.end())
//    {
//      Rect mr = boundingRect(Mat(*itc_again));
//      Mat auxRoi(thres_window, mr);
//      if (/*verifyCharSizes(auxRoi)*/1) vecRect.push_back(mr);
//      ++itc_again;
//    }



////    ////save single char image after segment
////    for(int char_num=0;char_num<vecRect.size();char_num++)
////    {
////         Mat single_char=thres_window(vecRect.at(char_num));
////         single_char_vec.push_back(single_char);
////        if(save)
////        {
////            const char* single_char_folder_ = "../../../src/easyocr/char_img";
////            std::stringstream ss(std::stringstream::in | std::stringstream::out);
////            ss << single_char_folder_ << "/" << im_num << "_sob" << char_num << ".jpg";
////            imwrite(ss.str(),single_char);
////        }
////        while(1)
////        {
////          imshow( "single_char", single_char );
////          if(char(cvWaitKey(15))==27)break;
////        }
////    }


//    thres_window.release();
//    cvDestroyWindow("sharpen");
//    cvDestroyWindow("src_sobel");
//    cvDestroyWindow("open_sob");
//    cvDestroyWindow("thres_window");
//    cvDestroyWindow("erode_out");
//    cvDestroyWindow("sepertate_im");
//    cvDestroyWindow("sepertate_im_again");
//    cvDestroyWindow("single_char");
//    cvDestroyWindow("window_thresh");

//}

//void TextDetector::DrawBox(CvBox2D box,IplImage* img)
//{
//     CvPoint2D32f point[4];
//     int i;
//     for ( i=0; i<4; i++)
//     {
//         point[i].x = 0;
//         point[i].y = 0;
//    }
//     cvBoxPoints(box, point); //计算二维盒子顶点
//    CvPoint pt[4];
//    for ( i=0; i<4; i++)
//    {
//        pt[i].x = (int)point[i].x;
//        pt[i].y = (int)point[i].y;
//    }
//    cvLine( img, pt[0], pt[1],CV_RGB(255,0,0), 2, 8, 0 );
//    cvLine( img, pt[1], pt[2],CV_RGB(255,0,0), 2, 8, 0 );
//    cvLine( img, pt[2], pt[3],CV_RGB(255,0,0), 2, 8, 0 );
//    cvLine( img, pt[3], pt[0],CV_RGB(255,0,0), 2, 8, 0 );
//}


////void TextDetector::drawDetectLines(Mat& image,const vector<Vec4i>& lines,Scalar  color)
////{
////    // 将检测到的直线在图上画出来
////    vector<Vec4i>::const_iterator it=lines.begin();
////    while(it!=lines.end())
////    {
////        Point pt1((*it)[0],(*it)[1]);
////        Point pt2((*it)[2],(*it)[3]);
////        line(image,pt1,pt2,color,2); //  线条宽度设置为2
////        ++it;
////    }
////}


// rotate an image
void TextDetector::rotateImg_l(Mat source, Mat& img_rotate,float angle){
  Point2f src_center(source.cols / 2.0F, source.rows / 2.0F);
  Mat rot_mat = getRotationMatrix2D(src_center, angle, 1.0);
  cv::Rect bbox = cv::RotatedRect(src_center, source.size(), angle).boundingRect();
  rot_mat.at<double>(0, 2) += bbox.width / 2.0 - src_center.x;
  rot_mat.at<double>(1, 2) += bbox.height / 2.0 - src_center.y;
  warpAffine(source, img_rotate, rot_mat, bbox.size(), CV_INTER_LINEAR, 0);
}

void TextDetector::segmentAnalyse_l(vector<char_range_t> &peek_range_v,vector<char_range_t> &char_rang)
{
    int peek_num = peek_range_v.size();
    printf("peek_num = %d\n",peek_num);

    switch (peek_num) {
    case 5:
        {
            int min_gap = 200;
            int min_gap_num = 0;
            for(int i =1;i<peek_range_v.size();i++)
            {
                int peek_rang_gap = peek_range_v.at(i).begin - peek_range_v.at(i-1).end;
//                printf("peek_gap = %d\n",peek_rang_gap);
                if(peek_rang_gap<min_gap)
                {
                    min_gap = peek_rang_gap;
                    min_gap_num = i-1;
                }
            }
//            printf("min_gap = %d, min_gap_num = %d\n",min_gap,min_gap_num);


            for(int i=0;i<peek_range_v.size();i++)
            {
                if(i==min_gap_num)
                {
                    char_range_t connect_peek;
                    connect_peek.begin = peek_range_v.at(i).begin;
                    connect_peek.end = peek_range_v.at(i+1).end;
                    char_rang.push_back(connect_peek);
                    i++;
                }
                else
                {
                    char_rang.push_back(peek_range_v.at(i));
                }
            }
//            printf("char_rang.size = %d\n",char_rang.size());
        }
        break;
    case 6:
        {
            int min_gap_1 = 200;
            int min_gap_2 = 200;
            int min_gap1_num = 0;
            int min_gap2_num = 0;

            //find the shortest
            for(int i =1;i<peek_range_v.size();i++)
            {
                int peek_rang_gap = peek_range_v.at(i).begin - peek_range_v.at(i-1).end;
                if(peek_rang_gap<min_gap_1)
                {
                    min_gap_1 = peek_rang_gap;
                    min_gap1_num = i-1;
                }
            }

            //find the second shorter
            for(int i =1;i<peek_range_v.size();i++)
            {
                if(i!=min_gap1_num+1)
                {
                    int peek_rang_gap = peek_range_v.at(i).begin - peek_range_v.at(i-1).end;
                    if(peek_rang_gap<min_gap_2)
                    {
                        min_gap_2 = peek_rang_gap;
                        min_gap2_num = i-1;
                    }
                }
            }

            for(int i=0;i<peek_range_v.size();i++)
            {
                if(((i==min_gap1_num)&&(i+1==min_gap2_num))||((i==min_gap2_num)&&(i+1==min_gap1_num)))
                {
                    char_range_t connect_peek;
                    connect_peek.begin = peek_range_v.at(i).begin;
                    connect_peek.end = peek_range_v.at(i+2).end;
                    char_rang.push_back(connect_peek);
                    i++;
                    i++;
                }
                else if((i==min_gap1_num)||(i==min_gap2_num))
                {
                    char_range_t connect_peek;
                    connect_peek.begin = peek_range_v.at(i).begin;
                    connect_peek.end = peek_range_v.at(i+1).end;
                    char_rang.push_back(connect_peek);
                    i++;
                }
                else
                {
                    char_rang.push_back(peek_range_v.at(i));
                }
            }
        }
        break;
    case 7:
        {
            int min_gap_1 = 200;
            int min_gap_2 = 200;
            int min_gap_3 = 200;
            int min_gap1_num = 0;
            int min_gap2_num = 0;
            int min_gap3_num = 0;
            //find the shortest
            for(int i =1;i<peek_range_v.size();i++)
            {
                int peek_rang_gap = peek_range_v.at(i).begin - peek_range_v.at(i-1).end;
                if(peek_rang_gap<min_gap_1)
                {
                    min_gap_1 = peek_rang_gap;
                    min_gap1_num = i-1;
                }
            }

            //find the second shorter
            for(int i =1;i<peek_range_v.size();i++)
            {
                if(i!=min_gap1_num+1)
                {
                    int peek_rang_gap = peek_range_v.at(i).begin - peek_range_v.at(i-1).end;
                    if(peek_rang_gap<min_gap_2)
                    {
                        min_gap_2 = peek_rang_gap;
                        min_gap2_num = i-1;
                    }
                }
            }

            //find the third shorter
            for(int i =1;i<peek_range_v.size();i++)
            {
                if((i!=min_gap1_num+1)&&(i!=min_gap2_num+1))
                {
                    int peek_rang_gap = peek_range_v.at(i).begin - peek_range_v.at(i-1).end;
                    if(peek_rang_gap<min_gap_3)
                    {
                        min_gap_3 = peek_rang_gap;
                        min_gap3_num = i-1;
                    }
                }
            }



            for(int i=0;i<peek_range_v.size();i++)
            {
                if((((i==min_gap1_num)&&(i+1==min_gap2_num))||((i==min_gap2_num)&&(i+1==min_gap1_num)))
                 ||(((i==min_gap1_num)&&(i+1==min_gap3_num))||((i==min_gap3_num)&&(i+1==min_gap1_num)))
                 ||(((i==min_gap2_num)&&(i+1==min_gap3_num))||((i==min_gap3_num)&&(i+1==min_gap2_num))))
                {
                    char_range_t connect_peek;
                    connect_peek.begin = peek_range_v.at(i).begin;
                    connect_peek.end = peek_range_v.at(i+2).end;
                    char_rang.push_back(connect_peek);
                    i++;
                    i++;
                }
                else if((i==min_gap1_num)||(i==min_gap2_num)||(i==min_gap3_num))
                {
                    char_range_t connect_peek;
                    connect_peek.begin = peek_range_v.at(i).begin;
                    connect_peek.end = peek_range_v.at(i+1).end;
                    char_rang.push_back(connect_peek);
                    i++;
                }
                else
                {
                    char_rang.push_back(peek_range_v.at(i));
                }
            }
        }
        break;
    case 8:
        {
            int min_gap_1 = 200;
            int min_gap_2 = 200;
            int min_gap_3 = 200;
            int min_gap_4 = 200;
            int min_gap1_num = 0;
            int min_gap2_num = 0;
            int min_gap3_num = 0;
            int min_gap4_num = 0;

            //find the shortest
            for(int i =1;i<peek_range_v.size();i++)
            {
                int peek_rang_gap = peek_range_v.at(i).begin - peek_range_v.at(i-1).end;
                if(peek_rang_gap<min_gap_1)
                {
                    min_gap_1 = peek_rang_gap;
                    min_gap1_num = i-1;
                }
            }

            //find the second shorter
            for(int i =1;i<peek_range_v.size();i++)
            {
                if(i!=min_gap1_num+1)
                {
                    int peek_rang_gap = peek_range_v.at(i).begin - peek_range_v.at(i-1).end;
                    if(peek_rang_gap<min_gap_2)
                    {
                        min_gap_2 = peek_rang_gap;
                        min_gap2_num = i-1;
                    }
                }
            }

            //find the third shorter
            for(int i =1;i<peek_range_v.size();i++)
            {
                if((i!=min_gap1_num+1)&&(i!=min_gap2_num+1))
                {
                    int peek_rang_gap = peek_range_v.at(i).begin - peek_range_v.at(i-1).end;
                    if(peek_rang_gap<min_gap_3)
                    {
                        min_gap_3 = peek_rang_gap;
                        min_gap3_num = i-1;
                    }
                }
            }

            //find the forth shorter
            for(int i =1;i<peek_range_v.size();i++)
            {
                if((i!=min_gap1_num+1)&&(i!=min_gap2_num+1)&&(i!=min_gap3_num+1))
                {
                    int peek_rang_gap = peek_range_v.at(i).begin - peek_range_v.at(i-1).end;
                    if(peek_rang_gap<min_gap_4)
                    {
                        min_gap_4 = peek_rang_gap;
                        min_gap4_num = i-1;
                    }
                }
            }


            for(int i=0;i<peek_range_v.size();i++)
            {
                if((((i==min_gap1_num)&&(i+1==min_gap2_num))||((i==min_gap2_num)&&(i+1==min_gap1_num)))
                 ||(((i==min_gap1_num)&&(i+1==min_gap3_num))||((i==min_gap3_num)&&(i+1==min_gap1_num)))
                 ||(((i==min_gap1_num)&&(i+1==min_gap4_num))||((i==min_gap4_num)&&(i+1==min_gap1_num)))
                 ||(((i==min_gap2_num)&&(i+1==min_gap3_num))||((i==min_gap3_num)&&(i+1==min_gap2_num)))
                 ||(((i==min_gap2_num)&&(i+1==min_gap4_num))||((i==min_gap4_num)&&(i+1==min_gap2_num)))
                 ||(((i==min_gap3_num)&&(i+1==min_gap4_num))||((i==min_gap4_num)&&(i+1==min_gap3_num))))
                {
                    char_range_t connect_peek;
                    connect_peek.begin = peek_range_v.at(i).begin;
                    connect_peek.end = peek_range_v.at(i+2).end;
                    char_rang.push_back(connect_peek);
                    i++;
                    i++;
                }
                else if((i==min_gap1_num)||(i==min_gap2_num)||(i==min_gap3_num)||(i==min_gap4_num))
                {
                    char_range_t connect_peek;
                    connect_peek.begin = peek_range_v.at(i).begin;
                    connect_peek.end = peek_range_v.at(i+1).end;
                    char_rang.push_back(connect_peek);
                    i++;
                }
                else
                {
                    char_rang.push_back(peek_range_v.at(i));
                }
            }
        }
        break;
    case 9:
        {
            int min_gap_1 = 200;
            int min_gap_2 = 200;
            int min_gap_3 = 200;
            int min_gap_4 = 200;
            int min_gap_5 = 200;
            int min_gap1_num = 0;
            int min_gap2_num = 0;
            int min_gap3_num = 0;
            int min_gap4_num = 0;
            int min_gap5_num = 0;
//            printf("peek_gap = %d\n",min_gap_1);
            //find the shortest
            for(int i =1;i<peek_range_v.size();i++)
            {
                int peek_rang_gap = peek_range_v.at(i).begin - peek_range_v.at(i-1).end;
//                printf("peek_gap = %d\n",peek_rang_gap);
                if(peek_rang_gap<min_gap_1)
                {
                    min_gap_1 = peek_rang_gap;
                    min_gap1_num = i-1;
                }
            }

            //find the second shorter
            for(int i =1;i<peek_range_v.size();i++)
            {
                if(i!=min_gap1_num+1)
                {
                    int peek_rang_gap = peek_range_v.at(i).begin - peek_range_v.at(i-1).end;
                    if(peek_rang_gap<min_gap_2)
                    {
                        min_gap_2 = peek_rang_gap;
                        min_gap2_num = i-1;
                    }
                }
            }

            //find the third shorter
            for(int i =1;i<peek_range_v.size();i++)
            {
                if((i!=min_gap1_num+1)&&(i!=min_gap2_num+1))
                {
                    int peek_rang_gap = peek_range_v.at(i).begin - peek_range_v.at(i-1).end;
                    if(peek_rang_gap<min_gap_3)
                    {
                        min_gap_3 = peek_rang_gap;
                        min_gap3_num = i-1;
                    }
                }
            }

            //find the forth shorter
            for(int i =1;i<peek_range_v.size();i++)
            {
                if((i!=min_gap1_num+1)&&(i!=min_gap2_num+1)&&(i!=min_gap3_num+1))
                {
                    int peek_rang_gap = peek_range_v.at(i).begin - peek_range_v.at(i-1).end;
                    if(peek_rang_gap<min_gap_4)
                    {
                        min_gap_4 = peek_rang_gap;
                        min_gap4_num = i-1;
                    }
                }
            }

            //find the fifth shorter
            for(int i =1;i<peek_range_v.size();i++)
            {
                if((i!=min_gap1_num+1)&&(i!=min_gap2_num+1)&&(i!=min_gap3_num+1)&&(i!=min_gap4_num+1))
                {
                    int peek_rang_gap = peek_range_v.at(i).begin - peek_range_v.at(i-1).end;
                    if(peek_rang_gap<min_gap_5)
                    {
                        min_gap_5 = peek_rang_gap;
                        min_gap5_num = i-1;
                    }
                }
            }

            printf("min_gap_1 = %d, min_gap1_num = %d\n",min_gap_1,min_gap1_num);
            printf("min_gap_2 = %d, min_gap2_num = %d\n",min_gap_2,min_gap2_num);
            printf("min_gap_3 = %d, min_gap3_num = %d\n",min_gap_3,min_gap3_num);
            printf("min_gap_4 = %d, min_gap4_num = %d\n",min_gap_4,min_gap4_num);
            printf("min_gap_5 = %d, min_gap5_num = %d\n",min_gap_5,min_gap5_num);

            for(int i=0;i<peek_range_v.size();i++)
            {
                if((((i==min_gap1_num)&&(i+1==min_gap2_num))||((i==min_gap2_num)&&(i+1==min_gap1_num)))
                 ||(((i==min_gap1_num)&&(i+1==min_gap3_num))||((i==min_gap3_num)&&(i+1==min_gap1_num)))
                 ||(((i==min_gap1_num)&&(i+1==min_gap4_num))||((i==min_gap4_num)&&(i+1==min_gap1_num)))
                 ||(((i==min_gap1_num)&&(i+1==min_gap5_num))||((i==min_gap5_num)&&(i+1==min_gap1_num)))
                 ||(((i==min_gap2_num)&&(i+1==min_gap3_num))||((i==min_gap3_num)&&(i+1==min_gap2_num)))
                 ||(((i==min_gap2_num)&&(i+1==min_gap4_num))||((i==min_gap4_num)&&(i+1==min_gap2_num)))
                 ||(((i==min_gap2_num)&&(i+1==min_gap5_num))||((i==min_gap5_num)&&(i+1==min_gap2_num)))
                 ||(((i==min_gap3_num)&&(i+1==min_gap4_num))||((i==min_gap4_num)&&(i+1==min_gap3_num)))
                 ||(((i==min_gap3_num)&&(i+1==min_gap5_num))||((i==min_gap5_num)&&(i+1==min_gap3_num)))
                 ||(((i==min_gap4_num)&&(i+1==min_gap5_num))||((i==min_gap5_num)&&(i+1==min_gap4_num))))
                {
                    char_range_t connect_peek;
                    connect_peek.begin = peek_range_v.at(i).begin;
                    connect_peek.end = peek_range_v.at(i+2).end;
                    char_rang.push_back(connect_peek);
                    i++;
                    i++;
                }
                else if((i==min_gap1_num)||(i==min_gap2_num)||(i==min_gap3_num)||(i==min_gap4_num)
                      ||(i==min_gap5_num))
                {
                    char_range_t connect_peek;
                    connect_peek.begin = peek_range_v.at(i).begin;
                    connect_peek.end = peek_range_v.at(i+1).end;
                    char_rang.push_back(connect_peek);
                    i++;
                }
                else
                {
                    char_rang.push_back(peek_range_v.at(i));
                }
            }
        }
        break;
    case 10:
        {
            int min_gap_1 = 200;
            int min_gap_2 = 200;
            int min_gap_3 = 200;
            int min_gap_4 = 200;
            int min_gap_5 = 200;
            int min_gap_6 = 200;
            int min_gap1_num = 0;
            int min_gap2_num = 0;
            int min_gap3_num = 0;
            int min_gap4_num = 0;
            int min_gap5_num = 0;
            int min_gap6_num = 0;

            //find the shortest
            for(int i =1;i<peek_range_v.size();i++)
            {
                int peek_rang_gap = peek_range_v.at(i).begin - peek_range_v.at(i-1).end;
//                printf("peek_gap = %d\n",peek_rang_gap);
                if(peek_rang_gap<min_gap_1)
                {
                    min_gap_1 = peek_rang_gap;
                    min_gap1_num = i-1;
                }
            }

            //find the second shorter
            for(int i =1;i<peek_range_v.size();i++)
            {
                if(i!=min_gap1_num+1)
                {
                    int peek_rang_gap = peek_range_v.at(i).begin - peek_range_v.at(i-1).end;
                    if(peek_rang_gap<min_gap_2)
                    {
                        min_gap_2 = peek_rang_gap;
                        min_gap2_num = i-1;
                    }
                }
            }

            //find the third shorter
            for(int i =1;i<peek_range_v.size();i++)
            {
                if((i!=min_gap1_num+1)&&(i!=min_gap2_num+1))
                {
                    int peek_rang_gap = peek_range_v.at(i).begin - peek_range_v.at(i-1).end;
                    if(peek_rang_gap<min_gap_3)
                    {
                        min_gap_3 = peek_rang_gap;
                        min_gap3_num = i-1;
                    }
                }
            }

            //find the forth shorter
            for(int i =1;i<peek_range_v.size();i++)
            {
                if((i!=min_gap1_num+1)&&(i!=min_gap2_num+1)&&(i!=min_gap3_num+1))
                {
                    int peek_rang_gap = peek_range_v.at(i).begin - peek_range_v.at(i-1).end;
                    if(peek_rang_gap<min_gap_4)
                    {
                        min_gap_4 = peek_rang_gap;
                        min_gap4_num = i-1;
                    }
                }
            }

            //find the fifth shorter
            for(int i =1;i<peek_range_v.size();i++)
            {
                if((i!=min_gap1_num+1)&&(i!=min_gap2_num+1)&&(i!=min_gap3_num+1)&&(i!=min_gap4_num+1))
                {
                    int peek_rang_gap = peek_range_v.at(i).begin - peek_range_v.at(i-1).end;
                    if(peek_rang_gap<min_gap_5)
                    {
                        min_gap_5 = peek_rang_gap;
                        min_gap5_num = i-1;
                    }
                }
            }

            //find the sixth shorter
            for(int i =1;i<peek_range_v.size();i++)
            {
                if((i!=min_gap1_num+1)&&(i!=min_gap2_num+1)&&(i!=min_gap3_num+1)
                 &&(i!=min_gap4_num+1)&&(i!=min_gap5_num+1))
                {
                    int peek_rang_gap = peek_range_v.at(i).begin - peek_range_v.at(i-1).end;
                    if(peek_rang_gap<min_gap_6)
                    {
                        min_gap_6 = peek_rang_gap;
                        min_gap6_num = i-1;
                    }
                }
            }


            for(int i=0;i<peek_range_v.size();i++)
            {
                if((((i==min_gap1_num)&&(i+1==min_gap2_num))||((i==min_gap2_num)&&(i+1==min_gap1_num)))
                 ||(((i==min_gap1_num)&&(i+1==min_gap3_num))||((i==min_gap3_num)&&(i+1==min_gap1_num)))
                 ||(((i==min_gap1_num)&&(i+1==min_gap4_num))||((i==min_gap4_num)&&(i+1==min_gap1_num)))
                 ||(((i==min_gap1_num)&&(i+1==min_gap5_num))||((i==min_gap5_num)&&(i+1==min_gap1_num)))
                 ||(((i==min_gap1_num)&&(i+1==min_gap6_num))||((i==min_gap6_num)&&(i+1==min_gap1_num)))
                 ||(((i==min_gap2_num)&&(i+1==min_gap3_num))||((i==min_gap3_num)&&(i+1==min_gap2_num)))
                 ||(((i==min_gap2_num)&&(i+1==min_gap4_num))||((i==min_gap4_num)&&(i+1==min_gap2_num)))
                 ||(((i==min_gap2_num)&&(i+1==min_gap5_num))||((i==min_gap5_num)&&(i+1==min_gap2_num)))
                 ||(((i==min_gap2_num)&&(i+1==min_gap6_num))||((i==min_gap6_num)&&(i+1==min_gap2_num)))
                 ||(((i==min_gap3_num)&&(i+1==min_gap4_num))||((i==min_gap4_num)&&(i+1==min_gap3_num)))
                 ||(((i==min_gap3_num)&&(i+1==min_gap5_num))||((i==min_gap5_num)&&(i+1==min_gap3_num)))
                 ||(((i==min_gap3_num)&&(i+1==min_gap6_num))||((i==min_gap6_num)&&(i+1==min_gap3_num)))
                 ||(((i==min_gap4_num)&&(i+1==min_gap5_num))||((i==min_gap5_num)&&(i+1==min_gap4_num)))
                 ||(((i==min_gap4_num)&&(i+1==min_gap6_num))||((i==min_gap6_num)&&(i+1==min_gap4_num)))
                 ||(((i==min_gap5_num)&&(i+1==min_gap6_num))||((i==min_gap6_num)&&(i+1==min_gap5_num))))
                {
                    char_range_t connect_peek;
                    connect_peek.begin = peek_range_v.at(i).begin;
                    connect_peek.end = peek_range_v.at(i+2).end;
                    char_rang.push_back(connect_peek);
                    i++;
                    i++;
                }
                else if((i==min_gap1_num)||(i==min_gap2_num)||(i==min_gap3_num)||(i==min_gap4_num)
                      ||(i==min_gap5_num)||(i==min_gap6_num))
                {
                    char_range_t connect_peek;
                    connect_peek.begin = peek_range_v.at(i).begin;
                    connect_peek.end = peek_range_v.at(i+1).end;
                    char_rang.push_back(connect_peek);
                    i++;
                }
                else
                {
                    char_rang.push_back(peek_range_v.at(i));
                }
            }
        }
        break;
    case 11:
        {
            int min_gap_1 = 200;
            int min_gap_2 = 200;
            int min_gap_3 = 200;
            int min_gap_4 = 200;
            int min_gap_5 = 200;
            int min_gap_6 = 200;
            int min_gap_7 = 200;
            int min_gap1_num = 0;
            int min_gap2_num = 0;
            int min_gap3_num = 0;
            int min_gap4_num = 0;
            int min_gap5_num = 0;
            int min_gap6_num = 0;
            int min_gap7_num = 0;

            //find the shortest
            for(int i =1;i<peek_range_v.size();i++)
            {
                int peek_rang_gap = peek_range_v.at(i).begin - peek_range_v.at(i-1).end;
//                printf("peek_gap = %d\n",peek_rang_gap);
                if(peek_rang_gap<min_gap_1)
                {
                    min_gap_1 = peek_rang_gap;
                    min_gap1_num = i-1;
                }
            }

            //find the second shorter
            for(int i =1;i<peek_range_v.size();i++)
            {
                if(i!=min_gap1_num+1)
                {
                    int peek_rang_gap = peek_range_v.at(i).begin - peek_range_v.at(i-1).end;
                    if(peek_rang_gap<min_gap_2)
                    {
                        min_gap_2 = peek_rang_gap;
                        min_gap2_num = i-1;
                    }
                }
            }

            //find the third shorter
            for(int i =1;i<peek_range_v.size();i++)
            {
                if((i!=min_gap1_num+1)&&(i!=min_gap2_num+1))
                {
                    int peek_rang_gap = peek_range_v.at(i).begin - peek_range_v.at(i-1).end;
                    if(peek_rang_gap<min_gap_3)
                    {
                        min_gap_3 = peek_rang_gap;
                        min_gap3_num = i-1;
                    }
                }
            }

            //find the forth shorter
            for(int i =1;i<peek_range_v.size();i++)
            {
                if((i!=min_gap1_num+1)&&(i!=min_gap2_num+1)&&(i!=min_gap3_num+1))
                {
                    int peek_rang_gap = peek_range_v.at(i).begin - peek_range_v.at(i-1).end;
                    if(peek_rang_gap<min_gap_4)
                    {
                        min_gap_4 = peek_rang_gap;
                        min_gap4_num = i-1;
                    }
                }
            }

            //find the fifth shorter
            for(int i =1;i<peek_range_v.size();i++)
            {
                if((i!=min_gap1_num+1)&&(i!=min_gap2_num+1)&&(i!=min_gap3_num+1)&&(i!=min_gap4_num+1))
                {
                    int peek_rang_gap = peek_range_v.at(i).begin - peek_range_v.at(i-1).end;
                    if(peek_rang_gap<min_gap_5)
                    {
                        min_gap_5 = peek_rang_gap;
                        min_gap5_num = i-1;
                    }
                }
            }

            //find the sixth shorter
            for(int i =1;i<peek_range_v.size();i++)
            {
                if((i!=min_gap1_num+1)&&(i!=min_gap2_num+1)&&(i!=min_gap3_num+1)
                 &&(i!=min_gap4_num+1)&&(i!=min_gap5_num+1))
                {
                    int peek_rang_gap = peek_range_v.at(i).begin - peek_range_v.at(i-1).end;
                    if(peek_rang_gap<min_gap_6)
                    {
                        min_gap_6 = peek_rang_gap;
                        min_gap6_num = i-1;
                    }
                }
            }

            //find the seventh shorter
            for(int i =1;i<peek_range_v.size();i++)
            {
                if((i!=min_gap1_num+1)&&(i!=min_gap2_num+1)&&(i!=min_gap3_num+1)
                 &&(i!=min_gap4_num+1)&&(i!=min_gap5_num+1)&&(i!=min_gap6_num+1))
                {
                    int peek_rang_gap = peek_range_v.at(i).begin - peek_range_v.at(i-1).end;
                    if(peek_rang_gap<min_gap_7)
                    {
                        min_gap_7 = peek_rang_gap;
                        min_gap7_num = i-1;
                    }
                }
            }


//            printf("min_gap_1 = %d, min_gap1_num = %d\n",min_gap_1,min_gap1_num);
//            printf("min_gap_2 = %d, min_gap2_num = %d\n",min_gap_2,min_gap2_num);
//            printf("min_gap_3 = %d, min_gap3_num = %d\n",min_gap_3,min_gap3_num);
//            printf("min_gap_4 = %d, min_gap4_num = %d\n",min_gap_4,min_gap4_num);
//            printf("min_gap_5 = %d, min_gap5_num = %d\n",min_gap_5,min_gap5_num);
//            printf("min_gap_6 = %d, min_gap5_num = %d\n",min_gap_6,min_gap6_num);
//            printf("min_gap_7 = %d, min_gap5_num = %d\n",min_gap_7,min_gap7_num);


            for(int i=0;i<peek_range_v.size();i++)
            {
                if((((i==min_gap1_num)&&(i+1==min_gap2_num))||((i==min_gap2_num)&&(i+1==min_gap1_num)))
                 ||(((i==min_gap1_num)&&(i+1==min_gap3_num))||((i==min_gap3_num)&&(i+1==min_gap1_num)))
                 ||(((i==min_gap1_num)&&(i+1==min_gap4_num))||((i==min_gap4_num)&&(i+1==min_gap1_num)))
                 ||(((i==min_gap1_num)&&(i+1==min_gap5_num))||((i==min_gap5_num)&&(i+1==min_gap1_num)))
                 ||(((i==min_gap1_num)&&(i+1==min_gap6_num))||((i==min_gap6_num)&&(i+1==min_gap1_num)))
                 ||(((i==min_gap1_num)&&(i+1==min_gap7_num))||((i==min_gap7_num)&&(i+1==min_gap1_num)))
                 ||(((i==min_gap2_num)&&(i+1==min_gap3_num))||((i==min_gap3_num)&&(i+1==min_gap2_num)))
                 ||(((i==min_gap2_num)&&(i+1==min_gap4_num))||((i==min_gap4_num)&&(i+1==min_gap2_num)))
                 ||(((i==min_gap2_num)&&(i+1==min_gap5_num))||((i==min_gap5_num)&&(i+1==min_gap2_num)))
                 ||(((i==min_gap2_num)&&(i+1==min_gap6_num))||((i==min_gap6_num)&&(i+1==min_gap2_num)))
                 ||(((i==min_gap2_num)&&(i+1==min_gap7_num))||((i==min_gap7_num)&&(i+1==min_gap2_num)))
                 ||(((i==min_gap3_num)&&(i+1==min_gap4_num))||((i==min_gap4_num)&&(i+1==min_gap3_num)))
                 ||(((i==min_gap3_num)&&(i+1==min_gap5_num))||((i==min_gap5_num)&&(i+1==min_gap3_num)))
                 ||(((i==min_gap3_num)&&(i+1==min_gap6_num))||((i==min_gap6_num)&&(i+1==min_gap3_num)))
                 ||(((i==min_gap3_num)&&(i+1==min_gap7_num))||((i==min_gap7_num)&&(i+1==min_gap3_num)))
                 ||(((i==min_gap4_num)&&(i+1==min_gap5_num))||((i==min_gap5_num)&&(i+1==min_gap4_num)))
                 ||(((i==min_gap4_num)&&(i+1==min_gap6_num))||((i==min_gap6_num)&&(i+1==min_gap4_num)))
                 ||(((i==min_gap4_num)&&(i+1==min_gap7_num))||((i==min_gap7_num)&&(i+1==min_gap4_num)))
                 ||(((i==min_gap5_num)&&(i+1==min_gap6_num))||((i==min_gap6_num)&&(i+1==min_gap5_num)))
                 ||(((i==min_gap5_num)&&(i+1==min_gap7_num))||((i==min_gap7_num)&&(i+1==min_gap5_num)))
                 ||(((i==min_gap6_num)&&(i+1==min_gap7_num))||((i==min_gap7_num)&&(i+1==min_gap6_num))))
                {
                    char_range_t connect_peek;
                    connect_peek.begin = peek_range_v.at(i).begin;
                    connect_peek.end = peek_range_v.at(i+2).end;
                    char_rang.push_back(connect_peek);
                    i++;
                    i++;
                }
                else if((i==min_gap1_num)||(i==min_gap2_num)||(i==min_gap3_num)||(i==min_gap4_num)
                      ||(i==min_gap5_num)||(i==min_gap6_num)||(i==min_gap7_num))
                {
                    char_range_t connect_peek;
                    connect_peek.begin = peek_range_v.at(i).begin;
                    connect_peek.end = peek_range_v.at(i+1).end;
                    char_rang.push_back(connect_peek);
                    i++;
                }
                else
                {
                    char_rang.push_back(peek_range_v.at(i));
                }
            }
//            printf("char_rang.size = %d\n",char_rang.size());
        }
        break;
    case 12:
        {
            int min_gap_1 = 200;
            int min_gap_2 = 200;
            int min_gap_3 = 200;
            int min_gap_4 = 200;
            int min_gap_5 = 200;
            int min_gap_6 = 200;
            int min_gap_7 = 200;
            int min_gap_8 = 200;
            int min_gap1_num = 0;
            int min_gap2_num = 0;
            int min_gap3_num = 0;
            int min_gap4_num = 0;
            int min_gap5_num = 0;
            int min_gap6_num = 0;
            int min_gap7_num = 0;
            int min_gap8_num = 0;


            //find the shortest
            for(int i =1;i<peek_range_v.size();i++)
            {
                int peek_rang_gap = peek_range_v.at(i).begin - peek_range_v.at(i-1).end;
//                printf("peek_gap = %d\n",peek_rang_gap);
                if(peek_rang_gap<min_gap_1)
                {
                    min_gap_1 = peek_rang_gap;
                    min_gap1_num = i-1;
                }
            }

            //find the second shorter
            for(int i =1;i<peek_range_v.size();i++)
            {
                if(i!=min_gap1_num+1)
                {
                    int peek_rang_gap = peek_range_v.at(i).begin - peek_range_v.at(i-1).end;
                    if(peek_rang_gap<min_gap_2)
                    {
                        min_gap_2 = peek_rang_gap;
                        min_gap2_num = i-1;
                    }
                }
            }

            //find the third shorter
            for(int i =1;i<peek_range_v.size();i++)
            {
                if((i!=min_gap1_num+1)&&(i!=min_gap2_num+1))
                {
                    int peek_rang_gap = peek_range_v.at(i).begin - peek_range_v.at(i-1).end;
                    if(peek_rang_gap<min_gap_3)
                    {
                        min_gap_3 = peek_rang_gap;
                        min_gap3_num = i-1;
                    }
                }
            }

            //find the forth shorter
            for(int i =1;i<peek_range_v.size();i++)
            {
                if((i!=min_gap1_num+1)&&(i!=min_gap2_num+1)&&(i!=min_gap3_num+1))
                {
                    int peek_rang_gap = peek_range_v.at(i).begin - peek_range_v.at(i-1).end;
                    if(peek_rang_gap<min_gap_4)
                    {
                        min_gap_4 = peek_rang_gap;
                        min_gap4_num = i-1;
                    }
                }
            }

            //find the fifth shorter
            for(int i =1;i<peek_range_v.size();i++)
            {
                if((i!=min_gap1_num+1)&&(i!=min_gap2_num+1)&&(i!=min_gap3_num+1)&&(i!=min_gap4_num+1))
                {
                    int peek_rang_gap = peek_range_v.at(i).begin - peek_range_v.at(i-1).end;
                    if(peek_rang_gap<min_gap_5)
                    {
                        min_gap_5 = peek_rang_gap;
                        min_gap5_num = i-1;
                    }
                }
            }

            //find the sixth shorter
            for(int i =1;i<peek_range_v.size();i++)
            {
                if((i!=min_gap1_num+1)&&(i!=min_gap2_num+1)&&(i!=min_gap3_num+1)
                 &&(i!=min_gap4_num+1)&&(i!=min_gap5_num+1))
                {
                    int peek_rang_gap = peek_range_v.at(i).begin - peek_range_v.at(i-1).end;
                    if(peek_rang_gap<min_gap_6)
                    {
                        min_gap_6 = peek_rang_gap;
                        min_gap6_num = i-1;
                    }
                }
            }

            //find the seventh shorter
            for(int i =1;i<peek_range_v.size();i++)
            {
                if((i!=min_gap1_num+1)&&(i!=min_gap2_num+1)&&(i!=min_gap3_num+1)
                 &&(i!=min_gap4_num+1)&&(i!=min_gap5_num+1)&&(i!=min_gap6_num+1))
                {
                    int peek_rang_gap = peek_range_v.at(i).begin - peek_range_v.at(i-1).end;
                    if(peek_rang_gap<min_gap_7)
                    {
                        min_gap_7 = peek_rang_gap;
                        min_gap7_num = i-1;
                    }
                }
            }

            //find the eighth shorter
            for(int i =1;i<peek_range_v.size();i++)
            {
                if((i!=min_gap1_num+1)&&(i!=min_gap2_num+1)&&(i!=min_gap3_num+1)
                 &&(i!=min_gap4_num+1)&&(i!=min_gap5_num+1)&&(i!=min_gap6_num+1)&&(i!=min_gap7_num+1))
                {
                    int peek_rang_gap = peek_range_v.at(i).begin - peek_range_v.at(i-1).end;
                    if(peek_rang_gap<min_gap_8)
                    {
                        min_gap_8 = peek_rang_gap;
                        min_gap8_num = i-1;
                    }
                }
            }

            for(int i=0;i<peek_range_v.size();i++)
            {
                if((((i==min_gap1_num)&&(i+1==min_gap2_num))||((i==min_gap2_num)&&(i+1==min_gap1_num)))
                 ||(((i==min_gap1_num)&&(i+1==min_gap3_num))||((i==min_gap3_num)&&(i+1==min_gap1_num)))
                 ||(((i==min_gap1_num)&&(i+1==min_gap4_num))||((i==min_gap4_num)&&(i+1==min_gap1_num)))
                 ||(((i==min_gap1_num)&&(i+1==min_gap5_num))||((i==min_gap5_num)&&(i+1==min_gap1_num)))
                 ||(((i==min_gap1_num)&&(i+1==min_gap6_num))||((i==min_gap6_num)&&(i+1==min_gap1_num)))
                 ||(((i==min_gap1_num)&&(i+1==min_gap7_num))||((i==min_gap7_num)&&(i+1==min_gap1_num)))
                 ||(((i==min_gap1_num)&&(i+1==min_gap8_num))||((i==min_gap8_num)&&(i+1==min_gap1_num)))
                 ||(((i==min_gap2_num)&&(i+1==min_gap3_num))||((i==min_gap3_num)&&(i+1==min_gap2_num)))
                 ||(((i==min_gap2_num)&&(i+1==min_gap4_num))||((i==min_gap4_num)&&(i+1==min_gap2_num)))
                 ||(((i==min_gap2_num)&&(i+1==min_gap5_num))||((i==min_gap5_num)&&(i+1==min_gap2_num)))
                 ||(((i==min_gap2_num)&&(i+1==min_gap6_num))||((i==min_gap6_num)&&(i+1==min_gap2_num)))
                 ||(((i==min_gap2_num)&&(i+1==min_gap7_num))||((i==min_gap7_num)&&(i+1==min_gap2_num)))
                 ||(((i==min_gap2_num)&&(i+1==min_gap8_num))||((i==min_gap8_num)&&(i+1==min_gap2_num)))
                 ||(((i==min_gap3_num)&&(i+1==min_gap4_num))||((i==min_gap4_num)&&(i+1==min_gap3_num)))
                 ||(((i==min_gap3_num)&&(i+1==min_gap5_num))||((i==min_gap5_num)&&(i+1==min_gap3_num)))
                 ||(((i==min_gap3_num)&&(i+1==min_gap6_num))||((i==min_gap6_num)&&(i+1==min_gap3_num)))
                 ||(((i==min_gap3_num)&&(i+1==min_gap7_num))||((i==min_gap7_num)&&(i+1==min_gap3_num)))
                 ||(((i==min_gap3_num)&&(i+1==min_gap8_num))||((i==min_gap8_num)&&(i+1==min_gap3_num)))
                 ||(((i==min_gap4_num)&&(i+1==min_gap5_num))||((i==min_gap5_num)&&(i+1==min_gap4_num)))
                 ||(((i==min_gap4_num)&&(i+1==min_gap6_num))||((i==min_gap6_num)&&(i+1==min_gap4_num)))
                 ||(((i==min_gap4_num)&&(i+1==min_gap7_num))||((i==min_gap7_num)&&(i+1==min_gap4_num)))
                 ||(((i==min_gap4_num)&&(i+1==min_gap8_num))||((i==min_gap8_num)&&(i+1==min_gap4_num)))
                 ||(((i==min_gap5_num)&&(i+1==min_gap6_num))||((i==min_gap6_num)&&(i+1==min_gap5_num)))
                 ||(((i==min_gap5_num)&&(i+1==min_gap7_num))||((i==min_gap7_num)&&(i+1==min_gap5_num)))
                 ||(((i==min_gap5_num)&&(i+1==min_gap8_num))||((i==min_gap8_num)&&(i+1==min_gap5_num)))
                 ||(((i==min_gap6_num)&&(i+1==min_gap7_num))||((i==min_gap7_num)&&(i+1==min_gap6_num)))
                 ||(((i==min_gap6_num)&&(i+1==min_gap8_num))||((i==min_gap8_num)&&(i+1==min_gap6_num)))
                 ||(((i==min_gap7_num)&&(i+1==min_gap8_num))||((i==min_gap8_num)&&(i+1==min_gap7_num))))
                {
                    char_range_t connect_peek;
                    connect_peek.begin = peek_range_v.at(i).begin;
                    connect_peek.end = peek_range_v.at(i+2).end;
                    char_rang.push_back(connect_peek);
                    i++;
                    i++;
                }
                else if((i==min_gap1_num)||(i==min_gap2_num)||(i==min_gap3_num)||(i==min_gap4_num)
                      ||(i==min_gap5_num)||(i==min_gap6_num)||(i==min_gap7_num)||(i==min_gap8_num))
                {
                    char_range_t connect_peek;
                    connect_peek.begin = peek_range_v.at(i).begin;
                    connect_peek.end = peek_range_v.at(i+1).end;
                    char_rang.push_back(connect_peek);
                    i++;
                }
                else
                {
                    char_rang.push_back(peek_range_v.at(i));
                }
            }
        }
        break;
    default:
        {
            for(int i=0;i<peek_range_v.size();i++)
            {
                char_rang.push_back(peek_range_v.at(i));
            }
        }
        break;
    }
}

void TextDetector::segmentSrcProject_l(cv::Mat &spineGray, vector<Mat> &single_char_vec)
{
    srand((unsigned)time(NULL));

    ////gauss smoothing
    int m_GaussianBlurSize = 5;
    Mat mat_blur;
    GaussianBlur(spineGray, mat_blur, Size(m_GaussianBlurSize, m_GaussianBlurSize), 0, 0, BORDER_DEFAULT);

    ////histequal and sharpen
    Mat spineGrayTemp = mat_blur - 0.5;
    cv::Mat spineAhe;
    adaptiveHistEqual_l(spineGrayTemp, spineAhe, 0.01);
    cv::Mat spineShrpen;
    sharpenImage_l(spineAhe, spineShrpen);

    ////threshold
    cv::Mat thresh_src;
    threshold(spineShrpen, thresh_src, 0, 255, THRESH_OTSU+ CV_THRESH_BINARY);
    cv::Mat idx;
    findNonZero(thresh_src, idx);
    int one_count = (int)idx.total();
    float one_percent = (float)one_count/(float)thresh_src.total();
    std::cout<<"one_percent = "<<one_percent<<std::endl;
    if(one_percent>0.6)
    {
        inv_bin = true;
    }
    else
    {
        inv_bin = false;
    }

    cv::Mat thresh_src_temp;
    if(inv_bin)
    {
        threshold(spineShrpen, thresh_src_temp, 0, 255, THRESH_OTSU+ CV_THRESH_BINARY_INV);
        thresh_src = thresh_src_temp;
    }


    ////morphological open
    Mat element_src = getStructuringElement(MORPH_RECT, Size(2, 2));
    Mat open_src;
//    morphologyEx(thresh_src,open_src,MORPH_OPEN,element_src);
    open_src = thresh_src.clone();
#ifdef DEBUG
    while(1)
    {
      imshow("open_src", open_src);
      if(char(cvWaitKey(15))==27)break;
    }
#endif

    ////delete edge noise
    ////find contours
    cv::Mat thres_window = open_src.clone();
    Mat img_contours;
    thres_window.copyTo(img_contours);
    vector<vector<Point> > contours;
    findContours(img_contours,
                 contours,               // a vector of contours
                 CV_RETR_EXTERNAL,       // retrieve the external contours
                 CV_CHAIN_APPROX_NONE);  // all pixels of each contours
    Mat sepertate_im(thres_window.size(),thres_window.depth(),Scalar(255));
    drawContours(sepertate_im,contours,-1,Scalar(0),2);
#ifdef DEBUG
    while(1)
    {
      imshow("sepertate_im",sepertate_im);
      if(char(cvWaitKey(15))==27)break;
    }
    std::cout<<"find contours = "<<contours.size()<<std::endl;
#endif

    ////detect noise contours
    vector<vector<Point> > contours_remove;
    Mat sepertate_im_remove(thres_window.size(),thres_window.depth(),Scalar(255));
    //移除过长或过短的轮廓
    int cmin = 3; //最小轮廓长度
    int cmax = 2000;    //最大轮廓
    vector<vector<Point> >::iterator itc_mm = contours.begin();
    while (itc_mm!=contours.end())
    {
//        std::cout<<"itc_mm->size() = "<<itc_mm->size()<<std::endl;
        if (itc_mm->size() < cmin || itc_mm->size() > cmax)
        {
           contours_remove.push_back(*itc_mm);
           itc_mm = contours.erase(itc_mm);
#ifdef DEBUG
           std::cout<<"erase this contour for too short or too long !!!"<<std::endl;
#endif
        }
        else
            ++itc_mm;
    }

    // edge contours
//    itc_mm = contours.begin();
//    while (itc_mm!=contours.end())
//    {
//        //caculate contours mass center
//        vector<Point>::iterator itc_p = (*itc_mm).begin();
//        bool edge_cont = false;
//        Moments mu=  moments( *itc_mm, false );
//        Point2f mc = Point2f( mu.m10/mu.m00 , mu.m01/mu.m00 );

//        while(itc_p!=(*itc_mm).end())
//        {
//           if((itc_p->x<=2)||(itc_p->x>=thres_window.cols - 2)||(itc_p->y<=2)||(itc_p->y>=thres_window.rows - 2))
//           {
//               edge_cont = true;
//               break;
//           }
//           else
//           {
//               itc_p++;
//           }
//        }
////        std::cout<<"edge_cout = "<<edge_cont<<std::endl;
////        std::cout<<"mc = "<<mc.x<<" , "<<mc.y<<std::endl;
//        if(edge_cont&&((mc.x<0.03*open_src.cols)||(mc.x>0.97*open_src.cols)||(mc.y<0.03*open_src.rows)||(mc.y>0.97*open_src.rows)))
//        {
//             contours_remove.push_back(*itc_mm);
//             itc_mm = contours.erase(itc_mm);
//#ifdef DEBUG
//             std::cout<<"erase this contour for edge !!!"<<std::endl;
//#endif
//        }
//        else
//            ++itc_mm;
//    }

    /// isolate contours
    if(contours.size()!=1)
    {
        removeIsoContour_l(contours,contours_remove);
    }
    drawContours(sepertate_im_remove,contours,-1,Scalar(0),2);
#ifdef DEBUG
    while(1)
    {
      imshow("sepertate_im_remove",sepertate_im_remove);
      if(char(cvWaitKey(15))==27)break;
    }
#endif


    //delete noise contous
    for(int i=0;i<open_src.rows;i++)
    {
        for(int j=0;j<open_src.cols;j++)
        {
            vector<vector<Point> >::iterator itc_det = contours_remove.begin();
            while(itc_det != contours_remove.end())
            {
                int inner_point = pointPolygonTest(*itc_det,Point(j,i),false);
                if((inner_point == 1)||(inner_point == 0))
                {
                    open_src.at<uchar>(i, j) = 0;
                    break;
                }
                itc_det++;
            }
        }
    }
#ifdef DEBUG
    while(1)
    {
      imshow("open_src_remove",open_src);
      if(char(cvWaitKey(15))==27)break;
    }

//    cvDestroyWindow("open_sr");
//    cvDestroyWindow("sepertate_im_remove");
//    cvDestroyWindow("sepertate_im");
//    cvDestroyWindow("open_src_remove");
#endif



    //row segment
    Mat charRow,charRowTem;
    segmentRow_l(open_src,charRowTem,0.12);
    cv::resize(charRowTem,charRow,cv::Size(PIECEWIDTH,PIECEHEIGHT));
#ifdef DEBUG
        while(1)
        {
          imshow( "charRow", charRow );
          if(char(cvWaitKey(15))==27)break;
        }
#endif


    ////rotation estimate and rotation rectify
    Mat canImage(cv::Size(charRow.cols,charRow.rows),IPL_DEPTH_8U,1);
    Canny(charRow,canImage,30,200,3);
    vector<Vec2f> lines;
    HoughLines(canImage, lines, 1, CV_PI / 180, 30, 0, 0);
    int numLine=0;
    float sumAng=0.0;
    Mat charRowRGB(cv::Size(charRow.cols,charRow.rows),charRow.depth(),3);
    cvtColor(charRow,charRowRGB,CV_GRAY2RGB);
    for(int i=0;i<lines.size();i++)
    {
        float theta=lines[i][1];
        float rho = lines[i][0];
        if(theta>=0&&theta<15*CV_PI/180)
        {
            Point pt1, pt2;
            //cout << theta << endl;
            double a = cos(theta), b = sin(theta);
            double x0 = a*rho, y0 = b*rho;
            pt1.x = cvRound(x0 + 1000 * (-b));
            pt1.y = cvRound(y0 + 1000 * (a));
            pt2.x = cvRound(x0 - 1000 * (-b));
            pt2.y = cvRound(y0 - 1000 * (a));
            cv::line(charRowRGB, pt1, pt2, Scalar(55, 100, 195), 1, LINE_AA); //Scalar函数用于调节线段颜色

            numLine++;
            sumAng=sumAng+theta;
        }
    }
    float avAng;
    if(numLine!=0)
    {
        avAng=(sumAng/numLine)*180/CV_PI;
    }
    else
        avAng = 0;


    printf("avAng =%f\n",avAng );

//    while(1)
//    {
//      imshow( "charRowRGB", charRowRGB );
//      if(char(cvWaitKey(15))==27)break;
//    }
//    cvDestroyWindow("charRowRGB");


    Mat rotImage,rotImageTem;
    rotateImg_l(charRow, rotImageTem, avAng);
    cv::resize(rotImageTem,rotImage,cv::Size(PIECEWIDTH,PIECEHEIGHT));
    charRow= rotImage.clone();

//    while(1)
//    {
//      imshow( "rotImage", rotImage );
//      if(char(cvWaitKey(15))==27)break;
//    }
//    cvDestroyWindow("rotImage");


    //cols segment for main row
    vector<int> pos_v;
    pos_v.resize(charRow.cols,0);
    vector<char_range_t> peek_range_v;
    vector<char_range_t> char_rang;
    char_rang.clear();
    GetTextProjection_l(charRow,pos_v,V_PROJECT);
//    draw_projection_l(pos_v,V_PROJECT);
    GetPeekRange_l(pos_v,peek_range_v,1,2);

    //segment analyse
    segmentAnalyse_l(peek_range_v,char_rang);


    //connect closer neighbour peek_range,version 1.0
//    for(int i =0;i<peek_range_v.size();i++)
//    {   int peek_rang = peek_range_v.at(i).end - peek_range_v.at(i).begin;
//        if(peek_rang>0.01*charRow.cols)
//        {
//            if(i<peek_range_v.size()-1)
//            {
//                int peek_rang_next = peek_range_v.at(i+1).end - peek_range_v.at(i+1).begin;
//                int peek_rang_gap = peek_range_v.at(i+1).begin - peek_range_v.at(i).end;
//                if((peek_rang_next>0.01*charRow.cols)&&(peek_rang_gap <0.015*charRow.cols))
//                {
//                    if(i<peek_range_v.size()-2)
//                    {
//                        int peek_rang_next_next = peek_range_v.at(i+2).end - peek_range_v.at(i+2).begin;
//                        int peek_rang_gap_next = peek_range_v.at(i+2).begin - peek_range_v.at(i+1).end;
//                        if((peek_rang_next_next>0.01*charRow.cols)&&(peek_rang_gap_next <0.015*charRow.cols))
//                        {
//                            char_range_t connect_peek;
//                            connect_peek.begin = peek_range_v.at(i).begin;
//                            connect_peek.end = peek_range_v.at(i+2).end;
//                            char_rang.push_back(connect_peek);
//                            i = i+2;
//                        }
//                        else
//                        {
//                            char_range_t connect_peek;
//                            connect_peek.begin = peek_range_v.at(i).begin;
//                            connect_peek.end = peek_range_v.at(i+1).end;
//                            char_rang.push_back(connect_peek);
//                            i++;
//                        }
//                    }
//                    else
//                    {
//                        char_range_t connect_peek;
//                        connect_peek.begin = peek_range_v.at(i).begin;
//                        connect_peek.end = peek_range_v.at(i+1).end;
//                        char_rang.push_back(connect_peek);
//                        i++;
//                    }

//                }
//                else
//                {
//                    char_rang.push_back(peek_range_v.at(i));
//                }
//            }
//            else
//            {
//                char_rang.push_back(peek_range_v.at(i));
//            }

//        }
//    }

    vector<char_range_t>::iterator itc = char_rang.begin();
    vector<Rect> vecRect_temp;
    while (itc != char_rang.end())
    {
        Rect charRec((*itc).begin,0,(*itc).end-(*itc).begin,charRow.rows);
        vecRect_temp.push_back(charRec);
        ++itc;
    }




    ////seperate big char(connect char)
    vector<Rect> vecRect;
    if(vecRect_temp.size()<4)
    {
        for(int i = 0;i<vecRect_temp.size();i++)
        {
            if(vecRect_temp.at(i).width>=0.75*PIECEWIDTH)//4 char connect
            {
                int piece4 = vecRect_temp.at(i).width/4;
                int piece_h = vecRect_temp.at(i).height;
                Rect charRec1(vecRect_temp.at(i).x,vecRect_temp.at(i).y,piece4,piece_h);
                Rect charRec2(vecRect_temp.at(i).x+1*piece4,vecRect_temp.at(i).y,piece4,piece_h);
                Rect charRec3(vecRect_temp.at(i).x+2*piece4,vecRect_temp.at(i).y,piece4,piece_h);
                Rect charRec4(vecRect_temp.at(i).x+3*piece4,vecRect_temp.at(i).y,piece4,piece_h);
                vecRect.push_back(charRec1);
                vecRect.push_back(charRec2);
                vecRect.push_back(charRec3);
                vecRect.push_back(charRec4);
            }
            else if(vecRect_temp.at(i).width>=0.55*PIECEWIDTH)//3 char connect
            {
                int piece3 = vecRect_temp.at(i).width/3;
                int piece_h = vecRect_temp.at(i).height;
                Rect charRec1(vecRect_temp.at(i).x,vecRect_temp.at(i).y,piece3,piece_h);
                Rect charRec2(vecRect_temp.at(i).x+1*piece3,vecRect_temp.at(i).y,piece3,piece_h);
                Rect charRec3(vecRect_temp.at(i).x+2*piece3,vecRect_temp.at(i).y,piece3,piece_h);
                vecRect.push_back(charRec1);
                vecRect.push_back(charRec2);
                vecRect.push_back(charRec3);
            }
            else if(vecRect_temp.at(i).width>=0.35*PIECEWIDTH)//2 char connect
            {
                int piece2 = vecRect_temp.at(i).width/2;
                int piece_h = vecRect_temp.at(i).height;
                Rect charRec1(vecRect_temp.at(i).x,vecRect_temp.at(i).y,piece2,piece_h);
                Rect charRec2(vecRect_temp.at(i).x+1*piece2,vecRect_temp.at(i).y,piece2,piece_h);
                vecRect.push_back(charRec1);
                vecRect.push_back(charRec2);
            }
            else
            {
                vecRect.push_back(vecRect_temp.at(i));
            }
        }
    }
    else
    {
        for(int i = 0;i<vecRect_temp.size();i++)
        {
            vecRect.push_back(vecRect_temp.at(i));
        }
    }


    ////save single char image after segment
    for(int char_num=0;char_num<vecRect.size();char_num++)
    {
         Mat single_char_=charRow(vecRect.at(char_num));//modify on 0309
//         Mat single_char_seg;
//         segmentRow(single_char_,single_char_seg,1.2);

         //delete too small object(noise)
         cv::Mat idx_char;
         findNonZero(single_char_, idx_char);
         int one_count = (int)idx_char.total();
         float one_percent = (float)one_count/(float)single_char_.total();
//         std::cout<<"one_percent = "<<one_percent<<std::endl;
         if(one_percent>0.15)
         {
             Mat single_char;
//             ////进行open操作
//             Mat char_sob_ele = getStructuringElement(MORPH_RECT, Size(2, 2));
//             Mat char_sob;
//             morphologyEx(single_char,char_sob,MORPH_OPEN,element_sob);
             single_char = preprocessChar_l(single_char_);//modify on 0309
             single_char_vec.push_back(single_char);
         }
#ifdef DEBUG
        while(1)
        {
          imshow( "single_char", single_char_ );
          if(char(cvWaitKey(15))==27)break;
        }
#endif
    }

//#ifdef DEBUG
    cvDestroyWindow("single_char");
    cvDestroyWindow("charRow");

//#endif
}

//void TextDetector::imgQuantize(cv::Mat &src, cv::Mat &dst, double level){
//    dst = cv::Mat::zeros(src.rows, src.cols, CV_8U);
//    for (int i = 0; i < src.rows; i ++) {
//        uchar *data = src.ptr<uchar>(i);
//        uchar *data2 = dst.ptr<uchar>(i);
//        for (int j = 0; j < src.cols; j ++) {
//            if(data[j] <= level)
//                data2[j] = 1;
//            else
//                data2[j] = 2;
                
//        }
//    }
    
//}


