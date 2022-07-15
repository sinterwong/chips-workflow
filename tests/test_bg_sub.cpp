///运动物体检测――帧差法
#include "opencv2/opencv.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

//运动物体检测函数声明
bool MoveDetect(const cv::Mat &temp, const cv::Mat &frame, cv::Mat &result) {

  //平滑、帧差或背景差、二值化、膨胀、腐蚀。
  // 1.平滑处理
  cv::Mat dst_temp;
  cv::blur(temp, dst_temp, cv::Size(3, 3), cv::Point(-1, -1)); // filter2D

  cv::Mat dst_frame;
  cv::blur(frame, dst_frame, cv::Size(3, 3), cv::Point(-1, -1)); // filter2D

  // 2.帧差
  // 2.1将background和frame转为灰度图
  cv::Mat gray1, gray2;
  cv::cvtColor(dst_temp, gray1, cv::COLOR_BGR2GRAY);
  cv::cvtColor(dst_frame, gray2, cv::COLOR_BGR2GRAY);
  // 2.2.将background和frame做差
  cv::Mat diff;
  cv::absdiff(gray1, gray2, diff);
  // cv::imshow("absdiff", diff);

  // 3.对差值图diff_thresh进行阈值化处理  二值化
  cv::Mat diff_thresh;

  /*
  getStructuringElement()
  这个函数的第一个参数表示内核的形状，有三种形状可以选择。
  矩形：MORPH_RECT;
  交叉形：MORPH_CORSS;
  椭圆形：MORPH_ELLIPSE;
  第二和第三个参数分别是内核的尺寸以及锚点的位置
  */
  cv::Mat kernel_erode = getStructuringElement(
      cv::MORPH_RECT, cv::Size(3, 3)); //函数会返回指定形状和尺寸的结构元素。
  //调用之后，调用膨胀与腐蚀函数的时候，第三个参数值保存了getStructuringElement返回值的Mat类型变量。也就是element变量。
  cv::Mat kernel_dilate =
      getStructuringElement(cv::MORPH_RECT, cv::Size(18, 18));

  //进行二值化处理，选择50，255为阈值
  cv::threshold(diff, diff_thresh, 50, 255, cv::THRESH_BINARY);
  // cv::imshow("threshold", diff_thresh);
  // 4.膨胀
  cv::dilate(diff_thresh, diff_thresh, kernel_dilate);
  // cv::imshow("dilate", diff_thresh);
  // 5.腐蚀
  cv::erode(diff_thresh, diff_thresh, kernel_erode);
  // cv::imshow("erode", diff_thresh);

  // 6.查找轮廓并绘制轮廓
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(diff_thresh, contours, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_NONE); //找轮廓函数
  cv::drawContours(result, contours, -1, cv::Scalar(0, 0, 255),
                   2); //在result上绘制轮廓
  // 7.查找正外接矩形
  std::vector<cv::Rect> boundRect(contours.size());
  for (int i = 0; i < contours.size(); i++) {
    boundRect[i] = cv::boundingRect(contours[i]);
    cv::rectangle(result, boundRect[i], cv::Scalar(0, 255, 0),
                  2); //在result上绘制正外接矩形
  }
  return true;
}

int main() {
  int width = 320;
  int height = 240;
  //读取帧、平滑、帧差或背景差、二值化、膨胀、腐蚀。
  // std::string uri = "/home/wangxt/workspace/projects/flowengine/tests/data/sample_1080p_h264.mp4";
  std::string uri = "rtsp://admin:zkfd123.com@114.242.23.39:9303/Streaming/Channels/101";
  cv::VideoCapture video(uri);
  auto FPS = video.get(cv::CAP_PROP_FPS); //获取FPS
  cv::VideoWriter writer;
  writer.open("output.avi", cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), FPS,
              cv::Size(width * 2, height), true);
  ///////////////////////////////////////////////////////////////
  if (!video.isOpened() || !writer.isOpened()) {
    std::cout << "!!! Failed to open file: " << std::endl;
    exit(-1);
  }
  cv::Mat frame;  //存储帧
  cv::Mat temp;   //存储前一帧图像
  cv::Mat result; //存储结果图像
  int count = 0;
  while (video.read(frame)) {
    cv::resize(frame, frame, cv::Size(width, height));
    cv::Mat result = frame.clone();
    // cv::imshow("frame", frame);

    //处理帧
    if (count == 0) //如果为第一帧（temp还为空）
    {
      count++;
      //调用MoveDetect()进行运动物体检测，返回值存入result
      MoveDetect(frame, frame, result);
    } else {
      //调用MoveDetect()进行运动物体检测，返回值存入result
      MoveDetect(temp, frame, result);
    }
    cv::Mat showImage;
    cv::hconcat(frame, result, showImage);
    writer.write(showImage);
    cv::imshow("show image", showImage);
    temp = std::move(frame);
    cv::waitKey(27);
    if (count > 3000) {
      break;
    }
  }
  return 0;
}
