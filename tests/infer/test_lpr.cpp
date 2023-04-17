#include <algorithm>
#include <cassert>
#include <cmath>
#include <codecvt>
#include <cstdlib>
#include <gflags/gflags.h>
#include <iostream>

#include <Eigen/Dense>

#include <locale>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <utility>

#include "preprocess.hpp"
#include "visionInfer.hpp"

DEFINE_string(img, "", "Specify face1 image path.");
DEFINE_string(det_model_path, "", "Specify the lprDet model path.");
DEFINE_string(rec_model_path, "", "Specify the lprNet model path.");

using namespace infer;
using namespace common;

using algo_ptr = std::shared_ptr<AlgoInfer>;

const std::wstring_view charsets =
    L"#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂"
    L"使领民航深0123456789ABCDEFGHJKLMNPQRSTUVWXYZ";

algo_ptr getVision(AlgoConfig &&config) {

  std::shared_ptr<AlgoInfer> vision = std::make_shared<VisionInfer>(config);
  if (!vision->init()) {
    FLOWENGINE_LOGGER_ERROR("Failed to init vision");
    std::exit(-1); // 强制中断
    return nullptr;
  }
  return vision;
}

void inference(cv::Mat &image, InferResult &ret,
               std::shared_ptr<AlgoInfer> vision) {

  RetBox region{"hello", {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};

  InferParams params{std::string("hello"),
                     ColorType::NV12,
                     0.0,
                     region,
                     {image.cols, image.rows, image.channels()}};

  vision->infer(image.data, params, ret);
}

std::string getChars(CharsRet const &charsRet) {
  std::wstring chars = L"";
  for (auto it = charsRet.begin(); it != charsRet.end(); ++it) {
    auto c = charsets.at(*it);
    chars += c;
  }
  // 定义一个UTF-8编码的转换器
  std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
  // 将宽字符字符串转换为UTF-8编码的多字节字符串
  std::string str = converter.to_bytes(chars);
  return str;
}

void splitMerge(cv::Mat const &image, cv::Mat &output) {
  int h = image.rows;
  int w = image.cols;
  cv::Rect upperRect{0, 0, w, static_cast<int>(5. / 12. * h)};
  cv::Rect lowerRect{0, static_cast<int>(1. / 3. * h), w,
                     h - static_cast<int>(1. / 3. * h)};
  cv::Mat imageUpper;
  cv::Mat imageLower = image(lowerRect);
  cv::resize(image(upperRect), imageUpper,
             cv::Size(imageLower.cols, imageLower.rows));
  cv::hconcat(imageUpper, imageLower, output);
};

void fourPointTransform(cv::Mat &input, cv::Mat &output,
                        infer::Points2f const &points) {
  assert(points.size() == 4);
  infer::Point2f tl = points[0];
  infer::Point2f tr = points[1];
  infer::Point2f br = points[2];
  infer::Point2f bl = points[3];
  // 计算相对位置
  auto min_x_p = std::min_element(
      points.begin(), points.end(),
      [](auto const &p1, auto const &p2) { return p1.x < p2.x; });
  auto min_y_p = std::min_element(
      points.begin(), points.end(),
      [](auto const &p1, auto const &p2) { return p1.y < p2.y; });
  int min_x = min_x_p->x;
  int min_y = min_y_p->y;
  tl.x -= min_x;
  tl.y -= min_y;
  bl.x -= min_x;
  bl.y -= min_y;
  tr.x -= min_x;
  tr.y -= min_y;
  br.x -= min_x;
  br.y -= min_y;
  int widthA = std::sqrt(std::pow(br.x - bl.x, 2) + std::pow(br.y - bl.y, 2));
  int widthB = std::sqrt(std::pow(tr.x - tl.x, 2) + std::pow(tr.y - tl.y, 2));
  int maxWidth = std::max(widthA, widthB);

  int heightA = std::sqrt(std::pow(tr.x - br.x, 2) + std::pow(tr.y - br.y, 2));
  int heightB = std::sqrt(std::pow(tl.x - bl.x, 2) + std::pow(tl.y - bl.y, 2));
  int maxHeight = std::max(heightA, heightB);

  // 定义原始图像坐标和变换后的目标图像坐标
  cv::Point2f src[4] = {cv::Point2f(tl.x, tl.y), cv::Point2f(tr.x, tr.y),
                        cv::Point2f(br.x, br.y), cv::Point2f(bl.x, bl.y)};
  cv::Point2f dst[4] = {cv::Point2f(0, 0), cv::Point2f(maxWidth - 1, 0),
                        cv::Point2f(maxWidth - 1, maxHeight - 1),
                        cv::Point2f(0, maxHeight - 1)};

  // 计算透视变换矩阵
  cv::Mat M = getPerspectiveTransform(src, dst);

  // 对原始图像进行透视变换
  cv::warpPerspective(input, output, M, cv::Size(maxWidth, maxHeight));
}

// 按照左上、右上、右下、左下的顺序排序
void sortFourPoints(Points2f &points) {
  assert(points.size() == 4);
  Point2f x1y1, x2y2, x3y3, x4y4;
  // 先对x排序，取出前两个根据y的大小决定左上和左下，后两个点根据y的大小决定右上和右下
  std::sort(points.begin(), points.end(),
            [](Point2f const &p1, Point2f const &p2) { return p1.x < p2.x; });
  if (points[0].y <= points[1].y) {
    x1y1 = points[0];
    x4y4 = points[1];
  } else {
    x1y1 = points[1];
    x4y4 = points[0];
  }
  if (points[2].y <= points[3].y) {
    x2y2 = points[2];
    x3y3 = points[3];
  } else {
    x2y2 = points[3];
    x3y3 = points[2];
  }
  points = {x1y1, x2y2, x3y3, x4y4};
  // for (auto &p : points) {
  //   std::cout << "x: " << p.x << ", "
  //             << "y: " << p.y << std::endl;
  // }
}

int main(int argc, char **argv) {

  gflags::SetUsageMessage("some usage message");
  gflags::SetVersionString("1.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  PointsDetAlgo lprDet_config{{
                                  1,
                                  {"images"},
                                  {"output"},
                                  FLAGS_det_model_path,
                                  "YoloPDet",
                                  {640, 640, 3},
                                  false,
                                  0,
                                  0,
                                  0.3,
                              },
                              4,
                              0.4};
  AlgoConfig pdet_config;
  pdet_config.setParams(lprDet_config);

  ClassAlgo lprNet_config{{
      1,
      {"images"},
      {"output"},
      FLAGS_rec_model_path,
      "CRNN",
      {176, 48, 3},
      false,
      0,
      0,
      0.3,
  }};
  AlgoConfig prec_config;
  prec_config.setParams(lprNet_config);

  auto lprDet = getVision(std::move(pdet_config));
  auto lprNet = getVision(std::move(prec_config));

  // 图片读取
  cv::Mat image_bgr = cv::imread(FLAGS_img);

  cv::Mat image_rgb;
  cv::cvtColor(image_bgr, image_rgb, cv::COLOR_BGR2RGB);

  cv::Mat image_nv12;
  infer::utils::RGB2NV12(image_rgb, image_nv12);

  InferResult lprDetRet;
  inference(image_nv12, lprDetRet, lprDet);

  auto pbboxes = std::get_if<KeypointsBoxes>(&lprDetRet.aRet);
  if (!pbboxes) {
    FLOWENGINE_LOGGER_ERROR("Person detection is failed!");
    return -1;
  }

  std::vector<std::pair<std::string, KeypointsBox>> results;

  for (auto &kbbox : *pbboxes) {

    cv::Mat licensePlateImage;
    cv::Rect2i rect{static_cast<int>(kbbox.bbox.bbox[0]),
                    static_cast<int>(kbbox.bbox.bbox[1]),
                    static_cast<int>(kbbox.bbox.bbox[2] - kbbox.bbox.bbox[0]),
                    static_cast<int>(kbbox.bbox.bbox[3] - kbbox.bbox.bbox[1])};
    infer::utils::cropImage(image_nv12, licensePlateImage, rect,
                            ColorType::NV12);
    if (kbbox.bbox.class_id == 1) { // 双层车牌情况
      // 车牌矫正
      sortFourPoints(kbbox.points); // 排序关键点
      cv::Mat lpr_rgb;
      cv::cvtColor(licensePlateImage, lpr_rgb, cv::COLOR_YUV2RGB_NV12);
      cv::Mat lpr_ted;
      // 相对于车牌图片的点
      fourPointTransform(lpr_rgb, lpr_ted, kbbox.points);

      // 上下分割，垂直合并车牌
      splitMerge(lpr_ted, licensePlateImage);
      utils::RGB2NV12(licensePlateImage, licensePlateImage);
      cv::imwrite("test_lpr_plate_src.jpg", lpr_rgb);
      cv::imwrite("test_lpr_plate_dst.jpg", lpr_ted);
      cv::imwrite("test_lpr_plate_input.jpg", licensePlateImage);
    }
    InferResult lprRecRet;
    inference(licensePlateImage, lprRecRet, lprNet);

    auto charIds = std::get_if<CharsRet>(&lprRecRet.aRet);
    if (!charIds) {
      FLOWENGINE_LOGGER_ERROR("OCRModule: Wrong algorithm type!");
      return -1;
    }

    std::string lprNumbers = getChars(*charIds);
    std::cout << "license-plate number is: " << lprNumbers << std::endl;

    results.emplace_back(std::make_pair(lprNumbers, kbbox));
  }
  for (auto &ret : results) {
    cv::Rect rect(ret.second.bbox.bbox[0], ret.second.bbox.bbox[1],
                  ret.second.bbox.bbox[2] - ret.second.bbox.bbox[0],
                  ret.second.bbox.bbox[3] - ret.second.bbox.bbox[1]);
    cv::rectangle(image_bgr, rect, cv::Scalar(0, 0, 255), 2);
    for (auto &p : ret.second.points) {
      cv::circle(image_bgr,
                 cv::Point{static_cast<int>(p.x), static_cast<int>(p.y)}, 3,
                 cv::Scalar{255, 255, 0});
    }
    cv::putText(image_bgr, ret.first, cv::Point(rect.x, rect.y),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 14, 50), 2);
  }
  cv::imwrite("test_lpr_out.jpg", image_bgr);

  gflags::ShutDownCommandLineFlags();
  return 0;
}

/*
./test_face --img1 /root/workspace/softwares/flowengine/data/face1.jpg \
            --img2 /root/workspace/softwares/flowengine/data/face2.jpg \
            --model_path
/root/workspace/softwares/flowengine/models/facenet_mobilenet_160x160.bin
*/