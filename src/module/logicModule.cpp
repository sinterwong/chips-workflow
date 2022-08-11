/**
 * @file logicModule.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-08-10
 *
 * @copyright Copyright (c) 2022
 *
 */

#include "logicModule.h"

namespace module {

LogicModule::LogicModule(Backend *ptr, const std::string &initName,
                             const std::string &initType,
                             const common::LogicConfig &params_,
                             const std::vector<std::string> &recv,
                             const std::vector<std::string> &send,
                             const std::vector<std::string> &pool)
    : Module(ptr, initName, initType, recv, send, pool), params(std::move(params_)) {}


bool LogicModule::drawResult(cv::Mat &image, AlgorithmResult const &rm) {

  for (auto &bbox : rm.bboxes) {
    cv::Rect rect(bbox.second[0], bbox.second[1],
                  bbox.second[2] - bbox.second[0],
                  bbox.second[3] - bbox.second[1]);
    cv::rectangle(image, rect, cv::Scalar(255, 255, 0), 2);
    cv::putText(image, bbox.first, cv::Point(rect.x, rect.y - 1),
                cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(255, 0, 255), 2);
  }

  for (auto &poly : rm.polys) {
    std::vector<cv::Point> fillContSingle;
    for (int i = 0; i < poly.second.size(); i += 2) {
      fillContSingle.emplace_back(
          cv::Point{static_cast<int>(poly.second[i]),
                    static_cast<int>(poly.second[i + 1])});
    }
    cv::fillPoly(image, std::vector<std::vector<cv::Point>>{fillContSingle},
                 cv::Scalar(0, 255, 255));
  }

  return true;
}


} // namespace module
