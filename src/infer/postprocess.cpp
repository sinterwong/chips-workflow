/**
 * @file postprocess.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-02-14
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "postprocess.hpp"
#include "logger/logger.hpp"

namespace infer {
namespace utils {

float iou(std::array<float, 4> const &lbox, std::array<float, 4> const &rbox) {
  float interBox[] = {
      std::max(lbox[0] - lbox[2] / 2.f, rbox[0] - rbox[2] / 2.f), // left
      std::min(lbox[0] + lbox[2] / 2.f, rbox[0] + rbox[2] / 2.f), // right
      std::max(lbox[1] - lbox[3] / 2.f, rbox[1] - rbox[3] / 2.f), // top
      std::min(lbox[1] + lbox[3] / 2.f, rbox[1] + rbox[3] / 2.f), // bottom
  };

  if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
    return 0.0f;

  float interBoxS = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2]);
  return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
}

void nms(DetRet &res, std::unordered_map<int, DetRet> &temp, float nms_thr) {
  for (auto it = temp.begin(); it != temp.end(); it++) {
    // std::cout << it->second[0].class_id << " --- " << std::endl;
    auto &dets = it->second;
    std::sort(dets.begin(), dets.end(), compare);
    for (size_t i = 0; i < dets.size(); ++i) {
      auto &item = dets[i];
      res.push_back(item);
      for (size_t j = i + 1; j < dets.size(); ++j) {
        if (iou(item.bbox, dets[j].bbox) > nms_thr) {
          dets.erase(dets.begin() + j);
          --j;
        }
      }
    }
  }
}

void restoryBoxes(DetRet &results, Shape const &shape,
                  Shape const &inputShape, bool isScale) {
  float rw, rh;
  if (isScale) {
    rw = std::min(inputShape[0] * 1.0 / shape.at(0),
                  inputShape[1] * 1.0 / shape.at(1));
    rh = rw;
  } else {
    rw = inputShape[0] * 1.0 / shape.at(0);
    rh = inputShape[1] * 1.0 / shape.at(1);
  }

  for (auto &ret : results) {
    int l = (ret.bbox[0] - ret.bbox[2] / 2.f) / rw;
    int t = (ret.bbox[1] - ret.bbox[3] / 2.f) / rh;
    int r = (ret.bbox[0] + ret.bbox[2] / 2.f) / rw;
    int b = (ret.bbox[1] + ret.bbox[3] / 2.f) / rh;
    ret.bbox[0] = l > 0 ? l : 0;
    ret.bbox[1] = t > 0 ? t : 0;
    ret.bbox[2] = r < shape[0] ? r : shape[0];
    ret.bbox[3] = b < shape[1] ? b : shape[1];
  }
}

void restoryPoints(PoseRet &results, Shape const &shape,
                   Shape const &inputShape, bool isScale) {}

} // namespace utils
} // namespace infer