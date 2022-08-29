/**
 * @file detection.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-06-03
 *
 * @copyright Copyright (c) 2022
 *
 */
#include "jetson/detection.hpp"
#include <algorithm>
#include <array>
#include <unordered_map>
#include <vector>

namespace infer {
namespace trt {

float DetectionInfer::iou(std::array<float, 4> const &lbox,
                         std::array<float, 4> const &rbox) const {
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

void DetectionInfer::nms(
    std::vector<DetectionResult> &res,
    std::unordered_map<int, std::vector<DetectionResult>> &m) const {
  for (auto it = m.begin(); it != m.end(); it++) {
    // std::cout << it->second[0].class_id << " --- " << std::endl;
    auto &dets = it->second;
    std::sort(dets.begin(), dets.end(), compare);
    for (size_t m = 0; m < dets.size(); ++m) {
      auto &item = dets[m];
      res.push_back(item);
      for (size_t n = m + 1; n < dets.size(); ++n) {
        if (iou(item.bbox, dets[n].bbox) > mParams.nms_thr) {
          dets.erase(dets.begin() + n);
          --n;
        }
      }
    }
  }
}

void DetectionInfer::renderOriginShape(std::vector<DetectionResult> &results,
                                      std::array<int, 3> const &shape) const {
  float rw, rh;
  if (mParams.isScale) {
    rw = std::min(mParams.inputShape[0] * 1.0 / shape.at(0),
                  mParams.inputShape[1] * 1.0 / shape.at(1));
    rh = rw;
  } else {
    rw = mParams.inputShape[0] * 1.0 / shape.at(0);
    rh = mParams.inputShape[1] * 1.0 / shape.at(1);
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

bool DetectionInfer::processOutput(BufferManager const &buffers,
                                  Result &result) const {
  std::unordered_map<int, std::vector<DetectionResult>> cls2bbox;
  generateBoxes(cls2bbox, buffers);
  nms(result.detResults, cls2bbox);
  // rect 还原成原始大小
  renderOriginShape(result.detResults, result.shape);
  std::vector<DetectionResult>::iterator it = result.detResults.begin();
  // 清除掉不符合要求的框
  for (; it != result.detResults.end();) {
    cv::Rect rect{static_cast<int>(it->bbox[0]), static_cast<int>(it->bbox[1]),
                  static_cast<int>(it->bbox[2] - it->bbox[0]),
                  static_cast<int>(it->bbox[3] - it->bbox[1])};
    if (rect.area() < 2 * 2)
      it = result.detResults.erase(it);
    else
      //迭代器指向下一个元素位置
      ++it;
  }

  return true;
};
} // namespace trt
} // namespace infer