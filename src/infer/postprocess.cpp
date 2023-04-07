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
#include <algorithm>

namespace infer::utils {

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

void nms(BBoxes &res, std::unordered_map<int, BBoxes> &temp, float nms_thr) {
  for (auto it = temp.begin(); it != temp.end(); it++) {
    // std::cout << it->second[0].class_id << " --- " << std::endl;
    BBoxes &dets = it->second;
    std::sort(dets.begin(), dets.end(), [](BBox const &a, BBox const &b) {
      return a.det_confidence > b.det_confidence;
    });
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

void nms_kbox(KeypointsBoxes &res,
              std::unordered_map<int, KeypointsBoxes> &temp, float nms_thr) {
  for (auto it = temp.begin(); it != temp.end(); it++) {
    // std::cout << it->second[0].class_id << " --- " << std::endl;
    KeypointsBoxes &kboxes = it->second;
    std::sort(kboxes.begin(), kboxes.end(),
              [](KeypointsBox const &a, KeypointsBox const &b) {
                return a.bbox.det_confidence > b.bbox.det_confidence;
              });
    for (size_t i = 0; i < kboxes.size(); ++i) {
      auto &item = kboxes[i];
      res.push_back(item);
      for (size_t j = i + 1; j < kboxes.size(); ++j) {
        if (iou(item.bbox.bbox, kboxes[j].bbox.bbox) > nms_thr) {
          kboxes.erase(kboxes.begin() + j);
          --j;
        }
      }
    }
  }
}

void scale_ration(float &rw, float &rh, Shape const &shape,
                  Shape const &inputShape, bool isScale) {
  if (isScale) {
    rw = std::min(inputShape[0] * 1.0 / shape.at(0),
                  inputShape[1] * 1.0 / shape.at(1));
    rh = rw;
  } else {
    rw = inputShape[0] * 1.0 / shape.at(0);
    rh = inputShape[1] * 1.0 / shape.at(1);
  }
}

void restoryBox(BBox &ret, float rw, float rh, Shape const &shape) {
  int l = (ret.bbox[0] - ret.bbox[2] / 2.f) / rw;
  int t = (ret.bbox[1] - ret.bbox[3] / 2.f) / rh;
  int r = (ret.bbox[0] + ret.bbox[2] / 2.f) / rw;
  int b = (ret.bbox[1] + ret.bbox[3] / 2.f) / rh;
  ret.bbox[0] = l > 0 ? l : 0;
  ret.bbox[1] = t > 0 ? t : 0;
  ret.bbox[2] = r < shape[0] ? r : shape[0];
  ret.bbox[3] = b < shape[1] ? b : shape[1];
}

void restoryPoints(Points2f &results, float rw, float rh) {
  for (auto &point : results) {
    point[0] /= rw;
    point[1] /= rh;
  }
}

/**
 * @brief 恢复原始比例坐标
 *
 * @param results
 * @param shape
 * @param inputShape
 * @param isScale
 */
void restoryBoxes(BBoxes &results, Shape const &shape, Shape const &inputShape,
                  bool isScale) {
  float rw, rh;
  scale_ration(rw, rh, shape, inputShape, isScale);

  for (auto &ret : results) {
    restoryBox(ret, rw, rh, shape);
  }
}

/**
 * @brief 恢复原始比例坐标
 *
 * @param results
 * @param shape
 * @param inputShape
 * @param isScale
 */
void restoryKeypointsBoxes(KeypointsBoxes &results, Shape const &shape,
                           Shape const &inputShape, bool isScale) {
  float rw, rh;
  scale_ration(rw, rh, shape, inputShape, isScale);
  for (auto &ret : results) {
    restoryBox(ret.bbox, rw, rh, shape); // 恢复bbox
    restoryPoints(ret.points, rw, rh);   // 恢复points
  }
}

} // namespace infer::utils