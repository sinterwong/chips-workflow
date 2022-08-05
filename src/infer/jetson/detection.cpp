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
#include "inference.h"
#include <algorithm>
#include <array>
#include <vector>

namespace infer {
namespace trt {

float DetctionInfer::iou(std::array<float, 4> const &lbox,
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

void DetctionInfer::nms(std::vector<DetectionResult> &res,
                        float *output) const {
  std::map<float, std::vector<DetectionResult>> m;
  // output += ((mParams.numAnchors) * (mParams.numClasses + 5));
  int num = mParams.numClasses + 5;
  for (int i = 0; i < mParams.numAnchors * num; i += num) {
    if (output[i + 4] <= mParams.cond_thr)
      continue;

    DetectionResult det;
    det.class_id = std::distance(
        output + i + 5, std::max_element(output + i + 5, output + i + num));
    int real_idx = i + 5 + det.class_id;
    det.det_confidence = output[real_idx];
    // std::cout << "**************************" << std::endl;
    // std::cout << det.class_id << std::endl;
    // std::cout << det.det_confidence << std::endl;
    // std::cout << "**************************" << std::endl;
    // exit(-1);
    memcpy(&det, &output[i], 5 * sizeof(float));
    if (m.count(det.class_id) == 0)
      m.emplace(det.class_id, std::vector<DetectionResult>());
    m[det.class_id].push_back(det);
  }
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

void DetctionInfer::renderOriginShape(std::vector<DetectionResult> &results,
                                      std::array<int, 3> const &shape) const {
  for (auto &ret : results) {
    int l, r, t, b;
    float ratio = std::min(mParams.inputShape[0] * 1.0 / shape.at(0),
                           mParams.inputShape[1] * 1.0 / shape.at(1));
    // if (r_h > r_w) {
    //   l = ret.bbox[0] - ret.bbox[2] / 2.f;
    //   r = ret.bbox[0] + ret.bbox[2] / 2.f;
    //   t = ret.bbox[1] - ret.bbox[3] / 2.f - (mParams.inputShape.at(1) - r_w *
    //   shape.at(1)) / 2; b = ret.bbox[1] + ret.bbox[3] / 2.f -
    //   (mParams.inputShape.at(1) - r_w * shape.at(1)) / 2; l = l / r_w; r = r
    //   / r_w; t = t / r_w; b = b / r_w;
    // } else {
    //   l = ret.bbox[0] - ret.bbox[2] / 2.f - (mParams.inputShape.at(0) - r_h *
    //   shape.at(0)) / 2; r = ret.bbox[0] + ret.bbox[2] / 2.f -
    //   (mParams.inputShape.at(0) - r_h * shape.at(0)) / 2; t = ret.bbox[1] -
    //   ret.bbox[3] / 2.f; b = ret.bbox[1] + ret.bbox[3] / 2.f; l = l / r_h; r
    //   = r / r_h; t = t / r_h; b = b / r_h;
    // }
    l = (ret.bbox[0] - ret.bbox[2] / 2.f) / ratio;
    r = (ret.bbox[0] + ret.bbox[2] / 2.f) / ratio;
    t = (ret.bbox[1] - ret.bbox[3] / 2.f) / ratio;
    b = (ret.bbox[1] + ret.bbox[3] / 2.f) / ratio;

    ret.bbox[0] = l;
    ret.bbox[1] = t;
    ret.bbox[2] = r;
    ret.bbox[3] = b;
  }
}

bool DetctionInfer::verifyOutput(Result const &result) const { return true; }

bool DetctionInfer::processOutput(BufferManager const &buffers,
                                  Result &result) const {
  float *hostOutputBuffer =
      static_cast<float *>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
  int outputSize = mParams.numAnchors * (mParams.numClasses + 5);
  // std::vector<DetectionResult> res;
  // nms(res, &hostOutputBuffer[b * outputSize]);  // if batch_size >
  // result.detResults.emplace_back(res);
  nms(result.detResults, hostOutputBuffer);
  // rect 还原成原始大小
  renderOriginShape(result.detResults, result.shape);
  return true;
};
} // namespace trt
} // namespace infer