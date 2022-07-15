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
#include "inference.h"
#include "jetson/detection.hpp"
#include <algorithm>
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

void DetctionInfer::nms(std::vector<DetectionResult> &res, float *output) const {
  std::map<float, std::vector<DetectionResult>> m;
  // output += ((mParams.numAnchors) * (mParams.numClasses + 5));
  int num = mParams.numClasses + 5;
  for (int i = 0; i < mParams.numAnchors * num; i += num) {
    if (output[i + 4] <= mParams.c_thr)
      continue;

    // // ---------- printer --------------
    // for (int b = 0; b < num; b++) {
    //   std::cout << output[i + b] << " ";
    // }
    // std::cout << std::endl;
    // // _________________________________
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

bool DetctionInfer::verifyOutput(Result const &result) const { return true; }

bool DetctionInfer::processOutput(BufferManager const &buffers,
                              Result &result) const {
  float *hostOutputBuffer =
      static_cast<float *>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
  int outputSize = mParams.numAnchors * (mParams.numClasses + 5);
  for (int b = 0; b < mParams.batchSize; b++) {
    std::vector<DetectionResult> res;
    nms(res, &hostOutputBuffer[b * outputSize]);
    result.detResults.push_back(res);
  }
  return true;
};
} // namespace trt
} // namespace infer