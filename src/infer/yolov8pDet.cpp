/**
 * @file yoloDet.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-11-07
 *
 * @copyright Copyright (c) 2022
 *
 */
#include "yolov8pDet.hpp"
#include "logger/logger.hpp"
#include "postprocess.hpp"

namespace infer::vision {

bool Yolov8PDet::processOutput(void **output, InferResult &result) const {
  std::unordered_map<int, KeypointsBoxes> cls2bbox;
  generateKeypointsBoxes(cls2bbox, output);
  auto rets = KeypointsBoxes();
  utils::nms_kbox(rets, cls2bbox, config.nmsThr);
  // rect 还原成原始大小
  utils::restoryKeypointsBoxes(rets, result.shape, config.inputShape,
                               config.isScale);
  KeypointsBoxes::iterator it = rets.begin();
  // 清除掉不符合要求的框
  for (; it != rets.end();) {
    int area = (it->bbox.bbox[3] - it->bbox.bbox[1]) *
               (it->bbox.bbox[2] - it->bbox.bbox[0]);
    if (area < 2 * 2)
      it = rets.erase(it);
    else
      // 迭代器指向下一个元素位置
      ++it;
  }
  result.aRet = std::move(rets);
  return true;
}

void Yolov8PDet::generateKeypointsBoxes(
    std::unordered_map<int, KeypointsBoxes> &m, void **outputs) const {
  float **output = reinterpret_cast<float **>(outputs);
  float *out = output[0]; // just one output
  // outputShapes: [1, 20, 8400]
  int numAnchors = modelInfo.outputShapes[0].at(2);
  // num = 4 + 1 + (numPoints * 3) = 5 + (numPoints * 3)
  int numPoints = config.numPoints * 3;

  // 遍历score行
  float *cxs = out;
  float *cys = out + numAnchors;
  float *widths = out + (2 * numAnchors);
  float *heights = out + (3 * numAnchors);
  float *scores = out + (4 * numAnchors);

  for (int i = 0; i < numAnchors; i++) {
    // 遍历每个score行的每个score
    if (scores[i] <= config.cond_thr)
      continue;
    KeypointsBox kBox;
    // build bbox
    kBox.bbox.bbox = {cxs[i], cys[i], widths[i], heights[i]};
    kBox.bbox.class_id = 0;
    kBox.bbox.det_confidence = scores[i];
    // build points
    for (int p = 0; p < numPoints; p += 3) {
      float *px = out + ((p + 5) * numAnchors);
      float *py = out + ((p + 6) * numAnchors);
      kBox.points.push_back(Point2f{px[i], py[i]});
    }
    if (m.count(kBox.bbox.class_id) == 0)
      m.emplace(kBox.bbox.class_id, KeypointsBoxes());
    m[kBox.bbox.class_id].push_back(kBox);
  }
}
} // namespace infer::vision