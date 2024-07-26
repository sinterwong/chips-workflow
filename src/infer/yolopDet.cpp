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
#include "yolopDet.hpp"
#include "logger/logger.hpp"
#include "postprocess.hpp"

namespace infer::vision {

bool YoloPDet::processOutput(void **output, InferResult &result) const {
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

void YoloPDet::generateKeypointsBoxes(
    std::unordered_map<int, KeypointsBoxes> &m, void **outputs) const {
  float **output = reinterpret_cast<float **>(outputs);
  float *out = output[0]; // just one output

  int numAnchors = modelInfo.outputShapes[0].at(1);
  // num = 4 + 1 + (numPoints * 2) + numClass = 5 + (numPoints * 2) + numClass
  int num = modelInfo.outputShapes[0].at(2);
  int pointsEndIndex = 5 + config.numPoints * 2;
  for (int i = 0; i < numAnchors * num; i += num) {
    if (out[i + 4] <= config.cond_thr)
      continue;
    KeypointsBox kBox;
    for (int p = 5; p < pointsEndIndex; p += 2) {
      kBox.points.push_back(Point2f{out[i + p], out[i + p + 1]});
    }
    kBox.bbox.class_id = std::distance(
        out + i + pointsEndIndex, // 4框 + 1置信度 + n个角点, 再往后是类别
        std::max_element(out + i + pointsEndIndex, out + i + num));
    int real_idx = i + pointsEndIndex + kBox.bbox.class_id;
    kBox.bbox.det_confidence = out[real_idx];
    memcpy(&kBox.bbox, &out[i], 5 * sizeof(float));
    if (m.count(kBox.bbox.class_id) == 0)
      m.emplace(kBox.bbox.class_id, KeypointsBoxes());
    m[kBox.bbox.class_id].push_back(kBox);
  }
}
} // namespace infer::vision