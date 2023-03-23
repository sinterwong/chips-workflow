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
#include "lprDet.hpp"
#include "logger/logger.hpp"
#include "postprocess.hpp"

namespace infer {
namespace vision {

bool LPRDet::processOutput(void **output, InferResult &result) const {
  std::unordered_map<int, KeypointsBoxes> cls2bbox;
  generateKeypointsBoxes(cls2bbox, output);
  auto rets = KeypointsBoxes();
  utils::nms_kbox(rets, cls2bbox, config->nms_thr);
  // rect 还原成原始大小
  utils::restoryKeypointsBoxes(rets, result.shape, config->inputShape, config->isScale);
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

void LPRDet::generateBoxes(std::unordered_map<int, BBoxes> &m,
                           void **outputs) const {
  float **output = reinterpret_cast<float **>(*outputs);
  int numAnchors = modelInfo.outputShapes[0].at(1);
  int num = modelInfo.outputShapes[0].at(2);
  for (int j = 0; j < numAnchors * num; j += num) {
    if (output[0][j + 4] <= config->cond_thr)
      continue;
    // std::cout << output[0][j + 0] << ", " << output[0][j + 1] << ", "
    //           << output[0][j + 2] << ", " << output[0][j + 3] << ", "
    //           << output[0][j + 4] << ", " << output[0][j + 5] << std::endl;
    BBox det;
    det.class_id = std::distance(
        output[0] + j + 13, // 4框 + 1置信度 + 8角点, 再往后是类别
        std::max_element(output[0] + j + 13, output[0] + j + num));
    int real_idx = j + 13 + det.class_id;
    det.det_confidence = output[0][real_idx];
    memcpy(&det, &output[0][j], 5 * sizeof(float));
    if (m.count(det.class_id) == 0)
      m.emplace(det.class_id, BBoxes());
    m[det.class_id].push_back(det);
  }
}

void LPRDet::generateKeypointsBoxes(std::unordered_map<int, KeypointsBoxes> &m,
                                    void **outputs) const {
  float **output = reinterpret_cast<float **>(*outputs);
  int numAnchors = modelInfo.outputShapes[0].at(1);
  int num = modelInfo.outputShapes[0].at(2); // num = 4 + 1 + 8 + 2 = 15
  for (int i = 0; i < numAnchors * num; i += num) {
    if (output[0][i + 4] <= config->cond_thr)
      continue;
    KeypointsBox kBox;
    for (int p = 5; p < 13; p += 2) {
      kBox.points.push_back(Point2f{output[0][i + p], output[0][i + p + 1]});
    }
    kBox.bbox.class_id = std::distance(
        output[0] + i + 13, // 4框 + 1置信度 + 8角点, 再往后是类别
        std::max_element(output[0] + i + 13, output[0] + i + num));
    int real_idx = i + 13 + kBox.bbox.class_id;
    kBox.bbox.det_confidence = output[0][real_idx];
    memcpy(&kBox.bbox, &output[0][i], 5 * sizeof(float));
    if (m.count(kBox.bbox.class_id) == 0)
      m.emplace(kBox.bbox.class_id, KeypointsBoxes());
    m[kBox.bbox.class_id].push_back(kBox);
  }
}

FlowEngineModuleRegister(LPRDet, AlgoConfig const &, ModelInfo const &);

} // namespace vision
} // namespace infer