
/**
 * @file detection.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-11-04
 *
 * @copyright Copyright (c) 2022
 *
 */
#include "detection.hpp"
#include "infer_utils.hpp"
#include <algorithm>
#include <array>
#include <unordered_map>
#include <vector>

namespace infer {
namespace vision {

bool Detection::processInput(cv::Mat const &input, void **output, common::ColorType, common::ColorType) const {
  // 后面可以根据需求，实现基于opencv的预处理，比如resize和图片类型转换（bgr->rgb,
  // bgr->nv12, nv12->bgr..)
  return true;
}

bool Detection::processOutput(void *output, Result &result) const {
  std::unordered_map<int, std::vector<DetectionResult>> cls2bbox;
  generateBoxes(cls2bbox, output);
  utils::nms(result.detResults, cls2bbox, mParams.nms_thr);
  // rect 还原成原始大小
  utils::renderOriginShape(result.detResults, result.shape, mParams.inputShape,
                           mParams.isScale);
  std::vector<DetectionResult>::iterator it = result.detResults.begin();
  // 清除掉不符合要求的框
  for (; it != result.detResults.end();) {
    int area = (it->bbox[3] - it->bbox[1]) * (it->bbox[2] - it->bbox[0]);
    if (area < 2 * 2)
      it = result.detResults.erase(it);
    else
      // 迭代器指向下一个元素位置
      ++it;
  }
  return true;
}

bool Detection::verifyOutput(Result const &) const {
  return true;
}
} // namespace vision
} // namespace infer