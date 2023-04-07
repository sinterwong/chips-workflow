
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
#include "postprocess.hpp"
#include "vision.hpp"
#include <algorithm>
#include <array>
#include <unordered_map>
#include <vector>

namespace infer::vision {

bool Detection::processInput(cv::Mat const &input, void **output,
                             common::ColorType) const {
  // 后面可以根据需求，实现基于opencv的预处理，比如resize和图片类型转换（bgr->rgb,
  // bgr->nv12, nv12->bgr..)
  return true;
}

bool Detection::processOutput(void **output, InferResult &result) const {
  std::unordered_map<int, BBoxes> cls2bbox;
  generateBoxes(cls2bbox, output);
  auto detRet = BBoxes();
  utils::nms(detRet, cls2bbox, config.nmsThr);
  // rect 还原成原始大小
  utils::restoryBoxes(detRet, result.shape, config.inputShape, config.isScale);
  BBoxes::iterator it = detRet.begin();
  // 清除掉不符合要求的框
  for (; it != detRet.end();) {
    int area = (it->bbox[3] - it->bbox[1]) * (it->bbox[2] - it->bbox[0]);
    if (area < 2 * 2)
      it = detRet.erase(it);
    else
      // 迭代器指向下一个元素位置
      ++it;
  }
  result.aRet = std::move(detRet);
  return true;
}

bool Detection::verifyOutput(InferResult const &) const { return true; }
} // namespace infer::vision