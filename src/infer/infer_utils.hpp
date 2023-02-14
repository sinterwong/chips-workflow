/**
 * @file infer_utils.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-11-04
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef __INFERENCE_UTILS_H_
#define __INFERENCE_UTILS_H_
#include "infer_common.hpp"
#include "opencv2/imgproc.hpp"
#include <array>
#include <chrono>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <vector>

namespace infer {
namespace utils {

template <typename F, typename... Args>
long long measureTime(F func, Args&&... args) {
  auto start = std::chrono::high_resolution_clock::now();
  func(std::forward<Args>(args)...);
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  return duration.count();
}
} // namespace utils
} // namespace infer
#endif
