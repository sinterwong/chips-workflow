/**
 * @file algo_header.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-02-28
 *
 * @copyright Copyright (c) 2023
 *
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#ifndef _FLOWENGINE_COMMON_ALGO_HEADER_HPP_
#define _FLOWENGINE_COMMON_ALGO_HEADER_HPP_

namespace common {

/**
 * @brief 算法类型
 *
 */
enum class AlgoType {
  Classifier = 0,
  Detecton,
  Segmentation,
  KeyPoint,
  Feature
};

/**
 * @brief 目前已经支持的算法种类
 *
 */
enum class SupportedAlgo : uint32_t {
  CocoDet = 0,
  HandDet,
  HeadDet,
  FireDet,
  SmogDet,
  SmokeCallCls,
  ExtinguisherCls,
  OiltubeCls,
  EarthlineCls,
};

} // namespace common
#endif // _FLOWENGINE_COMMON_CONFIG_HPP_