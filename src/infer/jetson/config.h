/**
 * @file config.h
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2022-05-30
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef __FLOW_DEMO_COMMON_H_
#define __FLOW_DEMO_COMMON_H_
#include <array>
#include <iostream>
#include <string>
#include <unordered_map>

enum WORKER_TYPE {
  PERSON_DETECTION = 0,
  HAND_POSE,
};

struct fe_config {
  unsigned short id;
  std::string device_type;
  WORKER_TYPE worker_type;
  std::string uri;
  float threshold;
  std::array<int, 4> region;
};

#endif