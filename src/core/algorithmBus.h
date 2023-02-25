/**
 * @file algorithmBus.h
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-02-17
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "common/common.hpp"
#include "common/config.hpp"

#ifndef __FLOWCORE_ALGORITHM_BUS_H_
#define __FLOWCORE_ALGORITHM_BUS_H_

using common::AlgorithmConfig;
using common::InferParams;
using common::InferResult;

class AlgorithmBus {
public:
  AlgorithmBus() = default;
  AlgorithmBus(AlgorithmConfig const &);

  virtual ~AlgorithmBus() {}

  virtual bool registered(std::string const &, AlgorithmConfig const &) = 0;

  virtual bool unregistered(std::string const &) = 0;

  virtual bool infer(std::string const &, void *, InferParams const &,
                     InferResult &) = 0;
};
#endif