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

#ifndef __FLOWCORE_ALGORITHM_BUS_H_
#define __FLOWCORE_ALGORITHM_BUS_H_

using common::AlgoConfig;
using common::AlgoRetType;
using common::AlgoSerial;
using common::InferParams;
using common::InferResult;

class AlgorithmBus {
public:
  AlgorithmBus() = default;
  AlgorithmBus(AlgoConfig const &);

  virtual ~AlgorithmBus() {}

  virtual bool registered(std::string const &, AlgoConfig const &) = 0;

  virtual bool unregistered(std::string const &) = 0;

  virtual bool infer(std::string const &, void *, InferParams const &,
                     InferResult &) = 0;

  virtual AlgoRetType getType(std::string const &) const noexcept = 0;

  virtual AlgoSerial getSerial(std::string const &) const noexcept = 0;

  virtual void getActiveAlgorithms(std::vector<std::string> &) = 0;
};
#endif