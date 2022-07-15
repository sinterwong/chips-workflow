/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef TENSORRT_ARGS_PARSER_H
#define TENSORRT_ARGS_PARSER_H

#include <string>
#include <vector>
#include <vector_types.h>
#ifdef _MSC_VER
#include "..\common\windows\getopt.h"
#else
#include <getopt.h>
#endif
#include <iostream>

namespace infer {
namespace trt {

//!
//! \brief The Params structure groups the basic parameters required
//!
struct BuildParams {
  int32_t batchSize{1}; //!< Number of inputs in a batch
  bool int8{false};     //!< Allow runnning the network in Int8 mode.
  bool fp16{false};     //!< Allow running the network in FP16 mode.
  std::vector<std::string> inputTensorNames;
  std::vector<std::string> outputTensorNames;
  int32_t dlaCore{-1}; //!< Specify the DLA core to run network on.
  std::vector<std::string>
      dataDirs; //!< Directory paths where sample data files are stored
};

//!
//! \brief The CaffeParams structure groups the additional parameters required
//! by
//!         networks that use caffe
//!
struct CaffeBuildParams : public BuildParams {
  std::string
      prototxtFileName; //!< Filename of prototxt design file of a network
  std::string
      weightsFileName;      //!< Filename of trained weights file of a network
  std::string meanFileName; //!< Filename of mean file of a network
};

//!
//! \brief The OnnxParams structure groups the additional parameters required by
//!         networks that use ONNX
//!
struct OnnxBuildParams : public BuildParams {
  std::string onnxFileName; //!< Filename of ONNX file of a network
};

//!
//! \brief The UffParams structure groups the additional parameters required by
//!         networks that use Uff
//!
struct UffBuildParams : public BuildParams {
  std::string uffFileName; //!< Filename of uff file of a network
};

} // namespace trt
} // namespace infer
#endif // TENSORRT_ARGS_PARSER_H
