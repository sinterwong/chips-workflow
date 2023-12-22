/**
 * @file vision.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-11-07
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef __INFER_VISION_H_
#define __INFER_VISION_H_
#include "infer_common.hpp"
#include "inference.h"
#include "opencv2/imgcodecs.hpp"
#include "preprocess.hpp"
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>

namespace infer::vision {
using common::AlgoRet;

class Vision {
public:
  explicit Vision(const AlgoConfig &_param, ModelInfo const &_info)
      : mParams(_param), modelInfo(_info) {
    config = mParams.getBaseParams();
  }

  virtual ~Vision(){};

  //!
  //! \brief ProcessInput that the input is correct for infer
  //! * The processInput of common version
  virtual bool processInput(FrameInfo const &inputData, cv::Mat &output) const {

    char *data = reinterpret_cast<char *>(*inputData.data);
    int height = inputData.shape.at(1);
    int width = inputData.shape.at(0);

    switch (inputData.type) {
    case common::ColorType::RGB888:
    case common::ColorType::BGR888: {
      cv::Mat image{height, width, CV_8UC3, data};
      // cv::imwrite("temp_out.jpg", image);

      std::array<int, 2> shape = {config->inputShape.at(0),
                                  config->inputShape.at(1)};
      if (!utils::resizeInput(image, config->isScale, shape)) {
        return false;
      }
      // cv::imwrite("temp_out_resized.jpg", image);
      output.create(cv::Size(image.channels(), image.cols * image.rows),
                    CV_8UC1);
      // output = image.clone();
      utils::hwc_to_chw(output.data, image.data, image.channels(), image.rows,
                        image.cols);

      // cv::imwrite("temp_out.jpg", output);
      break;
    }
    case common::ColorType::NV12: {
      output = cv::Mat(height * 3 / 2, width, CV_8UC1, data);
      utils::NV12toRGB(output, output);
      std::array<int, 2> shape = {config->inputShape.at(0),
                                  config->inputShape.at(1)};
      if (!utils::resizeInput(output, config->isScale, shape)) {
        return false;
      }
      utils::RGB2NV12(output, output);
      break;
    }
    case common::ColorType::None: {
      return false;
    }
    }
    return true;
  }

  //!
  //! \brief ProcessInput that the input is correct for infer
  //!
  virtual bool processOutput(void **, InferResult &) const = 0;

  //!
  //! \brief verifyOutput that the result is correct for infer
  //!
  virtual bool verifyOutput(InferResult const &) const = 0;

protected:
  //!< The parameters for the sample.
  AlgoConfig mParams;

  //!< The basic config of the model
  AlgoBase *config;

  //!< The information of model.
  ModelInfo modelInfo;
};
} // namespace infer::vision
#endif
