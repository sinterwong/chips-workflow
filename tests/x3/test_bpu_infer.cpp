/**
 * @file test_bpu_infer.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-01-16
 *
 * @copyright Copyright (c) 2023
 *
 */
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <sp_bpu.h>

#include "gflags/gflags.h"

#include "hb_dnn.h"
#include "preprocess.hpp"

DEFINE_string(model_path, "", "Specify the model path.");
DEFINE_string(image_path, "", "Specify the image path.");

int main(int argc, char **argv) {
  gflags::SetUsageMessage("some usage message");
  gflags::SetVersionString("1.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  cv::Mat image = cv::imread(FLAGS_image_path);
  cv::resize(image, image, cv::Size(640, 640));
  cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

  cv::Mat nv12_data;
  infer::utils::RGB2NV12(image, nv12_data);

  cv::imwrite("test_bpu_infer.jpg", nv12_data);

  bpu_module *engine = sp_init_bpu_module(FLAGS_model_path.c_str());

  hbDNNTensor output_tensor;

  sp_init_bpu_tensors(engine, &output_tensor);

  engine->output_tensor = &output_tensor;

  sp_bpu_start_predict(engine, reinterpret_cast<char *>(nv12_data.data));

  for (int i = 0; i < output_tensor.properties.alignedShape.numDimensions;
       i++) {
    std::cout << output_tensor.properties.alignedShape.dimensionSize[i] << ", ";
  }
  std::cout << std::endl;

  float *data = reinterpret_cast<float *>(output_tensor.sysMem[0].virAddr);

  for (int i = 4; i < 25200 * 6; i += 6) {
    if (data[i] > 0.3) {
      std::cout << data[i - 4] << ", " << data[i - 3] << ", " << data[i - 2]
                << ", " << data[i - 1] << ", " << data[i] << std::endl;
    }
  }

  std::cout << std::endl;
  sp_deinit_bpu_tensor(&output_tensor, 1);
  sp_release_bpu_module(engine);

  return 0;
}