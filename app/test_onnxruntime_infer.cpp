#include "preprocess.hpp"
#include <onnxruntime_cxx_api.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

int main(int argc, char **argv) {

  // input
  cv::Mat input_image_bgr = cv::imread(
      "/home/wangxt/workspace/projects/flowengine/tests/data/car.jpg");

  cv::Mat input_image_rgb;
  cv::cvtColor(input_image_bgr, input_image_rgb, cv::COLOR_BGR2RGB);

  cv::Mat input_ = input_image_rgb.clone();
  infer::utils::hwc_to_chw(input_.data, input_image_rgb.data,
                           input_image_rgb.channels(), input_image_rgb.rows,
                           input_image_rgb.cols);
  cv::imwrite("temp_out.jpg", input_);

  cv::resize(input_, input_, cv::Size(640, 640));
  input_.convertTo(input_, CV_32FC3, 1.0 / 255.0);

  int64_t input_size_ = 640 * 640 * 3;
  std::array<int64_t, 4> input_shape_{1, 3, 640, 640};

  // output
  std::array<int64_t, 3> output_shape_{1, 25200, 15};
  std::array<float, 1 * 25200 * 15> results_;

  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

  Ort::Value input_tensor_ = Ort::Value::CreateTensor<float>(
      memory_info, reinterpret_cast<float *>(input_.data), input_size_,
      input_shape_.data(), input_shape_.size());

  Ort::Value output_tensor_ = Ort::Value::CreateTensor<float>(
      memory_info, results_.data(), results_.size(), output_shape_.data(),
      output_shape_.size());

  Ort::Env env{ORT_LOGGING_LEVEL_VERBOSE, "test"};
  Ort::SessionOptions sessionOptions{nullptr};
  Ort::Session session_{env,
                        "/home/wangxt/workspace/projects/flowengine/tests/data/"
                        "models/plate_detect.onnx",
                        sessionOptions};

  // Get input and output names
  const char *input_names[] = {"input"};
  const char *output_names[] = {"output"};

  Ort::RunOptions run_options;
  session_.Run(run_options, input_names, &input_tensor_, 1, output_names,
               &output_tensor_, 1);

  // output
  auto output_tensor_data = output_tensor_.GetTensorMutableData<float>();

  // 打印部分结果，检查是否正确
  for (int i = 0; i < 30; i++) {
    std::cout << output_tensor_data[i] << std::endl;
  }

  return 0;
}