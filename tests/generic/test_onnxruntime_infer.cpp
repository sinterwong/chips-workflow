#include "preprocess.hpp"
#include <onnxruntime_cxx_api.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

int main(int argc, char **argv) {

  Ort::Env env{ORT_LOGGING_LEVEL_VERBOSE, "test"};
  Ort::SessionOptions sessionOptions{nullptr};
  Ort::Session session_{env,
                        "/home/wangxt/workspace/projects/flowengine/tests/data/"
                        "models/plate_detect.onnx",
                        sessionOptions};

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

  // 获取输入信息
  assert(session_.GetInputCount() == 1);
  auto inputInfo = session_.GetInputTypeInfo(0);
  assert(inputInfo.GetTensorTypeAndShapeInfo().GetDimensionsCount() == 4);
  auto input_shape_ = inputInfo.GetTensorTypeAndShapeInfo().GetShape();

  // output
  auto outputInfo = session_.GetOutputTypeInfo(0);
  auto output_shape_ = outputInfo.GetTensorTypeAndShapeInfo().GetShape();
  // std::array<float, 1 * 25200 * 15> results_;
  float *results_;
  int64_t output_size_ = 1;
  for (auto &dim : output_shape_) {
    output_size_ *= dim;
  }
  results_ = new float[output_size_];

  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

  Ort::Value input_tensor_ = Ort::Value::CreateTensor<float>(
      memory_info, reinterpret_cast<float *>(input_.data), input_size_,
      input_shape_.data(), input_shape_.size());

  Ort::Value output_tensor_ = Ort::Value::CreateTensor<float>(
      memory_info, results_, output_size_, output_shape_.data(),
      output_shape_.size());

  // Get input and output names
  const char *input_names[1];
  const char *output_names[1];
  std::string i_name = "input";
  std::string o_name = "output";

  input_names[0] = i_name.c_str();
  output_names[0] = o_name.c_str();

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