#include "visionInfer.hpp"
#include <gflags/gflags.h>
#include <memory>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#if (TARGET_PLATFORM == 0)
#include "x3/x3_inference.hpp"
using namespace infer::x3;
#elif (TARGET_PLATFORM == 1)
#include "jetson/trt_inference.hpp"
using namespace infer::trt;
#endif
// using namespace infer::vision;

DEFINE_string(image_path, "", "Specify config path.");

using common::InferResult;

int main(int argc, char **argv) {

  gflags::SetUsageMessage("some usage message");
  gflags::SetVersionString("1.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  cv::Mat image_bgr = cv::imread(FLAGS_image_path);

  // auto process_pose = [&](common::PoseRet const &ret) {};
  // auto process_cls = [&](common::ClsRet const &ret) {};
  // auto process_det = [&](common::DetRet const &ret) {
  //   FLOWENGINE_LOGGER_INFO("number of result: {}", ret.size());
  //   for (auto &bbox : ret) {
  //     cv::Rect rect(bbox.bbox[0], bbox.bbox[1], bbox.bbox[2] - bbox.bbox[0],
  //                   bbox.bbox[3] - bbox.bbox[1]);
  //     cv::rectangle(image_bgr, rect, cv::Scalar(0, 0, 255), 2);
  //   }
  //   cv::cvtColor(image_bgr, image_bgr, cv::COLOR_RGB2BGR);
  //   cv::imwrite("test_det_out.jpg", image_bgr);
  // };

  // InferResult ret;
  // ret.aRet = common::DetRet

  // std::visit(process_det, ret.aRet);

  gflags::ShutDownCommandLineFlags();
  return 0;
}
