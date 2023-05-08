#include "gflags/gflags.h"
#include <iostream>
#include <opencv2/videoio.hpp>
#include <string>

DEFINE_string(uri, "", "Specify the url of video.");

std::string getCodec(int fourcc) {
  char a[5];
  for (int i = 0; i < 4; i++)
    a[i] = fourcc >> (i * 8) & 255;
  a[4] = '\0';
  return std::string{a};
}

int main(int argc, char **argv) {

  gflags::SetUsageMessage("some usage message");
  gflags::SetVersionString("1.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  auto cap = cv::VideoCapture();

  cap.open(FLAGS_uri);
  int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
  int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
  int frameRate = cap.get(cv::CAP_PROP_FPS);
  int fourcc = static_cast<int>(cap.get(cv::CAP_PROP_FOURCC));

  std::cout << height << ", " << width << ", " << frameRate << std::endl;
  std::cout << fourcc << ", " << getCodec(fourcc) << std::endl;
  cap.release();

  gflags::ShutDownCommandLineFlags();
}