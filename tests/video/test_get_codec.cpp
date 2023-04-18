#include <iostream>
#include <opencv2/videoio.hpp>
#include <string>

std::string getCodec(int fourcc) {
  char a[5];
  for (int i = 0; i < 4; i++)
    a[i] = fourcc >> (i * 8) & 255;
  a[4] = '\0';
  return std::string{a};
}

int main() {
  auto cap = cv::VideoCapture();

  cap.open(
      "rtsp://admin:zkfd123.com@114.242.23.39:9303/Streaming/Channels/101");
  int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
  int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
  int frameRate = cap.get(cv::CAP_PROP_FPS);
  int fourcc = static_cast<int>(cap.get(cv::CAP_PROP_FOURCC));

  std::cout << height << ", " << width << ", " << frameRate << std::endl;
  std::cout << fourcc << ", " << getCodec(fourcc) << std::endl;
  cap.release();
}