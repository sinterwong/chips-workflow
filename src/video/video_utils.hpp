/**
 * @file video_utils.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-05-06
 *
 * @copyright Copyright (c) 2023
 *
 */
#include "common/common.hpp"
#include "logger/logger.hpp"
#include <atomic>
#include <chrono>
#include <experimental/filesystem>
#include <opencv2/videoio.hpp>
#include <thread>

#ifndef __FLOWENGINE_VIDEO_UTILS_H_
#define __FLOWENGINE_VIDEO_UTILS_H_

namespace video::utils {

/**
 * @brief h264 to mp4
 *
 * @param inputFile
 * @param outputFile
 */
inline void wrapH2642mp4(std::string const &h264File,
                         std::string const &mp4File) {
  if (!std::experimental::filesystem::exists(h264File)) {
    return;
  };
  FLOWENGINE_LOGGER_INFO("Wrap to mp4...");
  // cv::VideoCapture input_video(h264File);
  // cv::Size video_size =
  //     cv::Size((int)input_video.get(cv::CAP_PROP_FRAME_WIDTH),
  //              (int)input_video.get(cv::CAP_PROP_FRAME_HEIGHT));
  // double fps = input_video.get(cv::CAP_PROP_FPS);

  // cv::VideoWriter output_video(mp4File,
  //                              cv::VideoWriter::fourcc('H', '2', '6', '4'),
  //                              fps, video_size, true);

  // cv::Mat frame;
  // while (input_video.read(frame)) {
  //   output_video.write(frame);
  // }

  // input_video.release();
  // output_video.release();
  std::string cmd = "ffmpeg -i " + h264File + " -c:v copy " + mp4File;
  int ret = std::system(cmd.c_str());
  if (ret == -1) {
    perror("system");
    std::exit(EXIT_FAILURE);
  } else {
    if (WIFEXITED(ret)) {
      int status = WEXITSTATUS(ret);
      FLOWENGINE_LOGGER_INFO("Command exited with status {}", status);
    } else if (WIFSIGNALED(ret)) {
      int sig = WTERMSIG(ret);
      FLOWENGINE_LOGGER_INFO("Command was terminated by signal {}", sig);
    }
  }
}

/**
 * @brief Get the Codec object
 *
 * @param fourcc
 * @return std::string
 */
inline std::string getCodec(int fourcc) {
  char a[5];
  for (int i = 0; i < 4; i++) {
    a[i] = fourcc >> (i * 8) & 255;
  }
  a[4] = '\0';
  return std::string{a};
}

inline void readFrame(cv::VideoCapture &cap, cv::Mat &frame,
                      std::atomic<bool> &successFlag) {
  successFlag = cap.read(frame);
}

inline bool videoRecordWithFFmpeg(std::string const &url,
                                  std::string const &path, int videoDuration,
                                  int const TIMEOUT_MS = 3000) {
  // 构建ffmpeg命令
  std::string cmd =
      "ffmpeg -y -rtsp_transport tcp -analyzeduration 10M -probesize 10M -i " +
      url + " -c copy -t " + std::to_string(videoDuration) + " " + path;

  // 开启子进程执行FFmpeg命令
  int result = std::system(cmd.c_str());

  // 简单检查命令是否成功
  return result == 0;
}

inline bool videoRecordWithOpencv(std::string const &url,
                                  std::string const &path, int videoDuration,
                                  int const TIMEOUT_MS = 3000) {

  // 视频直接截取，不涉及编解码
  auto cap = cv::VideoCapture(url, cv::CAP_FFMPEG);

  // Check if the video file was loaded successfully
  if (!cap.isOpened()) {
    FLOWENGINE_LOGGER_ERROR("Could not open the video {}", url);
    return false;
  }

  int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
  int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
  int frameRate = cap.get(cv::CAP_PROP_FPS);
  // 总共需要录制的帧数
  int frameCount = videoDuration * frameRate;
  // 获取rtsp流的原始编解码信息
  int fourcc = static_cast<int>(cap.get(cv::CAP_PROP_FOURCC));

  FLOWENGINE_LOGGER_INFO(
      "The video message to be recorded: size {}x{}, rate {}, fourcc {}",
      height, width, frameRate, getCodec(fourcc));

  auto writer =
      cv::VideoWriter(path, fourcc, frameRate, cv::Size{width, height});

  if (!writer.isOpened()) {
    FLOWENGINE_LOGGER_ERROR("Could not initialize the video writer for {}",
                            path);
    cap.release();
    return false;
  }

  cv::Mat frame;
  std::atomic<bool> frameReadSuccessfully(false);
  while (frameCount > 0) {
    frameReadSuccessfully = false;
    std::thread readingThread(readFrame, std::ref(cap), std::ref(frame),
                              std::ref(frameReadSuccessfully));

    auto start = std::chrono::high_resolution_clock::now();
    while (!frameReadSuccessfully &&
           std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::high_resolution_clock::now() - start)
                   .count() < TIMEOUT_MS) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    if (frameReadSuccessfully) {
      readingThread.join();
      writer.write(frame);
      frameCount--;
    } else {
      // If we timed out, detach from the thread and handle the timeout
      readingThread.detach();
      FLOWENGINE_LOGGER_ERROR("Timeout while reading frame from {}", url);
      writer.release();
      cap.release();
      return false;
    }
  }
  writer.release();
  cap.release();
  return true;
}
} // namespace video::utils

#endif