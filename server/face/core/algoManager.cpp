/**
 * @file algoManager.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-11-10
 *
 * @copyright Copyright (c) 2023
 *
 */
#include "algoManager.hpp"

namespace server::face::core {
AlgoManager *AlgoManager::instance = nullptr;

std::future<bool> AlgoManager::infer(FramePackage const &framePackage,
                                     std::vector<float> &feature) {
  auto task = std::packaged_task<bool()>([&] {
    // 使用算法资源进行推理
    FrameInfo frame;
    frame.data = reinterpret_cast<void **>(&framePackage.frame->data);
    frame.inputShape = {framePackage.frame->cols, framePackage.frame->rows,
                        framePackage.frame->channels()};

    // TODO:暂时写死NV12格式，这里应该有一个宏来确定是什么推理数据
    frame.shape = {framePackage.frame->cols, framePackage.frame->rows * 2 / 3,
                   framePackage.frame->channels()};
    frame.type = ColorType::NV12;

    // 等待获取可用算法资源
    face_algo_ptr algo = getAvailableAlgo();

    bool ret = algo->forward(frame, feature);
    // 推理完成后标记算法为可用
    releaseAlgo(algo);
    return ret;
  });

  std::future<bool> ret = task.get_future();

  // 在单独的线程上运行任务
  task();

  return ret; // 移交调用者等待算法结果
}

std::future<bool> AlgoManager::infer(std::string const &url,
                                     std::vector<float> &feature) {

  // 推理任务定义
  auto task = std::packaged_task<bool()>([&] {
    // 检查url的类型是本地路径还是http
    std::shared_ptr<cv::Mat> image_bgr = nullptr;
    // 解析url类型，根据不同类型的url，获取图片
    if (isHttpUri(url)) {
      image_bgr = getImageFromURL(url.c_str());
    } else if (isBase64(url)) {
      image_bgr = str2mat(url);
    } else {
      // 不是http或base64，那么就是本地路径
      if (std::filesystem::exists(url)) {
        image_bgr = std::make_shared<cv::Mat>(cv::imread(url));
      }
    }
    if (!image_bgr) {
      return false;
    }

    // TODO: 暂时写死NV12格式推理
    cv::Mat input;
    infer::utils::BGR2NV12(*image_bgr, input);
    FrameInfo frame;
    frame.data = reinterpret_cast<void **>(&input.data);
    frame.inputShape = {input.cols, input.rows, input.channels()};
    // 暂时写死NV12格式，这里应该有一个宏来确定是什么推理数据
    frame.shape = {input.cols, input.rows * 2 / 3, input.channels()};
    frame.type = ColorType::NV12;

    // 前处理完成后再等待获取可用算法资源，提高并发度
    face_algo_ptr algo = getAvailableAlgo();
    bool ret = algo->forward(frame, feature);

    // 推理完成后标记算法为可用
    releaseAlgo(algo);
    return ret;
  });

  std::future<bool> ret = task.get_future();

  // 在单独的线程上运行任务
  task();

  return ret; // 移交调用者等待算法结果
}

} // namespace server::face::core