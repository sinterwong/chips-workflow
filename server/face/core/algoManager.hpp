/**
 * @file algoManager.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief 算法负载均衡管理，用来调度人脸逻辑。全局单例
 * @version 0.1
 * @date 2023-10-24
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "algoDispatcher.hpp"
#include "logger/logger.hpp"
#include <future>
#include <unordered_map>

#ifndef __SERVER_FACE_CORE_ALGO_MANAGER_HPP_
#define __SERVER_FACE_CORE_ALGO_MANAGER_HPP_

namespace server::face::core {

constexpr int DET_ALGO_NUM = 2;
constexpr int REC_ALGO_NUM = 1;
constexpr int QUALITY_ALGO_NUM = 1;
constexpr int KEY_POINTS_ALGO_NUM = 1;

inline std::future<bool> make_false_future() {
  std::promise<bool> prom;
  auto fut = prom.get_future();
  prom.set_value(false); // Set the value to false.
  return fut;
}

struct FramePackage {
  std::string cameraName;
  std::shared_ptr<cv::Mat> frame;
};

class AlgoManager {
public:
  static AlgoManager &getInstance() {
    static std::once_flag onceFlag;
    std::call_once(onceFlag, [] { instance = new AlgoManager(); });
    return *instance;
  }
  AlgoManager(AlgoManager const &) = delete;
  AlgoManager &operator=(AlgoManager const &) = delete;

public:
  std::future<bool> infer(AlgoType const &, FrameInfo &, InferResult &);

private:
  AlgoManager() {
    // 初始化算法调度器
    algoDispatchers[AlgoType::DET] =
        std::make_shared<AlgoDispatcher>(AlgoType::DET, DET_ALGO_NUM);
    algoDispatchers[AlgoType::REC] =
        std::make_shared<AlgoDispatcher>(AlgoType::REC, REC_ALGO_NUM);
    algoDispatchers[AlgoType::QUALITY] =
        std::make_shared<AlgoDispatcher>(AlgoType::QUALITY, QUALITY_ALGO_NUM);
    algoDispatchers[AlgoType::KEYPOINT] = std::make_shared<AlgoDispatcher>(
        AlgoType::KEYPOINT, KEY_POINTS_ALGO_NUM);
  }
  ~AlgoManager() {
    delete instance;
    instance = nullptr;
  }
  static AlgoManager *instance;

private:
  // 管理AlgoDispatcher
  std::unordered_map<AlgoType, std::shared_ptr<AlgoDispatcher>> algoDispatchers;
};
} // namespace server::face::core
#endif