/**
 * @file visionRegister.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2024-07-26
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef __FLOWENGINE_INFER_VISION_REGISTER_HPP_
#define __FLOWENGINE_INFER_VISION_REGISTER_HPP_

namespace infer::vision {

class VisionRegistrar {
public:
  static VisionRegistrar &getInstance() {
    static VisionRegistrar instance;
    return instance;
  }

  VisionRegistrar(const VisionRegistrar &) = delete;
  VisionRegistrar &operator=(const VisionRegistrar &) = delete;

private:
  VisionRegistrar();
};

} // namespace infer::vision

#endif