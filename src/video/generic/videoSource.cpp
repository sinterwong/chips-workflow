/**
 * @file videoSource.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-01-04
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "videoSource.hpp"

namespace video {
std::unique_ptr<videoSource> videoSource::Create(videoOptions const &options) {
  std::unique_ptr<videoSource> src;
  return src;
}

std::unique_ptr<videoSource> videoSource::Create() {
  std::unique_ptr<videoSource> src;
  return src;
}

std::string const videoSource::typeTostr(size_t type) { return ""; }

} // namespace video
