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

#include "x3/videoSource.hpp"
#include "x3/xCamera.hpp"
#include "x3/xDecoder.hpp"
#include <cstddef>

namespace module::utils {
std::unique_ptr<videoSource> videoSource::create(videoOptions const &options) {
  std::unique_ptr<videoSource> src;
  URI const &uri = options.resource;
  if (uri.protocol == "file" || uri.protocol == "rtsp") {
    src = XDecoder::create(options);
  } else if (uri.protocol == "csi" || uri.protocol == "v4l2") {
    src = XCamera::create(options);
  } else {
    FLOWENGINE_LOGGER_ERROR("videoSource -- unsupported protocol {}",
                            uri.protocol.size() > 0 ? uri.protocol : "null");
  }
  FLOWENGINE_LOGGER_INFO("create {} from {}", src->typeTostr(),
                         src->getResource().string);
  return src;
}

std::string const videoSource::typeTostr(size_t type) {
  if (type == XCamera::Type)
    return "gstCamera";
  else if (type == XDecoder::Type)
    return "gstDecoder";
  return "(unknown)";
}

} // namespace module::utils
