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
#include "xCamera.hpp"
#include "xDecoder.hpp"
#include <cstddef>

namespace video {
std::unique_ptr<videoSource> videoSource::Create(videoOptions const &options) {
  std::unique_ptr<videoSource> src;
  URI const &uri = options.resource;
  if (uri.protocol == "file" || uri.protocol == "rtsp") {
    src = XDecoder::Create(options);
  } else if (uri.protocol == "csi" || uri.protocol == "v4l2") {
    src = XCamera::Create(options);
  } else {
    FLOWENGINE_LOGGER_ERROR("videoSource -- unsupported protocol {}",
                            uri.protocol.size() > 0 ? uri.protocol : "null");
  }
  FLOWENGINE_LOGGER_INFO("XDecoder is created {} from {}", src->typeTostr(),
                         src->GetResource().string);
  return src;
}

std::unique_ptr<videoSource> videoSource::Create() {
  std::unique_ptr<videoSource> src = XDecoder::Create();
  FLOWENGINE_LOGGER_INFO("XDecoder is created");
  return src;
}

std::string const videoSource::typeTostr(size_t type) {
  if (type == XCamera::Type)
    return "XCamera";
  else if (type == XDecoder::Type)
    return "XDecoder";
  return "(unknown)";
}

} // namespace video
