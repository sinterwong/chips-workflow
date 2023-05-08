#include "uri.hpp"
#include <algorithm>
#include <iostream>
#include <stdio.h>

#include "logger/logger.hpp"
#include "logging.h"
// #include <filesystem>
#include <experimental/filesystem>

using namespace std::experimental;

namespace video::utils {

// toLower
static std::string toLower(const std::string &str) {
  std::string dst;
  dst.resize(str.size());
  std::transform(str.begin(), str.end(), dst.begin(), ::tolower);
  return dst;
}

// constructor
URI::URI() { port = -1; }

// constructor
URI::URI(std::string const &uri) { Parse(uri); }

// Parse
bool URI::Parse(std::string const &uri) {
  if (uri.empty())
    return false;

  string = uri;
  protocol = "";
  extension = "";
  location = "";
  port = -1;

  // look for protocol
  std::size_t pos = string.find("://");

  if (pos != std::string::npos) {
    protocol = string.substr(0, pos);
    location = string.substr(pos + 3, std::string::npos);
  } else {
    // check for some formats without specified protocol
    pos = string.find("/dev/video");

    if (pos == 0) {
      protocol = "v4l2";
    } else if (string.find(".") != std::string::npos ||
               string.find("/") != std::string::npos ||
               filesystem::exists(string)) {
      protocol = "file";
    } else if (sscanf(string.c_str(), "%i", &port) == 1) {
      protocol = "csi";
    } else if (string == "display") {
      protocol = "display";
    } else {
      FLOWENGINE_LOGGER_ERROR("URI -- invalid resource or file path:  {}",
                              string);
      return false;
    }

    location = string;

    // reconstruct full URI string
    string = protocol + "://";

    if (protocol == "file")
      string += filesystem::absolute(location); // URI paths should be absolute
    else
      string += location;
  }

  // protocol should be all lowercase for easier parsing
  protocol = toLower(protocol);

  // parse extra info (device ordinals, IP addresses, ect)
  if (protocol == "v4l2") {
    if (sscanf(location.c_str(), "/dev/video%i", &port) != 1) {
      FLOWENGINE_LOGGER_ERROR("URI -- failed to parse V4L2 device ID from {}",
                              location);
      return false;
    }
  } else if (protocol == "csi") {
    if (sscanf(location.c_str(), "%i", &port) != 1) {
      FLOWENGINE_LOGGER_ERROR(
          "URI -- failed to parse MIPI CSI device ID from {}", location);
      return false;
    }
  } else if (protocol == "display") {
    if (sscanf(location.c_str(), "%i", &port) != 1) {
      FLOWENGINE_LOGGER_ERROR("URI -- using default display device 0");
      port = 0;
    }
  } else if (protocol == "file") {
    filesystem::path p(location);
    extension = p.extension();
  } else {
    // search for ip/port format
    std::string port_str;
    pos = location.find(":");

    if (pos != std::string::npos) // "xxx.xxx.xxx.xxx:port"
    {
      if (protocol == "rtsp") // "user:pass@ip:port"
      {
        const std::size_t host_pos = location.find("@", pos + 1);
        const std::size_t port_pos = location.find(":", pos + 1);

        if (host_pos != std::string::npos && port_pos != std::string::npos)
          pos = port_pos;
      }

      port_str = location.substr(pos + 1, std::string::npos);
      location = location.substr(0, pos);
    } else if (std::count(location.begin(), location.end(), '.') == 0) // "port"
    {
      port_str = location;
      location = "127.0.0.1";
    }

    // parse the port number
    if (port_str.size() != 0) {
      if (sscanf(port_str.c_str(), "%i", &port) != 1) {
        if (protocol == "rtsp") {
          FLOWENGINE_LOGGER_WARN(
              "URI -- missing/invalid IP port from {}, default to port 554",
              string);
          port = 554;
        } else {
          FLOWENGINE_LOGGER_WARN("URI -- failed to parse IP port from {}",
                                 string);
          return false;
        }
      }
    }

    // convert "@:port" format to localhost
    if (location == "@")
      location = "127.0.0.1";
  }

  return true;
}

// Print
void URI::Print(std::string const &prefix) const {
  FLOWENGINE_LOGGER_INFO("{}-- URI: {}", prefix, string);
  FLOWENGINE_LOGGER_INFO("{}   - protocol:  {}", prefix, protocol);
  FLOWENGINE_LOGGER_INFO("{}   - location:  {}", prefix, location);

  if (extension.size() > 0)
    FLOWENGINE_LOGGER_INFO("{}   - extension: {}", prefix, extension);

  if (port > 0)
    FLOWENGINE_LOGGER_INFO("{}   - port:      {}", prefix, port);
}
} // namespace module::utils
