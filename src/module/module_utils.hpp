/**
 * @file module_utils.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-11-21
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <random>
#include <sstream>

#include <string>
#include <vector>

namespace module {
namespace utils {

inline unsigned int random_char() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 255);
  return dis(gen);
}

inline std::string generate_hex(const unsigned int len) {
  std::stringstream ss;
  for (auto i = 0; i < static_cast<int>(len); i++) {
    const auto rc = random_char();
    std::stringstream hexstream;
    hexstream << std::hex << rc;
    auto hex = hexstream.str();
    ss << (hex.length() < 2 ? '0' + hex : hex);
  }
  return ss.str();
}

cv::Mat str2mat(const std::string &imageBase64);

/**
 * Método que converte uma cv::Mat numa imagem em base64
 * @param img, imagem em cv::Mat
 * @return imagem em base64
 */
std::string mat2str(cv::Mat const &img);

std::string base64_encode(uchar const *bytesToEncode, unsigned int inLen);

std::string base64_decode(std::string const &encodedString);

static const std::string base64_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                        "abcdefghijklmnopqrstuvwxyz"
                                        "0123456789+/";

constexpr static inline bool is_base64(unsigned char c) {
  return (isalnum(c) || (c == '+') || (c == '/'));
}
} // namespace utils
} // namespace module