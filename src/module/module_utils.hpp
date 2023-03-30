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

#include "common/common.hpp"

#ifndef __FLOWENGINE_MODULE_UTILS_H_
#define __FLOWENGINE_MODULE_UTILS_H_

using common::RetBox;
using common::RetPoly;

namespace module::utils {

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

bool drawRetBox(cv::Mat &image, RetBox const &bbox,
                cv::Scalar const &scalar = {0, 0, 255});

bool drawRetPoly(cv::Mat &image, RetPoly const &poly,
                 cv::Scalar const &scalar = {0, 0, 255});

/**
 * @brief h264 to mp4
 *
 * @param inputFile
 * @param outputFile
 */
void wrapH2642mp4(std::string const &h264File, std::string const &mp4File);

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

bool readFile(std::string const &filename, std::string &ret);

bool writeJson(std::string const &config, std::string const &outPath);

bool retBoxes2json(std::vector<RetBox> const &retBoxes, std::string &result);

bool retPolys2json(std::vector<RetPoly> const &retPolys, std::string &result);

bool retChars2json(std::vector<RetPoly> const &retPolys, std::string &result);

} // namespace module::utils

#endif