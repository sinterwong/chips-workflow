/**
 * @file networkUtils.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-11-06
 *
 * @copyright Copyright (c) 2023
 *
 */
#ifndef __SERVER_NETWORK_UTILS_HPP_
#define __SERVER_NETWORK_UTILS_HPP_

#include "common/myBase64.hpp"
#include <curl/curl.h>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <string>
// #include <filesystem>
#include <experimental/filesystem>

using namespace std::experimental;

inline size_t curl_callback(void *ptr, size_t size, size_t nmemb,
                            std::string *data) {
  data->append((char *)ptr, size * nmemb);
  return size * nmemb;
}

#define MY_CURL_POST(url, add_member_code)                                     \
  do {                                                                         \
    CURL *curl = curl_easy_init();                                             \
    struct curl_slist *headers = NULL;                                         \
    headers = curl_slist_append(                                               \
        headers, "Content-Type:application/json;charset=UTF-8");               \
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);                       \
    curl_easy_setopt(curl, CURLOPT_POST, 1);                                   \
    curl_easy_setopt(curl, CURLOPT_URL, (url).c_str());                        \
                                                                               \
    nlohmann::json info;                                                       \
    add_member_code                                                            \
                                                                               \
        std::string s_out3 = info.dump();                                      \
    std::string res;                                                           \
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, s_out3.c_str());                \
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, s_out3.length());            \
                                                                               \
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_callback);              \
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &(res));                         \
                                                                               \
    CURLcode code = curl_easy_perform(curl);                                   \
    if (code) {                                                                \
      FLOWENGINE_LOGGER_ERROR("Post request was failed {}!", code);            \
      FLOWENGINE_LOGGER_INFO("Post result is {}", res);                        \
    }                                                                          \
    curl_easy_cleanup(curl);                                                   \
  } while (0)

enum class ImageInputType { HTTP = 0, FILEPATH, Base64 };

// 使用cURL从HTTP获取图片
inline std::shared_ptr<cv::Mat> getImageFromURL(const char *url) {
  CURL *curl;
  CURLcode res;
  std::string response_string;
  std::string header_string;

  curl = curl_easy_init();
  if (curl) {
    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_string);
    curl_easy_setopt(curl, CURLOPT_HEADERDATA, &header_string);

    // 执行HTTP请求
    res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
      fprintf(stderr, "curl_easy_perform() failed: %s\n",
              curl_easy_strerror(res));
      return nullptr;
    }
    // 清理
    curl_easy_cleanup(curl);
  }

  std::vector<uchar> data(response_string.begin(), response_string.end());
  cv::Mat image = cv::imdecode(data, cv::IMREAD_COLOR);
  return std::make_shared<cv::Mat>(image);
}

// 辅助函数来检查字符串是否以指定前缀开始
inline bool startsWith(const std::string &str, const std::string &prefix) {
  if (str.length() < prefix.length()) {
    return false;
  }
  return std::equal(prefix.begin(), prefix.end(), str.begin());
}

// 检查给定的URI是否为HTTP或HTTPS
inline bool isHttpUri(const std::string &uri) {
  // 转换为小写，以便不区分大小写地比较
  std::string lowerCaseUri = uri;
  std::transform(lowerCaseUri.begin(), lowerCaseUri.end(), lowerCaseUri.begin(),
                 ::tolower);

  return startsWith(lowerCaseUri, "http://") ||
         startsWith(lowerCaseUri, "https://");
}

inline bool isBase64(const std::string &uri) {
  // 转换为小写，以便不区分大小写地比较
  std::string lowerCaseUri = uri;
  std::transform(lowerCaseUri.begin(), lowerCaseUri.end(), lowerCaseUri.begin(),
                 ::tolower);

  return startsWith(lowerCaseUri, "data:image/jpeg;base64,");
}

inline ImageInputType getImageInputType(const std::string &uri) {
  if (isHttpUri(uri)) {
    return ImageInputType::HTTP;
  } else if (isBase64(uri)) {
    return ImageInputType::Base64;
  } else {
    return ImageInputType::FILEPATH;
  }
}

constexpr static bool is_base64(unsigned char c) {
  return (isalnum(c) || (c == '+') || (c == '/'));
}

inline std::string mat2str(cv::Mat const &m) {
  int params[3] = {0};
  params[0] = cv::IMWRITE_JPEG_QUALITY;
  params[1] = 100;

  std::vector<uchar> buf;
  cv::imencode(".jpg", m, buf, std::vector<int>(params, params + 2));
  uchar *result = reinterpret_cast<uchar *>(&buf[0]);
  std::vector<uint8_t> data(result, result + buf.size());
  std::string ret = flowengine::core::Base64::encode(data);
  return ret;
}

inline std::shared_ptr<cv::Mat> str2mat(const std::string &s) {
  // Decode data
  auto data = flowengine::core::Base64::decode(s);
  cv::Mat img = imdecode(data, cv::IMREAD_UNCHANGED);
  if (img.empty()) {
    return nullptr;
  }
  return std::make_shared<cv::Mat>(img);
}

inline std::shared_ptr<cv::Mat> getImageByUri(const std::string &uri) {
  // 检查url的类型是本地路径还是http
  std::shared_ptr<cv::Mat> image_bgr = nullptr;
  // 解析url类型，根据不同类型的url，获取图片
  if (isHttpUri(uri)) {
    image_bgr = getImageFromURL(uri.c_str());
  } else if (isBase64(uri)) {
    image_bgr = str2mat(uri);
  } else {
    // 不是http或base64，那么就是本地路径
    if (filesystem::exists(uri)) {
      image_bgr = std::make_shared<cv::Mat>(cv::imread(uri));
    }
  }
  return image_bgr;
}

#endif
