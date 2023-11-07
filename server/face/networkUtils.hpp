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

#include <curl/curl.h>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <string>

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

// 检查给定的URI是否为文件系统路径
inline bool isFileSystemPath(const std::string &uri) {
  // 这里我们假设如果不是HTTP/HTTPS，那就是文件系统路径
  return !isHttpUri(uri);
}

#endif
