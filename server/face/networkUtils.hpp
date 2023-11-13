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

static const std::string base64_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                        "abcdefghijklmnopqrstuvwxyz"
                                        "0123456789+/";
inline void base64_encode(uchar const *bytes_to_encode, unsigned int in_len,
                          std::string &ret) {
  int i = 0;
  int j = 0;
  unsigned char char_array_3[3];
  unsigned char char_array_4[4];

  while (in_len--) {
    char_array_3[i++] = *(bytes_to_encode++);
    if (i == 3) {
      char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
      char_array_4[1] =
          ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
      char_array_4[2] =
          ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
      char_array_4[3] = char_array_3[2] & 0x3f;

      for (i = 0; (i < 4); i++) {
        ret += base64_chars[char_array_4[i]];
      }
      i = 0;
    }
  }

  if (i) {
    for (j = i; j < 3; j++) {
      char_array_3[j] = '\0';
    }

    char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
    char_array_4[1] =
        ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
    char_array_4[2] =
        ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
    char_array_4[3] = char_array_3[2] & 0x3f;

    for (j = 0; (j < i + 1); j++) {
      ret += base64_chars[char_array_4[j]];
    }

    while ((i++ < 3)) {
      ret += '=';
    }
  }
}

constexpr static bool is_base64(unsigned char c) {
  return (isalnum(c) || (c == '+') || (c == '/'));
}

inline void base64_decode(std::string const &encoded_string, std::string &ret) {
  int in_len = encoded_string.size();
  int i = 0;
  int j = 0;
  int in_ = 0;
  unsigned char char_array_4[4], char_array_3[3];

  while (in_len-- && (encoded_string[in_] != '=') &&
         is_base64(encoded_string[in_])) {
    char_array_4[i++] = encoded_string[in_];
    in_++;

    if (i == 4) {
      for (i = 0; i < 4; i++) {
        char_array_4[i] = base64_chars.find(char_array_4[i]);
      }

      char_array_3[0] =
          (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
      char_array_3[1] =
          ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
      char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

      for (i = 0; (i < 3); i++) {
        ret += char_array_3[i];
      }

      i = 0;
    }
  }

  if (i) {
    for (j = i; j < 4; j++) {
      char_array_4[j] = 0;
    }

    for (j = 0; j < 4; j++) {
      char_array_4[j] = base64_chars.find(char_array_4[j]);
    }

    char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
    char_array_3[1] =
        ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
    char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

    for (j = 0; (j < i - 1); j++) {
      ret += char_array_3[j];
    }
  }
}

inline std::string mat2str(cv::Mat const &m) {
  int params[3] = {0};
  params[0] = cv::IMWRITE_JPEG_QUALITY;
  params[1] = 100;

  std::vector<uchar> buf;
  cv::imencode(".jpg", m, buf, std::vector<int>(params, params + 2));
  uchar *result = reinterpret_cast<uchar *>(&buf[0]);
  std::string ret;
  base64_encode(result, buf.size(), ret);
  return ret;
}

inline std::shared_ptr<cv::Mat> str2mat(const std::string &s) {
  // Decode data
  std::string decoded_string;
  base64_decode(s, decoded_string);
  std::vector<uchar> data(decoded_string.begin(), decoded_string.end());

  cv::Mat img = imdecode(data, cv::IMREAD_UNCHANGED);
  if (img.empty()) {
    return nullptr;
  }
  return std::make_shared<cv::Mat>(img);
}

#endif
