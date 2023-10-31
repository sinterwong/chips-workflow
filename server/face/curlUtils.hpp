/**
 * @file curlUtils.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-10-26
 *
 * @copyright Copyright (c) 2023
 *
 */
#ifndef __SERVER_MACRO_UTILS_HPP_
#define __SERVER_MACRO_UTILS_HPP_

#include <string>
size_t curl_callback(void *ptr, size_t size, size_t nmemb,
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

#endif