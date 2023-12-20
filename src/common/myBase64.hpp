/**
 * @file base64.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-11-14
 *
 * @copyright Copyright (c) 2023
 *
 */
#ifndef __FLOWENGINE_CORE_BASE64_HPP_
#define __FLOWENGINE_CORE_BASE64_HPP_

#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
namespace flowengine::core {
class Base64 {
public:
  static std::string encode(const std::vector<uint8_t> &input) {
    std::string encoded;
    int val = 0, valb = -6;
    for (uint8_t c : input) {
      val = (val << 8) + c;
      valb += 8;
      while (valb >= 0) {
        encoded.push_back(encodeTable[(val >> valb) & 0x3F]);
        valb -= 6;
      }
    }
    if (valb > -6)
      encoded.push_back(encodeTable[((val << 8) >> (valb + 8)) & 0x3F]);
    while (encoded.size() % 4)
      encoded.push_back('=');
    return encoded;
  }

  static std::vector<uint8_t> decode(const std::string &input) {
    std::string data = removePrefix(input, "data:image/jpeg;base64,");

    size_t padding = 0;
    if (!data.empty()) {
      if (data[data.size() - 1] == '=')
        padding++;
      if (data[data.size() - 2] == '=')
        padding++;
    }

    std::vector<uint8_t> decoded;
    decoded.reserve((data.size() / 4) * 3 - padding);

    int val = 0;
    int valb = -8;
    for (char c : data) {
      if (c == '=')
        break;
      if (decodeTable.find(c) == decodeTable.end()) {
        throw std::invalid_argument("Invalid character in Base64 string.");
      }
      val = (val << 6) + decodeTable.at(c);
      valb += 6;
      if (valb >= 0) {
        decoded.push_back(uint8_t((val >> valb) & 0xFF));
        valb -= 8;
      }
    }

    return decoded;
  }

private:
  static std::string removePrefix(const std::string &input,
                                  const std::string &prefix) {
    if (input.compare(0, prefix.size(), prefix) == 0) {
      return input.substr(prefix.size());
    }
    return input;
  }

  static inline const std::string encodeTable = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                                "abcdefghijklmnopqrstuvwxyz"
                                                "0123456789+/";

  static inline const std::unordered_map<char, uint8_t> decodeTable = {
      {'A', 0},  {'B', 1},  {'C', 2},  {'D', 3},  {'E', 4},  {'F', 5},
      {'G', 6},  {'H', 7},  {'I', 8},  {'J', 9},  {'K', 10}, {'L', 11},
      {'M', 12}, {'N', 13}, {'O', 14}, {'P', 15}, {'Q', 16}, {'R', 17},
      {'S', 18}, {'T', 19}, {'U', 20}, {'V', 21}, {'W', 22}, {'X', 23},
      {'Y', 24}, {'Z', 25}, {'a', 26}, {'b', 27}, {'c', 28}, {'d', 29},
      {'e', 30}, {'f', 31}, {'g', 32}, {'h', 33}, {'i', 34}, {'j', 35},
      {'k', 36}, {'l', 37}, {'m', 38}, {'n', 39}, {'o', 40}, {'p', 41},
      {'q', 42}, {'r', 43}, {'s', 44}, {'t', 45}, {'u', 46}, {'v', 47},
      {'w', 48}, {'x', 49}, {'y', 50}, {'z', 51}, {'0', 52}, {'1', 53},
      {'2', 54}, {'3', 55}, {'4', 56}, {'5', 57}, {'6', 58}, {'7', 59},
      {'8', 60}, {'9', 61}, {'+', 62}, {'/', 63}};
};

} // namespace flowengine::core

#endif
