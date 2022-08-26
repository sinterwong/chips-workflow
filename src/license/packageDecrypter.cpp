#include "packageDecrypter.hpp"
#include <fstream>
#include <iostream>

namespace fe_license {
/**
 *  @brief  Decrypt Package sent by user.
 *  @param[in] packageFile Name of Encrypted Package.
 *  @return Get user's Device-Tree Serial-Num(dtsID).
 */
std::string DecryptPackage(char *packageFile) {
  std::string dtsID;
  std::ifstream in(packageFile, std::ios::binary);
  if (!in) {
    in.close();
    std::cout << "Unable to open package." << std::endl;
    return dtsID;
  }

  // Format---- key(int) + encrypt_len(int) + context_str(string)
  // Read Machine ID
  int key;
  int encrypt_len;
  in.read((char *)&key, sizeof(int));
  in.read((char *)&encrypt_len, sizeof(int));

  char encrypt[encrypt_len];
  in.read(encrypt, encrypt_len * sizeof(char));

  // Decrypt
  std::string key_str = std::to_string(key);
  std::string encrypt_str(encrypt, encrypt_len);
  dtsID = xorEncrypt(key_str, encrypt_str);

  in.close();
  return dtsID;
}

/**
 *  XOR Encrypt Algorithm.
 *
 *  @param[in] key_str key used to generate code.
 *  @param[in] context_str context to encrypt.
 *
 *  @return encrypted code
 */
std::string xorEncrypt(const std::string &key_str, const std::string &context_str) {
  int key_len = key_str.length();
  int context_len = context_str.length();
  char encrypt_byte[context_len];

  for (int i = 0; i < context_len; i++) {
    int key_pos = i % key_len;
    encrypt_byte[i] = key_str[key_pos] ^ context_str[i];
  }
  return std::string(encrypt_byte, context_len);
}
} // namespace fe_license
