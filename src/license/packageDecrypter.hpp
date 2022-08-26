#ifndef __FLOWENGINE_PACKAGE_DECRYPTER_HPP_
#define __FLOWENGINE_PACKAGE_DECRYPTER_HPP_

#include <string>

namespace fe_license {

/**
 *  @brief  Decrypt Package sent by user.
 *  @return Get user's Machine ID.
 */
std::string DecryptPackage(char *packageFile);

/**
 *  XOR Encrypt Algorithm.
 *
 *  @param[in] key_str key used to generate code.
 *  @param[in] context_str context to encrypt.
 *
 *  @return encrypted code
 */
std::string xorEncrypt(const std::string &key_str,
                       const std::string &context_str);

} // namespace fe_license
#endif