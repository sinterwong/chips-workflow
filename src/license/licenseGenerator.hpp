#ifndef __FLOWENGINE_LICENSE_GENERATOR_HPP_
#define __FLOWENGINE_LICENSE_GENERATOR_HPP_
#include <iostream>
#include <fstream>
#include<openssl/rsa.h>
#include<openssl/pem.h>
#include<cstring>
#include<algorithm>

namespace fe_license {
#define KEY_LEN 1024
#define KEY_LEN_BYTE 128
#define ARGV_NUM 8

inline int min(int x, int y) { return x > y ? y : x; }
inline int maxCodeLen(RSA* pRsa) { return RSA_size(pRsa) - 11; }
int makeKey(char* privateKeyFile, char* publicKeyFile);
int SignGenerate(char* rsaPrivateKeyFileName, unsigned char* signIn, int signInLen, unsigned char* signOut, int* signOutLen);
/**
 * @brief Generate Message File which contains using times and config file's name
 * 
 */
int MessageGenerate(char* rsaPrivateKeyFileName, std::string& msgIn, std::string** msgOut);

/**
 *  XOR Encrypt Algorithm.
 *  
 *  @param[in] key_str key used to generate code.
 *  @param[in] context_str context to encrypt.
 *  
 *  @return encrypted code
*/
std::string encrypt(const std::string& key_str, const std::string& context_str);
}
#endif