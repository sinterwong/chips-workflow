#include "licenseGenerator.hpp"

namespace fe_license {
#define PRE_VERIFY_NUM 13

int makeKey(char *privateKeyFile, char *publicKeyFile) {
  // Generate Key Pair
  RSA *pRsa = RSA_new();
  BIGNUM *bn = BN_new();
  BN_set_word(bn, RSA_F4);
  int ret = RSA_generate_key_ex(pRsa, KEY_LEN, bn, nullptr);
  if (ret != 1) {
    std::cout << "Key Pair Generation Failed" << std::endl;
    RSA_free(pRsa);
    return -1;
  }
  // Public Key
  BIO *pBio = BIO_new_file(publicKeyFile, "wb");
  if (pBio == nullptr) {
    std::cout << "File Open Failed" << std::endl;
    RSA_free(pRsa);
    return -2;
  }
  PEM_write_bio_RSAPublicKey(pBio, pRsa);
  BIO_free_all(pBio);
  // Private Key
  pBio = BIO_new_file(privateKeyFile, "wb");
  if (pBio == nullptr) {
    std::cout << "File Open Failed" << std::endl;
    RSA_free(pRsa);
    return -2;
  }
  PEM_write_bio_RSAPrivateKey(pBio, pRsa, nullptr, nullptr, 0, nullptr,
                              nullptr);
  std::cout << "Key Pair Generation Succeeded" << std::endl;
  BIO_free_all(pBio);
  RSA_free(pRsa);

  return 0;
}

// If Verify Successfully, return 0
int SignGenerate(char *rsaPrivateKeyFileName, unsigned char *signIn,
                 int signInLen, unsigned char *signOut, int *signOutLen) {
  unsigned char hash[SHA256_DIGEST_LENGTH];
  // Read Private Key
  BIO *pBio = BIO_new_file(rsaPrivateKeyFileName, "rb");
  if (pBio == nullptr) {
    // Read File Failed
    std::cout << "Read File Failed" << std::endl;
    BIO_free_all(pBio);
    return -1;
  }
  RSA *pRsa = PEM_read_bio_RSAPrivateKey(pBio, nullptr, nullptr, nullptr);
  BIO_free_all(pBio);
  if (pRsa == nullptr) {
    // Read Key Failed
    std::cout << "Read Private Key Failed" << std::endl;
    RSA_free(pRsa);
    return -2;
  }
  // Abstract
  signInLen = signInLen > PRE_VERIFY_NUM ? PRE_VERIFY_NUM : signInLen;
  SHA256(signIn, signInLen, hash);

  // Sign
  int iRet = RSA_sign(NID_sha256, hash, SHA256_DIGEST_LENGTH, signOut,
                      (unsigned int *)signOutLen, pRsa);

  if (iRet != 1) {
    // Sign Generate Failed
    std::cout << "Signature Generation Failed" << std::endl;
    RSA_free(pRsa);
    return -2;
  }

  std::cout << "Signature Generation Succeeded" << std::endl;
  RSA_free(pRsa);

  return 0;
}

/**
 * @brief Generate Message File which contains using times and config file's
 * length and its name
 *
 */
int MessageGenerate(char *rsaPrivateKeyFileName, std::string &msgIn,
                    std::string **msgOut) {
  // Read Private Key
  BIO *pBio = BIO_new_file(rsaPrivateKeyFileName, "rb");
  if (pBio == nullptr) {
    // Read File Failed
    std::cout << "Read File Failed" << std::endl;
    BIO_free_all(pBio);
    return -1;
  }
  RSA *pRsa = PEM_read_bio_RSAPrivateKey(pBio, nullptr, nullptr, nullptr);
  BIO_free_all(pBio);

  if (pRsa == nullptr) {
    // Read Key Failed
    std::cout << "Read Private Key Failed" << std::endl;
    RSA_free(pRsa);
    return -2;
  }

  int len = RSA_size(pRsa);
  unsigned char msgOutArr[len];
  int ret =
      RSA_private_encrypt(msgIn.length(), (const unsigned char *)msgIn.c_str(),
                          (unsigned char *)msgOutArr, pRsa, RSA_PKCS1_PADDING);
  if (ret >= 0) {
    *msgOut = new std::string((char *)msgOutArr, ret);
  } else {
    return -3;
  }
  std::cout << "Message Writing Succeeded" << std::endl;
  RSA_free(pRsa);

  return 0;
}

/**
 *  XOR Encrypt Algorithm.
 *
 *  @param[in] key_str key used to generate code.
 *  @param[in] context_str context to encrypt.
 *
 *  @return encrypted code
 */
std::string encrypt(const std::string &key_str,
                    const std::string &context_str) {
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