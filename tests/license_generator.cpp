#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <openssl/pem.h>
#include <openssl/rsa.h>

#define KEY_LEN 1024
#define KEY_LEN_BYTE 128
#define ARGV_NUM 8

using std::cout;
using std::endl;
using std::string;

#define PRE_VERIFY_NUM 13

int makeKey(char *privateKeyFile, char *publicKeyFile) {
  // Generate Key Pair
  RSA *pRsa = RSA_new();
  BIGNUM *bn = BN_new();
  BN_set_word(bn, RSA_F4);
  int ret = RSA_generate_key_ex(pRsa, KEY_LEN, bn, nullptr);
  if (ret != 1) {
    cout << "Key Pair Generation Failed" << endl;
    RSA_free(pRsa);
    return -1;
  }
  // Public Key
  BIO *pBio = BIO_new_file(publicKeyFile, "wb");
  if (pBio == nullptr) {
    cout << "File Open Failed" << endl;
    RSA_free(pRsa);
    return -2;
  }
  PEM_write_bio_RSAPublicKey(pBio, pRsa);
  BIO_free_all(pBio);
  // Private Key
  pBio = BIO_new_file(privateKeyFile, "wb");
  if (pBio == nullptr) {
    cout << "File Open Failed" << endl;
    RSA_free(pRsa);
    return -2;
  }
  PEM_write_bio_RSAPrivateKey(pBio, pRsa, nullptr, nullptr, 0, nullptr,
                              nullptr);
  cout << "Key Pair Generation Succeeded" << endl;
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
    cout << "Read File Failed" << endl;
    BIO_free_all(pBio);
    return -1;
  }
  RSA *pRsa = PEM_read_bio_RSAPrivateKey(pBio, nullptr, nullptr, nullptr);
  BIO_free_all(pBio);
  if (pRsa == nullptr) {
    // Read Key Failed
    cout << "Read Private Key Failed" << endl;
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
    cout << "Signature Generation Failed" << endl;
    RSA_free(pRsa);
    return -2;
  }

  cout << "Signature Generation Succeeded" << endl;
  RSA_free(pRsa);

  return 0;
}

/**
 * @brief Generate Message File which contains using times and config file's
 * length and its name
 *
 */
int MessageGenerate(char *rsaPrivateKeyFileName, string &msgIn,
                    string **msgOut) {
  // Read Private Key
  BIO *pBio = BIO_new_file(rsaPrivateKeyFileName, "rb");
  if (pBio == nullptr) {
    // Read File Failed
    cout << "Read File Failed" << endl;
    BIO_free_all(pBio);
    return -1;
  }
  RSA *pRsa = PEM_read_bio_RSAPrivateKey(pBio, nullptr, nullptr, nullptr);
  BIO_free_all(pBio);

  if (pRsa == nullptr) {
    // Read Key Failed
    cout << "Read Private Key Failed" << endl;
    RSA_free(pRsa);
    return -2;
  }

  int len = RSA_size(pRsa);
  unsigned char msgOutArr[len];
  int ret =
      RSA_private_encrypt(msgIn.length(), (const unsigned char *)msgIn.c_str(),
                          (unsigned char *)msgOutArr, pRsa, RSA_PKCS1_PADDING);
  if (ret >= 0) {
    *msgOut = new string((char *)msgOutArr, ret);
  } else {
    return -3;
  }
  cout << "Message Writing Succeeded" << endl;
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
string encrypt(const string &key_str, const string &context_str) {
  int key_len = key_str.length();
  int context_len = context_str.length();
  char encrypt_byte[context_len];

  for (int i = 0; i < context_len; i++) {
    int key_pos = i % key_len;
    encrypt_byte[i] = key_str[key_pos] ^ context_str[i];
  }
  return string(encrypt_byte, context_len);
}

/*
 *  Generate License Using RSA Algorithm
 *  Output : Private & Public Key and a License
 *  argv[1] = privateKeyFile
 *  argv[2] = publicKeyFile
 *  argv[3] = Device-Tree Serial-Num(dtsID) for License Generation
 *  argv[4] = License's Name
 *  argv[5] = Times to use
 *  argv[6] = Config File Name & Address(used at config.exe)
 *  argv[7] = Message File outputName
 */
int main(int argc, char *argv[]) {
  if (argc != ARGV_NUM) {
    cout << "Incorrect arguments" << endl;
    return -1;
  }
  char *privateKeyFile = argv[1];
  char *publicKeyFile = argv[2];
  char *dtsID = argv[3];
  char *licenseAddress = argv[4];
  char *timeMsg = argv[5];
  char *configName = argv[6];
  char *messageFileName = argv[7];
  // Change Mac to Upper Case
  string dtsIDStr(dtsID);
  // transform(dtsIDStr.begin(), dtsIDStr.end(), dtsIDStr.begin(), ::toupper);
  //  Key Generation
  if (makeKey(privateKeyFile, publicKeyFile) != 0) {
    cout << "Key Generation Failed" << endl;
    return -2;
  }
  // Signature Generation
  int ret;
  char licenseContext[KEY_LEN_BYTE];
  int licenseLen = 0;
  ret = SignGenerate(privateKeyFile, (unsigned char *)dtsIDStr.c_str(),
                     dtsIDStr.length(), (unsigned char *)licenseContext,
                     &licenseLen);

  if (ret != 0) {
    cout << "License Generation Failed" << endl;
    return -3;
  }
  // Write License
  std::ofstream out(licenseAddress, std::ios::out | std::ios::binary);
  if (!out) {
    cout << "Can Not Open File" << endl;
    return -4;
  }

  // Write len first
  out.write((char *)&licenseLen, sizeof(int));
  out.write(licenseContext, licenseLen * sizeof(char));

  out.close();
  // Encrypt Message
  int timeToUse = atoi(timeMsg);
  int configFileLen = strlen(configName);
  char msgIn_arr[sizeof(int) + sizeof(int) + configFileLen * sizeof(char)];
  memcpy(msgIn_arr, &timeToUse, sizeof(int));
  memcpy(msgIn_arr + sizeof(int), &configFileLen, sizeof(int));
  memcpy(msgIn_arr + 2 * sizeof(int), configName, configFileLen * sizeof(char));
  string msgIn(msgIn_arr,
               sizeof(int) + sizeof(int) + configFileLen * sizeof(char));

  string *msgOutPtr = nullptr;
  MessageGenerate(privateKeyFile, msgIn, &msgOutPtr);
  // XOR Encrypt the RSA Encrypted Message and Write Message File
  out.open(messageFileName, std::ios::out | std::ios::binary);
  if (!out) {
    cout << "Can Not Open File" << endl;
    return -5;
  }

  // Format---- key(int) + RSA_encrypt_len(int) + context_str(string)
  int RSAMsgLen = msgOutPtr->length();
  srand(time(0));
  int key = rand();
  string key_str = std::to_string(key);
  string xorEncryptStr = encrypt(key_str, *msgOutPtr);
  out.write((char *)&key, sizeof(int));
  out.write((char *)&RSAMsgLen, sizeof(int));
  out.write(xorEncryptStr.c_str(), RSAMsgLen * sizeof(char));
  out.close();
  delete msgOutPtr;

  return 0;
}
