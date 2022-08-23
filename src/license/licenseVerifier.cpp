#include "licenseVerifier.hpp"
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <net/if.h>
#include <netinet/in.h>
#include <openssl/pem.h>
#include <pwd.h>
#include <string.h>
#include <sys/ioctl.h>

namespace fe_license {
#define PUBLIC_KEY "licenseA"
#define LICENSE_FILE "licenseB"
#define MESSAGE_FILE "licenseC"
#define PRE_VERIFY_NUM 13

//#define MACHINE_ID_FILE "/etc/machine-id"
/*
 * Using XOR Encrypt Algorithm to Check Trial Times
 */
bool checkTrial(const char *trialFile) {
  return checkTrialTime(trialFile, true) != CHECK_FAILED;
}

bool checkTimeToUse(const char *publicKeyFile, const char *licenseFile,
                    const char *messageFile) {
  return checkLicenseTime(publicKeyFile, licenseFile, messageFile, true) !=
         CHECK_FAILED;
}

/**
 * @brief   Get the Device-Tree Serial-Number of current device.
 * @param[in] dstID File Name of dstID
 * @return  Device-Tree Serial-Number of current device.
 */
std::string getDTSID(const char *dtsIDFile) {
  std::string id_str;
  std::ifstream in(dtsIDFile);
  if (!in) {
    in.close();
    std::cout << "Collection Error. Please try 'sudo'." << std::endl;
    return id_str;
  }
  in >> id_str;
  return id_str;
}

int checkLicense(const char *publicKeyFile, const char *licenseFile,
                 const char *messageFile, const char *dtsIDFile) {
  // Get Device-Tree Serial Num Address
  std::string DTSID = getDTSID(dtsIDFile);
  if (DTSID == "") {
    std::cout << "System Environment Setting Error." << std::endl;
    return -1;
  }

  // Read License File and Get Signature
  std::ifstream in(licenseFile, std::ios::in | std::ios::binary);
  if (!in) {
    in.close();
    std::cout << "Read licenseB Failed." << std::endl;
    return -2;
  }

  int signLen = 0;
  in.read((char *)&signLen, sizeof(int));
  char sign_arr[signLen];
  in.read(sign_arr, signLen * sizeof(char));

  // Verify License
  int ret =
      SignVerify(publicKeyFile, DTSID, (unsigned char *)sign_arr, signLen);
  in.close();
  if (ret != 0) {
    std::cout << "Read licenses Failed." << std::endl;
    return -3;
  }

  if (!checkTimeToUse(publicKeyFile, licenseFile, messageFile)) {
    return -4;
  }
  return 0;
}

// If Verify Successfully, return 0
int SignVerify(const char *rsaPublicKeyFileName, std::string &src,
               unsigned char *sign, int signLen) {
  unsigned char hash[SHA256_DIGEST_LENGTH];
  // Read Public Key
  BIO *pBio = BIO_new_file(rsaPublicKeyFileName, "rb");
  if (pBio == nullptr) {
    // Read File Failed
    std::cout << "Read licenseA Failed." << std::endl;
    BIO_free_all(pBio);
    return -1;
  }
  RSA *pRsa = PEM_read_bio_RSAPublicKey(pBio, nullptr, nullptr, nullptr);
  BIO_free_all(pBio);
  if (pRsa == nullptr) {
    // Read Key Failed
    std::cout << "Read licenseA Failed." << std::endl;
    RSA_free(pRsa);
    return -2;
  }

  // Abstract
  int signInLen = src.size() > PRE_VERIFY_NUM ? PRE_VERIFY_NUM : src.size();
  SHA256((unsigned char *)src.c_str(), signInLen, hash);
  // Verify Sign
  int iRet =
      RSA_verify(NID_sha256, hash, SHA256_DIGEST_LENGTH, sign, signLen, pRsa);

  if (iRet == 1) {
    // Verify Successfully
    return 0;
  }
  // Sign Generate Failed
  RSA_free(pRsa);
  return -3;
}

/**
 *  Inner Call
 * @param publicKeyFile
 * @param licenseFile
 * @param messageFile
 * @return
 */
int checkLicenseTime(const char *publicKeyFile, const char *licenseFile,
                     const char *messageFile, bool isMinus) {
  // Read Public Key
  BIO *pBio = BIO_new_file(publicKeyFile, "rb");
  if (pBio == nullptr) {
    // Read File Failed
    std::cout << "Read licenseA Failed." << std::endl;
    return CHECK_FAILED;
  }
  RSA *pRsa = PEM_read_bio_RSAPublicKey(pBio, nullptr, nullptr, nullptr);
  BIO_free_all(pBio);
  if (pRsa == nullptr) {
    // Read Key Failed
    std::cout << "Read licenseA Failed." << std::endl;
    return CHECK_FAILED;
  }

  // Check if license file exist
  std::ifstream in(licenseFile, std::ios::in | std::ios::binary);
  if (!in) {
    std::cout << "Read licenseB Failed." << std::endl;
    in.close();
    RSA_free(pRsa);
    return CHECK_FAILED;
  }
  in.close();

  // Read Message File
  in.open(messageFile, std::ios::in | std::ios::binary);
  if (!in) {
    in.close();
    RSA_free(pRsa);
    std::cout << "Read licenseC Failed." << std::endl;
    return CHECK_FAILED;
  }

  // Format---- key(int) + RSA_encrypt_len(int) + context_str(string)
  int key;
  int RSAMsgLen;
  in.read((char *)&key, sizeof(int));
  in.read((char *)&RSAMsgLen, sizeof(int));
  char xorEncryptMsg[RSAMsgLen];
  in.read(xorEncryptMsg, RSAMsgLen * sizeof(char));
  in.close();
  // XOR Decrypt
  std::string key_str = std::to_string(key);
  std::string encrypt_str(xorEncryptMsg, RSAMsgLen);
  std::string msgIn = encrypt(key_str, encrypt_str);
  // RSA Decrypt
  char buffer[1024];
  memset(buffer, 0, 1024 * sizeof(char));
  int ret = RSA_public_decrypt(
      RSAMsgLen, reinterpret_cast<const unsigned char *>(msgIn.c_str()),
      reinterpret_cast<unsigned char *>(buffer), pRsa, RSA_PKCS1_PADDING);

  if (ret < 0) {
    std::cout << "Read licenses Failed." << std::endl;
    RSA_free(pRsa);
    return CHECK_FAILED;
  }
  RSA_free(pRsa);
  // Message = time_to_use(int) + config_file_name(int) +
  // config_file_name(char*)
  int maxTimeToUse = 0;
  int configFileLen = 0;
  memcpy(&maxTimeToUse, buffer, sizeof(int));
  memcpy(&configFileLen, buffer + sizeof(int), sizeof(int));
  char configFile[configFileLen + 1];
  memcpy(configFile, buffer + 2 * sizeof(int), configFileLen * sizeof(char));
  configFile[configFileLen] = '\0';

  // Check time at config file
  in.open(configFile, std::ios::binary);
  if (!in) {
    std::cout << "Check config failed. Please make sure config succeeds."
              << std::endl;
    in.close();
    return CHECK_FAILED;
  }
  // Decrypt config file
  int leftTime;
  int encryptTime;
  in.read((char *)&key, sizeof(int));
  in.read((char *)&encryptTime, sizeof(int));
  leftTime = key ^ encryptTime;
  in.close();

  if (leftTime <= 0) {
    std::cout << "Running out off License Times." << std::endl;
    return CHECK_FAILED;
  }
  if (leftTime > maxTimeToUse) {
    leftTime = maxTimeToUse;
  }

  std::ofstream out(configFile, std::ios::binary);
  if (!out.is_open()) {
    std::cout << "Check license failed. Please make sure config succeeds."
              << std::endl;
    out.close();
    return CHECK_FAILED;
  }

  key = rand();
  if (isMinus) {
    leftTime--;
  }
  encryptTime = leftTime ^ key;
  out.write((char *)&key, sizeof(int));
  out.write((char *)&encryptTime, sizeof(int));
  out.close();

  return leftTime;
}

int checkTrialTime(const char *trialFile, bool isMinus) {
  std::ifstream in(trialFile, std::ios::binary);
  if (!in) {
    std::cout << "Check config failed. Please make sure config succeeds."
              << std::endl;
    in.close();
    return CHECK_FAILED;
  } else {
    int trialTime;
    int key;
    int encryptTrialTime;
    in.read((char *)&key, sizeof(int));
    in.read((char *)&encryptTrialTime, sizeof(int));
    trialTime = key ^ encryptTrialTime;
    in.close();
    if (trialTime <= 0) {
      std::cout << "Running out off Trial." << std::endl;
      return CHECK_FAILED;
    } else {
      if (trialTime >= TRIAL_TIMES) {
        trialTime = TRIAL_TIMES;
      }

      std::ofstream out(trialFile, std::ios::binary);
      if (!out.is_open()) {
        std::cout << "Check trial failed. Please make sure config succeeds."
                  << std::endl;
        out.close();
        return CHECK_FAILED;
      }

      key = rand();
      if (isMinus) {
        trialTime--;
      }
      encryptTrialTime = key ^ trialTime;
      out.write((char *)&key, sizeof(int));
      out.write((char *)&encryptTrialTime, sizeof(int));
      out.close();
      return trialTime;
    }
  }
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

void readConfOnce(std::ifstream &in, std::string &addr) {
  int key;
  int addr_len;
  std::string key_str;
  std::string encrypt_str;

  in.read((char *)&key, sizeof(int));
  in.read((char *)&addr_len, sizeof(int));
  char encryptBuffer[addr_len];
  in.read(encryptBuffer, addr_len * sizeof(char));
  key_str = std::to_string(key);
  encrypt_str = std::string(encryptBuffer, addr_len);
  addr = encrypt(key_str, encrypt_str);
}
} // namespace fe_license