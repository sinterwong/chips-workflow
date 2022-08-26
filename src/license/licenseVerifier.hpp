#ifndef __FLOWENGINE_LICENSE_VERIFIER_HPP_
#define __FLOWENGINE_LICENSE_VERIFIER_HPP_

#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unistd.h>
#include <vector>

namespace fe_license {

#define MAX_INTERFACE 16
#define TRIAL_TIMES 20
//#define TRIAL_FILE "/etc/apt/.en_t" // AKA. ENHANCE TRIAL
#define CHECK_FAILED -1

#define PUBLIC_KEY "licenseA"
#define LICENSE_FILE "licenseB"
#define MESSAGE_FILE "licenseC"

#define PRE_VERIFY_NUM 13

/**
 * @brief   Get the Device-Tree Serial-Number of current device.
 * @param[in] dstID File Name of dstID
 * @return  Device-Tree Serial-Number of current device.
 */
std::string getDTSID(const char *dtsIDFile);

bool checkTrial(const char *trialFile);

int checkLicense(const char *publicKeyFile, const char *licenseFile,
                 const char *messageFile, const char *dtsIDFile);

int SignVerify(const char *rsaPublicKeyFileName, std::string &src,
               unsigned char *sign, int signLen);

bool checkTimeToUse(const char *publicKeyFile, const char *licenseFile,
                    const char *messageFile);

/**
 * @brief Check if a file exist
 *
 * @param file File to check.
 * @return true return true if file exist, false otherwise.
 */
inline bool checkExist(const std::string &file) {
  return access(file.c_str(), F_OK) == 0;
}

/**
 *  Inner Call
 * @param publicKeyFile
 * @param licenseFile
 * @param messageFile
 * @return
 */
int checkLicenseTime(const char *publicKeyFile, const char *licenseFile,
                     const char *messageFile, bool isMinus);

int checkTrialTime(const char *trialFile, bool isMinus);

/**
 *  XOR Encrypt Algorithm.
 *
 *  @param[in] key_str key used to generate code.
 *  @param[in] context_str context to encrypt.
 *
 *  @return encrypted code
 */
std::string encrypt(const std::string &key_str, const std::string &context_str);

void readConfOnce(std::ifstream &in, std::string &addr);

} // namespace fe_license
#endif