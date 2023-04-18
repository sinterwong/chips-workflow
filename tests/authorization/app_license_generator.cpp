#include "license/licenseGenerator.hpp"
using namespace fe_license;

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
    std::cout << "Incorrect arguments" << std::endl;
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
  std::string dtsIDStr(dtsID);
  // transform(dtsIDStr.begin(), dtsIDStr.end(), dtsIDStr.begin(), ::toupper);
  //  Key Generation
  if (makeKey(privateKeyFile, publicKeyFile) != 0) {
    std::cout << "Key Generation Failed" << std::endl;
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
    std::cout << "License Generation Failed" << std::endl;
    return -3;
  }
  // Write License
  std::ofstream out(licenseAddress, std::ios::out | std::ios::binary);
  if (!out) {
    std::cout << "Can Not Open File" << std::endl;
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
  std::string msgIn(msgIn_arr,
                    sizeof(int) + sizeof(int) + configFileLen * sizeof(char));

  std::string *msgOutPtr = nullptr;
  MessageGenerate(privateKeyFile, msgIn, &msgOutPtr);
  // XOR Encrypt the RSA Encrypted Message and Write Message File
  out.open(messageFileName, std::ios::out | std::ios::binary);
  if (!out) {
    std::cout << "Can Not Open File" << std::endl;
    return -5;
  }

  // Format---- key(int) + RSA_encrypt_len(int) + context_str(string)
  int RSAMsgLen = msgOutPtr->length();
  srand(time(0));
  int key = rand();
  std::string key_str = std::to_string(key);
  std::string xorEncryptStr = encrypt(key_str, *msgOutPtr);
  out.write((char *)&key, sizeof(int));
  out.write((char *)&RSAMsgLen, sizeof(int));
  out.write(xorEncryptStr.c_str(), RSAMsgLen * sizeof(char));
  out.close();
  delete msgOutPtr;

  return 0;
}
// ./app_license_generator privateKey licenseA 1423721066487 licenseB 1000 /etc/apt/.config licenseC
