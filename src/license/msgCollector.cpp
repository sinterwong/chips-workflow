#include "msgCollector.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <net/if.h>
#include <netinet/in.h>
#include <sys/ioctl.h>
#include <unistd.h>

namespace fe_license {

/**
 *  Generate package containing user's MAC addresses.
 *
 *  @param[in] outputFile generate file.
 *  @param[in] dtsIDFile File Name of Device-Tree Serial-Num.
 *
 *  @return return true if succeed, false otherwise.
 */
bool generateMsgPackage(std::string &outputFile, std::string &dtsIDFile) {
  // Get DTS-ID
  std::string dtsID = getDTSID(dtsIDFile.c_str());
  if (dtsID == "") {
    return false;
  }

  std::ofstream out(outputFile, std::ios::binary);

  // Format---- key(int) + encrypt_len(int) + context_str(string)
  int key = rand();
  std::string key_str = std::to_string(key);
  std::string encrypt_str = encrypt(key_str, dtsID);

  int encrypt_len = encrypt_str.length();

  out.write((char *)&key, sizeof(int));
  out.write((char *)&encrypt_len, sizeof(int));
  out.write(encrypt_str.c_str(), encrypt_len * sizeof(char));

  out.close();

  return true;
}
} // namespace fe_config