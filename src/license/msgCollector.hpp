#ifndef __FLOWENGINE_MSG_COLLECTOR_HPP_
#define __FLOWENGINE_MSG_COLLECTOR_HPP_

#include <string>
#include <vector>
#include "licenseVerifier.hpp"

namespace fe_license {

/**
 *  Generate package containing user's MAC addresses.
 *
 *  @param[in] outputFile generate file.
 *  @param[in] dtsIDFile File Name of Device-Tree Serial-Num.
 *
 *  @return return true if succeed, false otherwise.
 */
bool generateMsgPackage(std::string &outputFile, std::string &dtsIDFile);
} // namespace fe_license
#endif