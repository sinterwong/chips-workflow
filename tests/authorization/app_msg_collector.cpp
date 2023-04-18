#include "license/msgCollector.hpp"
#include <iostream>
#include <fstream>
#include <ctime>
#include <cstdlib>

#define PACKAGE_NAME "package"
#define CONF_FILE "msg_collect_conf"    // Conf_file: Address_num(int) + Address_num*[key(int)+address_len(int)+xorEncrypAddr]
#define CONF_ADDR_NUM 1

using namespace fe_license;
/**
 * @brief CONF_FILE: dtsIDFile
 */
int main(int argc, char** argv)
{
    std::string packageName = PACKAGE_NAME;
    srand(time(0));

    // Read Conf File
    std::ifstream in(CONF_FILE, std::ios::binary);
    if(!in)
    {
        in.close();
        std::cout << "Read Conf file failed." << std::endl;
        return -1;
    }

    int addr_num;
    std::string key_str;
    std::string encrypt_str;

    in.read((char*)&addr_num, sizeof(int));
    if(addr_num != CONF_ADDR_NUM)
    {
        in.close();
        std::cout << "Conf file error." << std::endl;
        return -2;
    }

    // Begin reading address
    std::string dtsIDFile;
    readConfOnce(in, dtsIDFile);
    in.close();


    if(!generateMsgPackage(packageName, dtsIDFile))
    {
        std::cout << "Failed to generate package." << std::endl;
        return -1;
    }
    std::cout << "Package generation succeeds." << std::endl;
    return 0;
}
