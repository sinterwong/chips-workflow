#include "license/packageDecrypter.hpp"
#include <iostream>


int main(int argc, char* argv[])
{
    if(argc != 2)
    {
        std::cout << "Wrong arguments." << std::endl;
        return -1;
    }
    char* packageFile = argv[1];

    std::string dtsID = fe_license::DecryptPackage(packageFile);
    if(dtsID == "")
    {
        std::cout << "Decrypt Failed." << std::endl;
        return -2;
    }

    std::cout << "Device-Tree Serial-Num: " << dtsID << std::endl;

	return 0;
}
