#include <iostream>
#include <fstream>
#include <ctime>
#include <cstdlib>
#include <string>

void writeConfOnce(std::ofstream& out, std::string&& addr);
std::string encrypt(const std::string& key_str, const std::string& context_str);

/**
 * @brief Conf_file: Address_num(int) + Address_num*[key(int)+address_len(int)+xorEncrypAddr]
 * @param[in] argv 
 * argv[1] = Conf File Name
 * argv[2] = Address_Num
 * argv[i] = Address i (i > 2)
 */
int main(int argc, char** argv)
{
    if(argc < 3)
    {
        std::cout << "Incorrect Arguments." << std::endl;
        return -1;
    }

    char* confFile = argv[1];
    int addrNum = atoi(argv[2]);
    if(argc != addrNum + 3)
    {
        std::cout << "Incorrect Arguments." << std::endl;
        return -2;
    }

    srand(time(0));

    // Write Conf File
    std::ofstream out(confFile, std::ios::binary);
    if(!out)
    {
        out.close();
        std::cout << "Write Conf file failed." << std::endl;
        return -3;
    }

    out.write((char*)&addrNum, sizeof(int));
    for(int i = 3; i < argc; i++)
    {
        writeConfOnce(out, argv[i]);
    }
    out.close();

    std::cout << "Write Conf file succeeds." << std::endl;
    return 0;
}


void writeConfOnce(std::ofstream& out, std::string&& addr)
{
    int key = rand();
    std::string key_str = std::to_string(key);;
    std::string encrypt_str = encrypt(key_str, addr);
    int encrypt_len = encrypt_str.length();

    out.write((char*)&key, sizeof(int));
    out.write((char*)&encrypt_len, sizeof(int));
    out.write(encrypt_str.c_str(), encrypt_len * sizeof(char));
}


/**
 *  XOR Encrypt Algorithm.
 *  
 *  @param[in] key_str key used to generate code.
 *  @param[in] context_str context to encrypt.
 *  
 *  @return encrypted code
*/
std::string encrypt(const std::string& key_str, const std::string& context_str)
{
    int key_len = key_str.length();
    int context_len = context_str.length();
    char encrypt_byte[context_len];

    for(int i = 0; i < context_len; i++)
    {
        int key_pos = i % key_len;
        encrypt_byte[i] = key_str[key_pos] ^ context_str[i];
    }
    return std::string(encrypt_byte, context_len);
}


// ./app_conf_generator msg_collect_conf 1 /proc/device-tree/serial-number
