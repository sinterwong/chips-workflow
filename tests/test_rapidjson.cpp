#include "logger/logger.hpp"
#include "rapidjson/document.h"
#include <fstream>
#include <iostream>
#include <iterator>

bool readFile(std::string const &filename, std::string &result) {
  std::ifstream input_file(filename);
  if (!input_file.is_open()) {
    std::cerr << "Could not open the file - '" << filename << "'" << std::endl;
    return false;
  }
  result = std::string((std::istreambuf_iterator<char>(input_file)),
                       std::istreambuf_iterator<char>());
  return true;
}

int main(int argc, char **argv) {
  FlowEngineLoggerInit(true, true, true, true);
  std::string filePath =
      "/home/wangxt/workspace/projects/flowengine/build/agent_err.json";
  std::string jsonstr;
  readFile(filePath, jsonstr);
  std::cout << jsonstr << std::endl;
  std::cout << jsonstr.size() << std::endl;
  rapidjson::Document d;
  if (d.Parse(jsonstr.c_str()).HasParseError()) {
    auto error_code = d.GetParseError();
    std::cout << error_code << std::endl;
  }
  if (!d.IsObject()) {
    FLOWENGINE_LOGGER_ERROR("must be an object");
  }

  return 0;
}