/**
 * @file FaceDto.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2023-11-14
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#ifndef __SERVER_FACE_DTO_FACE_REQUEST_DTO_HPP_
#define __SERVER_FACE_DTO_FACE_REQUEST_DTO_HPP_

#include "oatpp/core/Types.hpp"
#include "oatpp/core/macro/codegen.hpp"
#include <string>

namespace server::face {
#include OATPP_CODEGEN_BEGIN(DTO)

class FaceDto : public oatpp::DTO {
  DTO_INIT(FaceDto, DTO)
  DTO_FIELD_INFO(userId) { info->description = "The user's ID number"; }
  DTO_FIELD(String, userId);
  DTO_FIELD_INFO(url) { info->description = "The url of image"; }
  DTO_FIELD(String, url);

public:
  std::string toString() const {
    auto str = "FaceRequestDto(\n"
               "  userId: " +
               userId + ",\n" + "  url: " + url + "\n" + ")";
    return str;
  }
};
#include OATPP_CODEGEN_END(DTO) ///< End DTO codegen section
} // namespace server::face
#endif