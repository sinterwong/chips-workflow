#ifndef __CRUD_STREAM_DTO_HPP_
#define __CRUD_STREAM_DTO_HPP_

#include "oatpp/core/Types.hpp"
#include "oatpp/core/macro/codegen.hpp"
namespace server::face {
#include OATPP_CODEGEN_BEGIN(DTO)

class StreamDto : public oatpp::DTO {

  DTO_INIT(StreamDto, DTO)

  DTO_FIELD_INFO(name) { info->description = "The name of stream"; }
  DTO_FIELD(String, name);

  DTO_FIELD_INFO(url) { info->description = "The url of stream"; }
  DTO_FIELD(String, url);

public:
  std::string toString() const {
    auto str = "StreamDto(\n"
               "  name: " +
               (name ? *name : "null") + ",\n" +
               "  url: " + (url ? *url : "null") + "\n" + ")";
    return str;
  }
};

#include OATPP_CODEGEN_END(DTO)
} // namespace server::face
#endif // CRUD_STATUSDTO_HPP
