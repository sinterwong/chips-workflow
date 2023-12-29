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

  DTO_FIELD_INFO(libName) { info->description = "The name of face library"; }
  DTO_FIELD(String, libName);

  DTO_FIELD_INFO(url) { info->description = "The url of stream"; }
  DTO_FIELD(String, url);

  DTO_FIELD_INFO(interfaceUrl) {
    info->description = "Results are sent to this url";
  }
  DTO_FIELD(String, interfaceUrl);

public:
  std::string toString() const {
    return "StreamDto: { name = " + name + ", libName = " + libName +
           ", url = " + url + ", interfaceUrl = " + interfaceUrl + " }";
  }
};

#include OATPP_CODEGEN_END(DTO)
} // namespace server::face
#endif // CRUD_STATUSDTO_HPP
