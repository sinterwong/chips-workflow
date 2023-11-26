#ifndef __CRUD_STATUS_DTO_HPP_
#define __CRUD_STATUS_DTO_HPP_

#include "oatpp/core/Types.hpp"
#include "oatpp/core/macro/codegen.hpp"
namespace server::face {
#include OATPP_CODEGEN_BEGIN(DTO)

class StatusDto : public oatpp::DTO {

  DTO_INIT(StatusDto, DTO)

  DTO_FIELD_INFO(status) { info->description = "Short status text"; }
  DTO_FIELD(String, status);

  DTO_FIELD_INFO(code) { info->description = "Status code"; }
  DTO_FIELD(Int32, code);

  DTO_FIELD_INFO(message) { info->description = "Verbose message"; }
  DTO_FIELD(String, message);

public:
  std::string toString() const {
    auto str = "StatusDto(\n"
               "  status: " +
               (status ? *status : "null") + ",\n" +
               "  code: " + std::to_string(*code) + ",\n" +
               "  message: " + (message ? *message : "null") + "\n" + ")";
    return str;
  }
};

#include OATPP_CODEGEN_END(DTO)
} // namespace server::face
#endif // CRUD_STATUSDTO_HPP
