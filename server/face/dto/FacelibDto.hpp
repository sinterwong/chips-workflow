
#ifndef __CRUD_FACELIB_DTO_HPP_
#define __CRUD_FACELIB_DTO_HPP_

#include "oatpp/core/Types.hpp"
#include "oatpp/core/macro/codegen.hpp"
#include "oatpp/core/utils/ConversionUtils.hpp"

namespace server::face {

#include OATPP_CODEGEN_BEGIN(DTO)

class FacelibDto : public oatpp::DTO {

  DTO_INIT(FacelibDto, DTO)

  DTO_FIELD(Vector<Int32>, ids);
  DTO_FIELD(Vector<String>, urls);

public:
  std::string toString() const {

    std::string idsStr = "null";
    if (ids) {
      idsStr = "";
      for (const auto &id : *ids) {
        idsStr += std::to_string(id) + ", ";
      }
      if (!idsStr.empty()) {
        idsStr.pop_back(); // remove last space
        idsStr.pop_back(); // remove last comma
      }
    }

    std::string urlsStr =
        urls ? (!urls->empty() ? urls->at(0)->c_str() : "null") : "null";

    auto str = "FacelibDto(\n"
               "  ids: " +
               idsStr + ",\n" + "  urls: " + urlsStr + "\n" + ")";
    return str;
  }
};

#include OATPP_CODEGEN_END(DTO)
} // namespace server::face
#endif // CRUD_STATUSDTO_HPP
