#ifndef SwaggerComponent_hpp
#define SwaggerComponent_hpp

#include <oatpp-swagger/Model.hpp>
#include <oatpp-swagger/Resources.hpp>
#include <oatpp/core/macro/component.hpp>
namespace server::face {
/**
 *  Swagger ui is served at
 *  http://host:port/swagger/ui
 */
class SwaggerComponent {
public:
  /**
   *  General API docs info
   */
  OATPP_CREATE_COMPONENT(std::shared_ptr<oatpp::swagger::DocumentInfo>,
                         swaggerDocumentInfo)
  ([] {
    oatpp::swagger::DocumentInfo::Builder builder;

    builder.setTitle("User entity service")
        .setDescription("Face server API")
        .setVersion("1.0")
        .setContactName("Sinter")
        .setContactUrl("http://114.242.23.39:9797")

        .setLicenseName("Apache License, Version 2.0")
        .setLicenseUrl("http://www.apache.org/licenses/LICENSE-2.0")

        .addServer("http://114.242.23.39:9797", "server on x3pi");

    return builder.build();
  }());

  /**
   *  Swagger-Ui Resources (<oatpp-examples>/lib/oatpp-swagger/res)
   */
  OATPP_CREATE_COMPONENT(std::shared_ptr<oatpp::swagger::Resources>,
                         swaggerResources)
  ([] {
    // Make sure to specify correct full path to oatpp-swagger/res folder !!!
    return oatpp::swagger::Resources::loadResources(
        "/root/workspace/projects/oatpp-swagger/res");
  }());
};
} // namespace server::face
#endif /* SwaggerComponent_hpp */