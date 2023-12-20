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

    builder.setTitle("Face service's API")
        .setDescription(
            "该页面描述了人脸库服务API的使用方法。通过这些API，您可以在人脸库中"
            "创建、更新、删除和搜索用户，同时提供了一系列人脸相关的算法任务。每"
            "个API调用都会返回一个JSON格式的响应。")
        .setVersion("1.0")
        .setContactName("Sinter")
        .setContactUrl("http://localhost:9797")
        .setContactEmail("sinterwong@gmail.com")

        // .setLicenseName("Apache License, Version 2.0")
        // .setLicenseUrl("http://www.apache.org/licenses/LICENSE-2.0")

        .addServer("http://localhost:9797", "face server on x3pi");

    return builder.build();
  }());

  /**
   *  Swagger-Ui Resources (<oatpp-examples>/lib/oatpp-swagger/res)
   */
  OATPP_CREATE_COMPONENT(std::shared_ptr<oatpp::swagger::Resources>,
                         swaggerResources)
  ([] {
    // Make sure to specify correct full path to oatpp-swagger/res folder !!!
    return oatpp::swagger::Resources::loadResources("/opt/deploy/swagger_res");
  }());
};
} // namespace server::face
#endif /* SwaggerComponent_hpp */