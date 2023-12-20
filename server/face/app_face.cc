/**
 * @file app_face.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-10-20
 *
 * @copyright Copyright (c) 2023
 *
 */
#include "oatpp-swagger/Controller.hpp"
#include <iostream>
#include <memory>
#include <oatpp/core/Types.hpp>
#include <oatpp/core/macro/codegen.hpp>
#include <oatpp/network/Server.hpp>
#include <oatpp/parser/json/mapping/ObjectMapper.hpp>
#include <oatpp/web/server/HttpConnectionHandler.hpp>
#include <oatpp/web/server/HttpRouter.hpp>

#include "AppComponent.hpp"
#include "FaceController.hpp"
#include "StaticController.hpp"
#include "VideoController.hpp"

namespace server::face {
const auto initLogger = []() -> decltype(auto) {
  FlowEngineLoggerInit(true, true, true, true);
  return true;
}();

void run() {
  FlowEngineLoggerSetLevel(1);

  AppComponent components;

  // Get router component
  OATPP_COMPONENT(std::shared_ptr<oatpp::web::server::HttpRouter>, router);

  oatpp::web::server::api::Endpoints docEndpoints;

  docEndpoints.append(
      router->addController(FaceController::createShared())->getEndpoints());

  docEndpoints.append(
      router->addController(VideoController::createShared())->getEndpoints());

  router->addController(oatpp::swagger::Controller::createShared(docEndpoints));
  router->addController(StaticController::createShared());

  // Get connection handler component
  OATPP_COMPONENT(std::shared_ptr<oatpp::network::ConnectionHandler>,
                  connectionHandler);

  // Get connection provider component
  OATPP_COMPONENT(std::shared_ptr<oatpp::network::ServerConnectionProvider>,
                  connectionProvider);

  // create server
  oatpp::network::Server server(connectionProvider, connectionHandler);

  OATPP_LOGD("Server", "Running on port %s...",
             connectionProvider->getProperty("port").toString()->c_str());
  server.run();

  /* stop db connection pool */
  OATPP_COMPONENT(
      std::shared_ptr<oatpp::provider::Provider<oatpp::sqlite::Connection>>,
      dbConnectionProvider);
  dbConnectionProvider->stop();
}
} // namespace server::face

int main(int argc, const char *argv[]) {

  FlowEngineLoggerSetLevel(2);

  oatpp::base::Environment::init();

  server::face::run();

  /* Print how many objects were created during app running, and what have
   * left-probably leaked */
  /* Disable object counting for release builds using '-D
   * OATPP_DISABLE_ENV_OBJECT_COUNTERS' flag for better performance */
  std::cout << "\nEnvironment:\n";
  std::cout << "objectsCount = " << oatpp::base::Environment::getObjectsCount()
            << "\n";
  std::cout << "objectsCreated = "
            << oatpp::base::Environment::getObjectsCreated() << "\n\n";

  oatpp::base::Environment::destroy();

  return 0;
}