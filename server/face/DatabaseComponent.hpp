/**
 * @file DatabaseComponent.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-11-13
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef __SERVER_FACE_DATABASE_COMPONENT_HPP_
#define __SERVER_FACE_DATABASE_COMPONENT_HPP_

#include "UserDb.hpp"
#include "oatpp/core/macro/component.hpp"
#include <memory>

namespace server::face {

class DatabaseComponent {
public:
  /**
   * Create database connection provider component
   */
  OATPP_CREATE_COMPONENT(
      std::shared_ptr<oatpp::provider::Provider<oatpp::sqlite::Connection>>,
      dbConnectionProvider)
  ([] {
    /* Create database-specific ConnectionProvider */
    auto connectionProvider =
        std::make_shared<oatpp::sqlite::ConnectionProvider>(
            "/opt/deploy/db/user.db");

    /* Create database-specific ConnectionPool */
    return oatpp::sqlite::ConnectionPool::createShared(
        connectionProvider, 10 /* max-connections */,
        std::chrono::seconds(5) /* connection TTL */);
  }());

  /**
   * Create database client
   */
  OATPP_CREATE_COMPONENT(std::shared_ptr<UserDb>, userDb)
  ([] {
    /* Get database ConnectionProvider component */
    OATPP_COMPONENT(
        std::shared_ptr<oatpp::provider::Provider<oatpp::sqlite::Connection>>,
        connectionProvider);

    /* Create database-specific Executor */
    auto executor =
        std::make_shared<oatpp::sqlite::Executor>(connectionProvider);

    /* Create MyClient database client */
    return std::make_shared<UserDb>(executor);
  }());
};

} // namespace server::face

#endif