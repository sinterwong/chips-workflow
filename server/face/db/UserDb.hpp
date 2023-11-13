/**
 * @file UserDb.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-11-09
 *
 * @copyright Copyright (c) 2023
 *
 */
#ifndef __SERVER_FACE_DB_USER_DB_HPP_
#define __SERVER_FACE_DB_USER_DB_HPP_

#include "UserDto.hpp"
#include <memory>
#include <oatpp-sqlite/orm.hpp>
/**
 * @brief UserDb client definitions
 *
 */
namespace server::face {

#include OATPP_CODEGEN_BEGIN(DbClient) //<- Begin Codegen

class UserDb : public oatpp::orm::DbClient {
public:
  UserDb(std::shared_ptr<oatpp::orm::Executor> const &executor)
      : oatpp::orm::DbClient(executor) {

    oatpp::orm::SchemaMigration migration(executor);
    migration.addFile(1 /* start from version 1 */,
                      "/opt/deploy/sql/001_init.sql");
    // TODO - Add more migrations here.
    migration.migrate(); // <-- run migrations. This guy will throw on error.

    auto version = executor->getSchemaVersion();
    OATPP_LOGD("UserDb", "Migration - OK. Version=%ld.", version);
  }

  // query是一个宏，用于定义一个函数，函数名为第一个参数，第二个参数为SQL语句，第三个参数为SQL语句中的参数
  QUERY(createUser,
        "INSERT INTO AppUser(idNumber, feature) VALUES (:user.idNumber, "
        ":user.feature);",
        PARAM(oatpp::Object<UserDto>, user))

  QUERY(getUserById, "SELECT * FROM AppUser WHERE id=:id;",
        PARAM(oatpp::Int32, id))

  QUERY(getIdByIdNumber, "SELECT id FROM AppUser WHERE idNumber=:idNumber;",
        PARAM(oatpp::String, idNumber))

  QUERY(getUserIdNumberById, "SELECT idNumber FROM AppUser WHERE id=:id;",
        PARAM(oatpp::Int32, id))

  QUERY(getAllUsers, "SELECT * FROM AppUser;")

  QUERY(updateUserById,
        "UPDATE AppUser SET idNumber=:user.idNumber, feature=:user.feature "
        "WHERE id=:id;",
        PARAM(oatpp::Int32, id), PARAM(oatpp::Object<UserDto>, user))

  QUERY(updateUserByIdNumber,
        "UPDATE AppUser SET idNumber=:user.idNumber, feature=:user.feature "
        "WHERE idNumber=:idNumber;",
        PARAM(oatpp::String, idNumber), PARAM(oatpp::Object<UserDto>, user))

  QUERY(deleteUserById, "DELETE FROM AppUser WHERE id=:id;",
        PARAM(oatpp::Int32, id))

  QUERY(deleteUserByIdNumber, "DELETE FROM AppUser WHERE idNumber=:idNumber;",
        PARAM(oatpp::String, idNumber))

  QUERY(getFeaturesOfAllUsers,
        "SELECT id, feature FROM AppUser WHERE feature IS NOT NULL;")
};

#include OATPP_CODEGEN_END(DbClient) //<- End Codegen
} // namespace server::face

#endif