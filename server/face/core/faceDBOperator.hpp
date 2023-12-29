/**
 * @file faceDBOperator.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-12-29
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef __FACE_DB_OPERATOR_H_
#define __FACE_DB_OPERATOR_H_

#include "AppComponent.hpp"
#include "UserDb.hpp"
#include <mutex>
#include <oatpp/web/protocol/http/Http.hpp>

namespace server::face::core {
class FaceDBOperator {
public:
  static FaceDBOperator &getInstance() {
    static std::once_flag onceFlag;
    std::call_once(onceFlag, [] { instance = new FaceDBOperator(); });
    return *instance;
  }
  FaceDBOperator(FaceDBOperator const &) = delete;
  FaceDBOperator &operator=(FaceDBOperator const &) = delete;

private:
  FaceDBOperator() {}
  ~FaceDBOperator() {
    delete instance;
    instance = nullptr;
  }
  static FaceDBOperator *instance;

public:
  // 从用户列表中提取id和特征向量
  void getIdsAndFeatures(std::vector<long> &ids,
                         std::vector<std::vector<float>> &features,
                         oatpp::Vector<oatpp::Object<UserDto>> const &users);

  // 恢复人脸库
  bool restoreFacelib(std::string const &libName,
                      oatpp::Object<StatusDto> &status,
                      bool needCreate = false);
  // 获取所有人脸
  oatpp::Vector<oatpp::Object<UserDto>>
  getAllUsers(oatpp::provider::ResourceHandle<oatpp::orm::Connection> const
                  &connection = nullptr);

  // 通过身份证号查询
  oatpp::Int32
  getIdByIdNumber(oatpp::String const &idNumber,
                  oatpp::Object<StatusDto> &status,
                  oatpp::provider::ResourceHandle<oatpp::orm::Connection> const
                      &connection = nullptr);

  // 通过id查询idNumber
  oatpp::String
  getIdNumberById(oatpp::Int32 const &id, oatpp::Object<StatusDto> &status,
                  oatpp::provider::ResourceHandle<oatpp::orm::Connection> const
                      &connection = nullptr);

  // 通过id查询libName
  oatpp::String
  getLibNameById(oatpp::Int32 const &id, oatpp::Object<StatusDto> &status,
                 oatpp::provider::ResourceHandle<oatpp::orm::Connection> const
                     &connection = nullptr);

  // 查询某libName中的所有用户
  oatpp::Vector<oatpp::Object<UserDto>> getUsersByLibName(
      oatpp::String const &libName, oatpp::Object<StatusDto> &status,
      oatpp::provider::ResourceHandle<oatpp::orm::Connection> const
          &connection = nullptr);

  // 通过libNumber查询libName
  oatpp::String getLibNameByIdNumber(
      oatpp::String const &idNumber, oatpp::Object<StatusDto> &status,
      oatpp::provider::ResourceHandle<oatpp::orm::Connection> const
          &connection = nullptr);

  // 新增人脸到数据库并返回id
  oatpp::Int32
  insertUser(std::string const &idNumber, std::string const &libName,
             std::string const &feature, oatpp::Object<StatusDto> &status,
             oatpp::provider::ResourceHandle<oatpp::orm::Connection> const
                 &connection = nullptr);

  // 更新人脸到数据库
  oatpp::Int32 updateUserByIdNumber(
      std::string const &idNumber, std::string const &libName,
      std::string const &feature, oatpp::Object<StatusDto> &status,
      oatpp::provider::ResourceHandle<oatpp::orm::Connection> const
          &connection = nullptr);

  // 删除人脸
  oatpp::Int32 deleteUserByIdNumber(
      std::string const &idNumber, oatpp::Object<StatusDto> &status,
      oatpp::provider::ResourceHandle<oatpp::orm::Connection> const
          &connection = nullptr);

  // 执行sql语句
  std::shared_ptr<oatpp::orm::QueryResult>
  executeSql(oatpp::String const &sql,
             oatpp::provider::ResourceHandle<oatpp::orm::Connection> const
                 &connection = nullptr);

private:
  using Status = oatpp::web::protocol::http::Status;
  OATPP_COMPONENT(std::shared_ptr<UserDb>, m_database);
};

} // namespace server::face::core

#endif /* __FACE_DB_OPERATOR_H_ */