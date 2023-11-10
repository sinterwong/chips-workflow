/**
 * @file faceService.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-10-23
 *
 * @copyright Copyright (c) 2023
 *
 */
#include "AppComponent.hpp"
#include "FacelibDto.hpp"
#include "StatusDto.hpp"
#include "algoManager.hpp"
#include "faceLibManager.hpp"
#include "logger/logger.hpp"
#include "preprocess.hpp"
#include <cstddef>
#include <oatpp/web/protocol/http/Http.hpp>
#include <string>
#include <vector>

#ifndef __CRUD_FACE_SERVICE_HPP_
#define __CRUD_FACE_SERVICE_HPP_
namespace server::face {

inline std::string joinIds(const std::vector<long> &ids,
                           const std::string &separator = ", ") {
  std::ostringstream stream;
  for (size_t i = 0; i < ids.size(); ++i) {
    stream << ids[i];
    if (i < ids.size() - 1) {
      stream << separator;
    }
  }
  return stream.str();
}

class FaceService {
private:
  using Status = oatpp::web::protocol::http::Status;

  // 辅助函数，处理批处理操作中的失败和错误ID，减少重复代码
  void handleBatchErrors(const std::vector<long> &failed_ids,
                         const std::vector<long> &err_ids,
                         const oatpp::Object<StatusDto> &status);

  // 辅助函数，批量算法调用，减少重复代码
  void batchInfer(const std::vector<std::string> &urls,
                  std::vector<float *> &vecs,
                  std::vector<std::vector<float>> &features,
                  std::vector<long> &failed_ids);

public:
  // 新增单个人脸
  oatpp::Object<StatusDto> createUser(oatpp::Int32 const &id,
                                      oatpp::String const &url);
  // 更新
  oatpp::Object<StatusDto> updateUser(oatpp::Int32 const &id,
                                      oatpp::String const &url);

  // 删除
  oatpp::Object<StatusDto> deleteUser(oatpp::Int32 const &id);

  // 通过ID查询
  oatpp::Object<StatusDto> searchUser(oatpp::String const &url);

  // 两图比对
  oatpp::Object<StatusDto> compareTwoPictures(oatpp::String const &url1,
                                              oatpp::String const &url2);

  // 批量新增
  oatpp::Object<StatusDto> createBatch(const oatpp::Object<FacelibDto> &users);

  // 批量更新
  oatpp::Object<StatusDto> updateBatch(oatpp::Object<FacelibDto> const &users);

  // 批量删除
  oatpp::Object<StatusDto> deleteBatch(oatpp::Object<FacelibDto> const &users);
};
} // namespace server::face
#endif