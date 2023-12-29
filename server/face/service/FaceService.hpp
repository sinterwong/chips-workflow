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
#include "FaceDto.hpp"
#include "ImageDto.hpp"
#include "StatusDto.hpp"
#include "UserDto.hpp"
#include "faceLibManager.hpp"
#include "logger/logger.hpp"
#include "networkUtils.hpp"
#include "preprocess.hpp"
#include <cstddef>
#include <memory>
#include <opencv2/imgcodecs.hpp>
#include <string>
#include <vector>

#include <oatpp/web/protocol/http/Http.hpp>

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

inline std::string joinIdNumbers(const std::vector<std::string> &idNumbers,
                                 const std::string &separator = ", ") {
  std::ostringstream stream;
  for (size_t i = 0; i < idNumbers.size(); ++i) {
    stream << idNumbers[i];
    if (i < idNumbers.size() - 1) {
      stream << separator;
    }
  }
  return stream.str();
}

const static std::string backupImageDir = "/opt/deploy/backup";
inline void backupImage(oatpp::Int32 const &idx, std::string const &libName,
                        std::string const &idNumber, std::string const &url) {
  std::string fileName =
      std::to_string(idx) + "_" + libName + "_" + idNumber + ".jpg";
  cv::imwrite(backupImageDir + "/" + fileName, *getImageByUri(url));
}

class FaceService {
private:
  // 辅助函数，处理批处理操作中的失败和错误ID，减少重复代码
  void handleBatchErrors(std::vector<std::string> const &errIdNumbersAlgo,
                         std::vector<std::string> const &errIdNumbersDB,
                         const oatpp::Object<StatusDto> &status);

  // 辅助函数，批量算法调用，减少重复代码
  void batchInfer(const std::vector<std::string> &urls,
                  std::vector<std::string> const &idNumbers,
                  std::vector<std::vector<float>> &features,
                  std::vector<std::string> &errIdNumbers);

  // 提取特征向量
  bool extractFeature(std::string const &url, std::vector<float> &feature);

  // 特征转base64
  std::string feature2base64(std::vector<float> &feature);



public:
  // Get 新增单个人脸
  oatpp::Object<StatusDto> createUser(oatpp::String const &idNumber,
                                      oatpp::String const &libName,
                                      oatpp::String const &url);
  // Post 新增单个人脸
  oatpp::Object<StatusDto> createUser(oatpp::Object<FaceDto> const &user);

  // Get 更新
  oatpp::Object<StatusDto> updateUser(oatpp::String const &idNumber,
                                      oatpp::String const &libName,
                                      oatpp::String const &url);
  // Post 更新
  oatpp::Object<StatusDto> updateUser(oatpp::Object<FaceDto> const &user);

  // 删除
  oatpp::Object<StatusDto> deleteUser(oatpp::String const &idNumber);

  // Get 通过图片查询
  oatpp::Object<StatusDto> searchUser(oatpp::String const &libName,
                                      oatpp::String const &url);

  // Post 通过图片查询
  oatpp::Object<StatusDto> searchUser(oatpp::Object<ImageDto> const &images);

  // Get 两图比对
  oatpp::Object<StatusDto> compareTwoPictures(oatpp::String const &url1,
                                              oatpp::String const &url2);

  // Post 两图比对
  oatpp::Object<StatusDto>
  compareTwoPictures(oatpp::Vector<oatpp::Object<ImageDto>> const &images);

  // 批量新增
  oatpp::Object<StatusDto>
  createBatch(oatpp::Vector<oatpp::Object<FaceDto>> const &users);

  // 批量更新
  oatpp::Object<StatusDto>
  updateBatch(oatpp::Vector<oatpp::Object<FaceDto>> const &users);

  // 批量删除
  oatpp::Object<StatusDto>
  deleteBatch(oatpp::Vector<oatpp::Object<FaceDto>> const &users);

  // Get 人脸质检
  oatpp::Object<StatusDto> faceQuality(oatpp::String const &url);

  // Post 人脸质检
  oatpp::Object<StatusDto> faceQuality(oatpp::Object<ImageDto> const &image);
};
} // namespace server::face
#endif