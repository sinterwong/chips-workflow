/**
 * @file statusCode.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-11-06
 *
 * @copyright Copyright (c) 2023
 *
 */

#include <string>
#include <unordered_map>

#ifndef __SERVER_FACE_STATUS_CODE_HPP_
#define __SERVER_FACE_STATUS_CODE_HPP_

namespace server::face {

// X宏技巧，这里指定以宏的调用而不管逻辑
#define STATUS_CODES                                                           \
  X(Ok, "Operation completed successfully.")                                   \
  X(Error, "An error has occurred.")                                           \
  X(NotFound, "The requested item was not found.")                             \
  X(NoPermission, "You do not have permission to perform this operation.")     \
  X(InvalidInput, "The input provided was invalid.")

// define the enum of status code
enum class StatusCode {
#define X(Enum, String) Enum,
  STATUS_CODES
#undef X
};

const std::unordered_map<StatusCode, std::string> statusCodeMessages = {
#define X(Enum, String) {StatusCode::Enum, String},
    STATUS_CODES
#undef X
};

} // namespace server::face
#endif