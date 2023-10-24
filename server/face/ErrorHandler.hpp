#ifndef CRUD_ERRORHANDLER_HPP
#define CRUD_ERRORHANDLER_HPP

#include "StatusDto.hpp"
#include "oatpp/web/protocol/http/outgoing/ResponseFactory.hpp"
#include "oatpp/web/server/handler/ErrorHandler.hpp"
namespace server::face {
class ErrorHandler : public oatpp::web::server::handler::ErrorHandler {
private:
  using OutgoingResponse = oatpp::web::protocol::http::outgoing::Response;
  using Status = oatpp::web::protocol::http::Status;
  using ResponseFactory = oatpp::web::protocol::http::outgoing::ResponseFactory;

private:
  std::shared_ptr<oatpp::data::mapping::ObjectMapper> m_objectMapper;

public:
  ErrorHandler(
      const std::shared_ptr<oatpp::data::mapping::ObjectMapper> &objectMapper)
      : m_objectMapper(objectMapper) {}

  std::shared_ptr<OutgoingResponse>
  handleError(const Status &status, const oatpp::String &message,
              const Headers &headers) override {
    auto error = StatusDto::createShared();
    error->status = "ERROR";
    error->code = status.code;
    error->message = message;

    auto response =
        ResponseFactory::createResponse(status, error, m_objectMapper);

    for (const auto &pair : headers.getAll()) {
      response->putHeader(pair.first.toString(), pair.second.toString());
    }

    return response;
  }
};
} // namespace server::face
#endif // CRUD_ERRORHANDLER_HPP