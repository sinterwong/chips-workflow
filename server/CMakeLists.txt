CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
PROJECT(FlowEngineSrc)

LOAD_OATPP()
LOAD_OATPP_SQLITE()
LOAD_OATPP_SWAGGER()

LINK_LIBRARIES(
    gflags
    oatpp::oatpp
    oatpp::oatpp-swagger
    oatpp::oatpp-sqlite
)

# submodule
ADD_SUBDIRECTORY(face)
