CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
PROJECT(FlowEngineSrc)

LOAD_OATPP()

LINK_LIBRARIES(
    gflags
    oatpp::oatpp
    oatpp::oatpp-swagger
)

# submodule
ADD_SUBDIRECTORY(face)
