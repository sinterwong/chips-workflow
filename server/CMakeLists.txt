CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
PROJECT(FlowEngineSrc)

LOAD_OATPP()
LOAD_OATPP_SQLITE()

LINK_LIBRARIES(
    gflags
    oatpp::oatpp
    oatpp::oatpp-sqlite
    ${SQLITE_LIBS}
)

# submodule
ADD_SUBDIRECTORY(face)
