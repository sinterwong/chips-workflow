CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
PROJECT(FlowEngine3rdparty)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=deprecated-declarations")

# NOTE: build with source
SET(SPDLOG_BUILD_SHARED ON)
ADD_SUBDIRECTORY(spdlog)

ADD_SUBDIRECTORY(gflags)

ADD_SUBDIRECTORY(eigen)

FILE(MAKE_DIRECTORY ${PROJECT_INCLUDE_DIR})

# Copy nlohmann to ${PROJECT_INCLUDE_DIR}
FILE(COPY nlohmann DESTINATION ${PROJECT_INCLUDE_DIR})

# Copy crow to ${PROJECT_INCLUDE_DIR}
FILE(COPY crow DESTINATION ${PROJECT_INCLUDE_DIR})

# MESSAGE(INFO "--------------------------------")
# SET(NN_STATIC_LIB ON CACHE BOOL "enable static library building of nanomsg")
# SET(NN_TESTS OFF CACHE BOOL "disable test building of nanomsg")
# SET(NN_TOOLS OFF CACHE BOOL "disable tool building of nanomsg")
# SET(NN_ENABLE_DOC OFF CACHE BOOL "disable doc building of nanomsg")
# ADD_SUBDIRECTORY(nanomsg)
