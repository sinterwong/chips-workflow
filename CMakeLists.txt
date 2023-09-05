CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
SET(CMAKE_EXPORT_COMPILE_COMMANDS 1)

# --------------------------------------------
# Project setting
# --------------------------------------------
INCLUDE(cmake/messagecolor.cmake)
INCLUDE(cmake/utils.cmake)
FLOWENGINE_APP_EXTRACT_VERSION()
PROJECT(FlowEngine VERSION ${FLOWENGINE_APP_VERSION} LANGUAGES C CXX)

OPTION(BUILD_TESTS "Build with tests" ON)
OPTION(BUILD_APP "Build with application" ON)
MESSAGE(INFO "--------------------------------")
MESSAGE(STATUS "Build FlowEngine: ${FLOWENGINE_APP_VERSION}")
MESSAGE(STATUS "Build with tests: ${BUILD_TESTS}")
MESSAGE(STATUS "Build with application: ${BUILD_APP}")
MESSAGE(STATUS "CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")

IF(CMAKE_TOOLCHAIN_FILE)
    MESSAGE(STATUS "CMAKE_TOOLCHAIN_FILE: ${CMAKE_TOOLCHAIN_FILE}")
ENDIF()

# --------------------------------------------
# Set default build to debug
# --------------------------------------------
IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose Release, Debug, RelWithDeblnfo or MinSizeRel" FORCE)
ENDIF()

IF(NOT TARGET_OS)
    SET(TARGET_OS linux)
ENDIF()

IF(NOT TARGET_ARCH)
    SET(TARGET_ARCH aarch64)
ENDIF()

IF(NOT TARGET_HARDWARE)
    SET(TARGET_HARDWARE none) # [none/x3/jetson]
ENDIF()

SET(PROJECT_OUTPUT_DIR ${PROJECT_BINARY_DIR}/${TARGET_ARCH})
SET(PROJECT_INCLUDE_DIR ${PROJECT_OUTPUT_DIR}/include)
FILE(MAKE_DIRECTORY ${PROJECT_INCLUDE_DIR})
FILE(MAKE_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin)
MESSAGE("-- system arch:  ${CMAKE_SYSTEM_PROCESSOR}")
MESSAGE("-- output path:  ${PROJECT_OUTPUT_DIR}")
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin)
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/lib)
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/lib)

MESSAGE(STATUS "Your target os : ${TARGET_OS}")
MESSAGE(STATUS "Your target arch : ${TARGET_ARCH}")
MESSAGE(STATUS "Your target hardware : ${TARGET_HARDWARE}")

MESSAGE(STATUS "Build type: " ${CMAKE_BUILD_TYPE})

# --------------------------------------------
# Set compilation options
# --------------------------------------------
MESSAGE(INFO "--------------------------------")
INCLUDE(cmake/toolchain_options.cmake)
INCLUDE(cmake/load_3rdparty.cmake)

MESSAGE(INFO "Loading 3rdparty libraries")

IF(TARGET_HARDWARE STREQUAL "x3")
    MESSAGE(STATUS "Loading X3 dependent libraries")
    LOAD_X3()
    ADD_DEFINITIONS(-DDECODER_POOL_SIZE=6)
ELSEIF(TARGET_HARDWARE STREQUAL "jetson")
    MESSAGE(STATUS "Loading Jetson dependent libraries")
    LOAD_Jetson()
    ADD_DEFINITIONS(-DDECODER_POOL_SIZE=1)
ELSEIF(TARGET_HARDWARE STREQUAL "rockchip")
    MESSAGE(STATUS "Loading Rockchip dependent libraries")
    LOAD_ROCKCHIP()
ELSE()
    IF(WITH_MNN_ENGINE)
        MESSAGE(INFO "--------------------------------")
        MESSAGE(STATUS "Using MNN: on")
        LOAD_MNN()
    ENDIF()

    MESSAGE(WARNING "No TARGET_HARDWARE specified!")
ENDIF()

MESSAGE(INFO "3rdparty libraries has loaded.")

# --------------------------------------------
# Build 3rdparty
# --------------------------------------------
MESSAGE(INFO "--------------------------------")

# --------------------------------------------
ADD_SUBDIRECTORY(3rdparty)

# Build application
# --------------------------------------------
MESSAGE(INFO "--------------------------------")
ADD_SUBDIRECTORY(src)

IF(BUILD_TESTS)
    MESSAGE(INFO "--------------------------------")
    ADD_SUBDIRECTORY(tests)
ENDIF()

IF(BUILD_APP)
    MESSAGE(INFO "--------------------------------")
    ADD_SUBDIRECTORY(app)
ENDIF()