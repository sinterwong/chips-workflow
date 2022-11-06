CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
PROJECT(Tests)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=deprecated-declarations -Wno-error=unused-function")
INCLUDE_DIRECTORIES(
    ${PROJECT_INCLUDE_DIR}
    ${PROJECT_INCLUDE_DIR}/core
    ${PROJECT_INCLUDE_DIR}/infer
    ${COMMON_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${Protobuf_INCLUDE_DIR}
    ${OPENSSL_INCLUDE_DIR}
    ${CURL_INCLUDE_DIR}
    ${DNN_PATH}/include
    ${APPSDK_PATH}/include
    ${APPSDK_PATH}/include/vio
    ${APPSDK_PATH}/include/libmm
)

LINK_DIRECTORIES(
    ${OpenCV_LIBRARY_DIR}
    ${CURL_LIBRARY_DIR}
    ${OPENSSL_LIBRARY_DIR}
    ${DNN_LIB_PATH}
    ${APPSDK_PATH}/lib
    ${APPSDK_PATH}/lib/hbbpu
    ${APPSDK_PATH}/lib/hbmedia
)
SET(FE_LIBS core license module module_logger infer)
IF(TARGET_HARDWARE STREQUAL "x3")
    MESSAGE(STATUS "Loading X3 dependent libraries")
    SET(CHIPS_DEPAND_LIBS x3_infer x3_module ${BPU_libs} ${HB_MEDIA_libs} rt dl)
ELSEIF(TARGET_HARDWARE EQUAL "jetson")
    MESSAGE(STATUS "Loading Jetson dependent libraries")
    SET(CHIPS_DEPAND_LIBS jetson_infer jetson_infer_cu jetson_module nvinfer nvparsers nvonnxparser nvinfer_plugin)
ELSE()
    SET(CHIPS_DEPAND_LIBS)
ENDIF()
LIST(APPEND FE_LIBS ${CHIPS_DEPAND_LIBS})

LINK_LIBRARIES(
    ${OpenCV_LIBS}
    ${Protobuf_LIBRARY}
    ${CURL_LIBS}
    ${OPENSSL_LIBS}
    ${FE_LIBS}
    gflags::gflags
    pthread
    spdlog
)

FILE(GLOB APP_SOURCES *.cpp)
FILE(GLOB APP_HARDWARE_SOURCES ${TARGET_HARDWARE}/*.cpp)
LIST(APPEND APP_SOURCES ${APP_HARDWARE_SOURCES})
MESSAGE(STATUS "APP SOURCES: ${APP_SOURCES}")
FOREACH(sourcefile ${APP_SOURCES})
    STRING(REGEX MATCH "[^/]+$" sourcefilewithoutpath ${sourcefile})
    STRING(REPLACE ".cpp" "" testname ${sourcefilewithoutpath})
    ADD_EXECUTABLE( ${testname} ${sourcefile})
    SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin/tests)

    INSTALL(TARGETS ${testname}
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)
ENDFOREACH(sourcefile ${APP_SOURCES})
