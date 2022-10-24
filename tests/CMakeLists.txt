CMAKE_MINIMUM_REQUIRED(VERSION 3.1)
PROJECT(Tests)

INCLUDE_DIRECTORIES(
    ${PROJECT_INCLUDE_DIR}
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
    ${COMMON_LIBRARY_DIR}
    ${OpenCV_LIBRARY_DIR}
    ${CURL_LIBRARY_DIR}
    ${OPENSSL_LIBRARY_DIR}
    ${DNN_LIB_PATH}
    ${APPSDK_PATH}/lib
    ${APPSDK_PATH}/lib/hbbpu
    ${APPSDK_PATH}/lib/hbmedia
)

SET(JETSON_LIBS jetson_infer jetson_infer_cu jetson_module nvinfer nvparsers nvonnxparser nvinfer_plugin)
SET(X3_LIBS x3_infer x3_module)

LINK_LIBRARIES(
    ${OpenCV_LIBS}
    ${Protobuf_LIBRARY}
    ${CURL_LIBS}
    ${OPENSSL_LIBS}
    gflags::gflags
    pthread
    spdlog
    core
    module
    module_logger 
    license
    ${X3_LIBS}
    ${JETSON_LIBS}
)

FILE(GLOB APP_SOURCES *.cpp)
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
