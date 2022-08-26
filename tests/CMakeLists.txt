CMAKE_MINIMUM_REQUIRED(VERSION 3.1)
PROJECT(Tests)

INCLUDE_DIRECTORIES(
    ${PROJECT_INCLUDE_DIR}
    ${COMMON_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${Protobuf_INCLUDE_DIR}
    ${OPENSSL_INCLUDE_DIR}
    ${CURL_INCLUDE_DIR}
)

LINK_DIRECTORIES(
    ${COMMON_LIBRARY_DIR}
    ${OpenCV_LIBRARY_DIR}
    ${CURL_LIBRARY_DIR}
    ${OPENSSL_LIBRARY_DIR}
)

LINK_LIBRARIES(
    ${OpenCV_LIBS}
    ${Protobuf_LIBRARY}
    ${CURL_LIBS}
    ${OPENSSL_LIBS}
    gflags::gflags
    pthread
    spdlog
    core
    infer
    infer_cu
    module
    module_logger 
    license
    nvinfer 
    nvparsers 
    nvonnxparser 
    nvinfer_plugin
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
