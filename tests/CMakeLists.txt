CMAKE_MINIMUM_REQUIRED(VERSION 3.1)
PROJECT(Tests)

INCLUDE_DIRECTORIES(
    ${PROJECT_INCLUDE_DIR}
    ${PROJECT_INCLUDE_DIR}/core
    ${PROJECT_INCLUDE_DIR}/infer
    ${PROJECT_INCLUDE_DIR}/module
    ${3RDPARTY_DIR}/spdlog/include
    ${COMMON_INCLUDE_DIR}
    ${FREETYPE_INCLUDE_DIR}
    ${FREETYPE_INCLUDE_DIR}/freetype2
    ${OpenCV_INCLUDE_DIRS}
    ${Protobuf_INCLUDE_DIR}
    ${OPENSSL_INCLUDE_DIR}
    ${CURL_INCLUDE_DIR}
    ${FFMPEG_INCLUDE_DIR}
)

LINK_DIRECTORIES(
    ${COMMON_LIBRARY_DIR}
    ${FREETYPE_LIBRARY_DIR}
    ${OpenCV_LIBRARY_DIR}
    ${CURL_LIBRARY_DIR}
    ${OPENSSL_LIBRARY_DIR}
    ${FFMPEG_LIBRARY_DIR}
)

LINK_LIBRARIES(
    ${FREETYPE_LIBS}
    ${OpenCV_LIBS}
    ${Protobuf_LIBRARY}
    ${CURL_LIBS}
    ${FFMPEG_LIBS}
    ${OPENSSL_LIBS}
    gflags::gflags
    pthread
    spdlog
    core
    infer
    infer_cu
    module
    module_logger 
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
