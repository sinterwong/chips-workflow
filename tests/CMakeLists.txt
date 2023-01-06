CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
PROJECT(Tests)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=deprecated-declarations -Wno-error=unused-function")
INCLUDE_DIRECTORIES(
    ${PROJECT_INCLUDE_DIR}
    ${PROJECT_INCLUDE_DIR}/core
    ${PROJECT_INCLUDE_DIR}/infer
    ${PROJECT_INCLUDE_DIR}/module
    ${PROJECT_INCLUDE_DIR}/jetson-utils
    ${OpenCV_INCLUDE_DIRS}
    ${Protobuf_INCLUDE_DIR}
    ${OPENSSL_INCLUDE_DIR}
    ${CURL_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${X3_INCLUDE}
)

LINK_DIRECTORIES(
    ${OpenCV_LIBRARY_DIR}
    ${CURL_LIBRARY_DIR}
    ${OPENSSL_LIBRARY_DIR}
    ${X3_DIRECTORIES}
)
SET(FE_LIBS core license module module_logger infer utils)
IF(TARGET_HARDWARE STREQUAL "x3")
    MESSAGE(STATUS "Loading X3 dependent libraries")
    SET(CHIPS_DEPAND_LIBS ${X3_LIBS})
ELSEIF(TARGET_HARDWARE STREQUAL "jetson")
    MESSAGE(STATUS "Loading Jetson dependent libraries")
    SET(CHIPS_DEPAND_LIBS jetson_infer_cu jetson-utils)
ELSE()
    SET(CHIPS_DEPAND_LIBS)
ENDIF()

LINK_LIBRARIES(
    ${OpenCV_LIBS}
    ${Protobuf_LIBRARY}
    ${CURL_LIBS}
    ${OPENSSL_LIBS}
    ${FE_LIBS}
    gflags::gflags
    pthread
    spdlog
    stdc++fs
)
SET(LINK "-Wl,--no-as-needed")
FILE(GLOB APP_SOURCES *.cpp)
FILE(GLOB APP_HARDWARE_SOURCES ${TARGET_HARDWARE}/*.cpp)
LIST(APPEND APP_SOURCES ${APP_HARDWARE_SOURCES})
MESSAGE(STATUS "APP SOURCES: ${APP_SOURCES}")
FOREACH(sourcefile ${APP_SOURCES})
    STRING(REGEX MATCH "[^/]+$" sourcefilewithoutpath ${sourcefile})
    STRING(REPLACE ".cpp" "" testname ${sourcefilewithoutpath})
    ADD_EXECUTABLE( ${testname} ${sourcefile})
    TARGET_LINK_LIBRARIES(${testname} PRIVATE ${LINK} core module_logger)
    SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin/tests)

    INSTALL(TARGETS ${testname}
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)
ENDFOREACH(sourcefile ${APP_SOURCES})
