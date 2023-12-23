CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
PROJECT(Application)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lz")
INCLUDE_DIRECTORIES(
    ${PROJECT_INCLUDE_DIR}
    ${PROJECT_INCLUDE_DIR}/utils
    ${PROJECT_INCLUDE_DIR}/video
    ${PROJECT_INCLUDE_DIR}/jetson-utils
    ${PROJECT_INCLUDE_DIR}/video/x3
    ${PROJECT_INCLUDE_DIR}/video/jetson
    ${PROJECT_INCLUDE_DIR}/video/generic
    ${3RDPARTY_DIR}/gflags/include
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${FFMPEG_INCLUDE_DIR}
)

LINK_DIRECTORIES(
    ${OpenCV_LIBRARY_DIR}
    ${X3_DIRECTORIES}
)

SET(CHIPS_DEPAND_LIBS)

IF(TARGET_HARDWARE STREQUAL "x3")
    MESSAGE(STATUS "Loading X3 dependent libraries")
    SET(CHIPS_DEPAND_LIBS ${X3_LIBS})
ELSEIF(TARGET_HARDWARE STREQUAL "generic")
    MESSAGE(STATUS "Loading generic dependent libraries")
    SET(CHIPS_DEPAND_LIBS aom x264 x265 vpx m z)
ENDIF()

LINK_LIBRARIES(
    ${OpenCV_LIBS}
    ${FFMPEG_LIBS}
    ${OPENSSL_LIBS}
    ${CURL_LIBRARIES}
    ${CHIPS_DEPAND_LIBS}
    gflags
    pthread
    module 
    video 
    infer 
    license 
    module_logger 
    stdc++fs

)
SET(LINK "-Wl,--no-as-needed")
FILE(MAKE_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin/video)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin/video)
FILE(GLOB APP_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
MESSAGE(STATUS "APP SOURCES: ${APP_SOURCES}")
FOREACH(sourcefile ${APP_SOURCES})
    STRING(REGEX MATCH "[^/]+$" sourcefilewithoutpath ${sourcefile})
    STRING(REPLACE ".cpp" "" testname ${sourcefilewithoutpath})
    ADD_EXECUTABLE( ${testname} ${sourcefile})

    INSTALL(TARGETS ${testname}
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)
ENDFOREACH(sourcefile ${APP_SOURCES})