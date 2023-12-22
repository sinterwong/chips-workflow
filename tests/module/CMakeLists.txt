CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
PROJECT(Application)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=deprecated-declarations -Wno-error=unused-function")
INCLUDE_DIRECTORIES(
    ${PROJECT_INCLUDE_DIR}
    ${PROJECT_INCLUDE_DIR}/utils
    ${PROJECT_INCLUDE_DIR}/module
    ${PROJECT_INCLUDE_DIR}/jetson-utils
    ${PROJECT_INCLUDE_DIR}/infer
    ${PROJECT_INCLUDE_DIR}/video
    ${PROJECT_INCLUDE_DIR}/video/x3
    ${PROJECT_INCLUDE_DIR}/video/jetson
    ${PROJECT_INCLUDE_DIR}/video/generic
    ${PROJECT_INCLUDE_DIR}/infer/x3
    ${PROJECT_INCLUDE_DIR}/infer/jetson
    ${PROJECT_INCLUDE_DIR}/infer/generic
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${3RDPARTY_DIR}/gflags/include
    ${CURL_INCLUDE_DIRS}
)

LINK_DIRECTORIES(
    ${OpenCV_LIBRARY_DIR}
    ${OpenCV_LIBRARY_DIR}
    ${X3_DIRECTORIES}
)
IF(TARGET_HARDWARE STREQUAL "x3")
    MESSAGE(STATUS "Loading X3 dependent libraries")
    SET(CHIPS_DEPAND_LIBS ${X3_LIBS})
ELSEIF(TARGET_HARDWARE STREQUAL "jetson")
    MESSAGE(STATUS "Loading Jetson dependent libraries")
    SET(CHIPS_DEPAND_LIBS jetson-utils)
ELSE()
    SET(CHIPS_DEPAND_LIBS ${ONNXRUNTIME_LIBS})
ENDIF()

LINK_LIBRARIES(
    ${OpenCV_LIBS}
    ${FFMPEG_LIBS}
    ${CURL_LIBRARIES}
    gflags
    pthread
    core
    video
    infer
    module
    module_logger 
    stdc++fs
)

SET(LINK "-Wl,--no-as-needed")
FILE(MAKE_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin/module)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin/module)
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